//! Multi-turn REPL chat using the Qwen3.5 Jinja chat template.
//!
//! Run with:
//!     cargo run --release --example chat
//!
//! Commands at the `User>` prompt:
//!     /reset   clear conversation history (keeps the system prompt)
//!     /exit    quit
//!
//! Env vars:
//!     CHAT_SYSTEM_PROMPT      override the default system prompt
//!     CHAT_MAX_NEW_TOKENS     cap per-turn generation (default 512)
//!     CHAT_DEBUG_RENDER=1     log the rendered template before each turn
//!
//! Design note: a fresh `Qwen35GpuSession` is built per turn rather than
//! extending one across turns. The Qwen chat template strips past
//! `<think>` blocks from prior assistant turns, so the rendered turn-N
//! prompt does not line up with what an across-turn session's KV cache
//! has already seen at the end of turn N-1. Delta-based reuse would
//! corrupt context. Per-turn rebuild is the honest baseline; smarter
//! cross-turn reuse is a separate optimization.

use std::io::{BufRead, Write};

use inferaived::language_model::{Qwen35Config, Qwen35GpuModel, Qwen35GpuSession};
use inferaived::sampling::{SamplingParams, StoppingCriteria};
use minijinja::value::Value;
use minijinja::{Environment, ErrorKind, context};
use minijinja_contrib::pycompat::unknown_method_callback;
use safetensors::SafeTensors;
use serde::Serialize;
use tokenizers::Tokenizer;
use tokio_stream::StreamExt;
use tracing::{debug, info};
use wgpu::{
    BackendOptions, Backends, DeviceDescriptor, ExperimentalFeatures, Features, Instance,
    InstanceDescriptor, InstanceFlags, MemoryBudgetThresholds, MemoryHints, PowerPreference,
    RequestAdapterOptions, Trace,
};

const MODEL_SAFETENSORS: &str = "model/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors";
const MODEL_TOKENIZER: &str = "model/Qwen3.5-0.8B/tokenizer.json";
const MODEL_CONFIG: &str = "model/Qwen3.5-0.8B/config.json";
const MODEL_CHAT_TEMPLATE: &str = "model/Qwen3.5-0.8B/chat_template.jinja";

const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful assistant.";
const DEFAULT_MAX_NEW_TOKENS: usize = 512;

#[derive(Serialize, Clone)]
struct Message {
    role: String,
    content: String,
}

fn features(supported: Features) -> Features {
    let mut required = Features::empty();
    if supported.contains(Features::SHADER_F16) {
        required |= Features::SHADER_F16;
    }
    if supported.contains(Features::TIMESTAMP_QUERY) {
        required |= Features::TIMESTAMP_QUERY;
    }
    if supported.contains(Features::SUBGROUP) {
        required |= Features::SUBGROUP;
    }
    if supported.contains(Features::SUBGROUP_BARRIER) {
        required |= Features::SUBGROUP_BARRIER;
    }
    if supported.contains(Features::SHADER_FLOAT32_ATOMIC) {
        required |= Features::SHADER_FLOAT32_ATOMIC;
    }
    required
}

/// HF chat templates call a Python global `raise_exception(...)` to fail
/// fast on malformed inputs. minijinja doesn't ship it; surface it as a
/// template error so the caller sees a meaningful message.
fn raise_exception(msg: String) -> Result<Value, minijinja::Error> {
    Err(minijinja::Error::new(ErrorKind::InvalidOperation, msg))
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // -- load model assets --
    let buffer = std::fs::read(MODEL_SAFETENSORS).expect("Failed to read safetensors");
    let tensors = SafeTensors::deserialize(&buffer[..]).expect("Failed to deserialize tensors");
    let tokenizer = Tokenizer::from_file(MODEL_TOKENIZER).expect("Failed to load tokenizer");
    let config = Qwen35Config::from_json_file(MODEL_CONFIG).expect("Failed to load model config");
    let chat_template_src =
        std::fs::read_to_string(MODEL_CHAT_TEMPLATE).expect("Failed to read chat_template.jinja");

    // -- jinja env --
    // The Qwen template uses Python string methods (`.startswith`, `.split`,
    // `.rstrip`, `.lstrip`) and `dict.items()`. `pycompat`'s unknown-method
    // callback routes those through a Python-compatible implementation.
    let mut env = Environment::new();
    env.set_unknown_method_callback(unknown_method_callback);
    env.add_function("raise_exception", raise_exception);
    env.add_template("chat", &chat_template_src)
        .expect("Failed to compile chat_template.jinja");

    // -- wgpu init --
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::PRIMARY,
        flags: InstanceFlags::default(),
        memory_budget_thresholds: MemoryBudgetThresholds::default(),
        backend_options: BackendOptions::default(),
        display: None,
    });
    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .expect("Failed to request adapter");
    let (device, queue) = adapter
        .request_device(&DeviceDescriptor {
            label: None,
            required_features: features(adapter.features()),
            required_limits: adapter.limits(),
            experimental_features: ExperimentalFeatures::default(),
            memory_hints: MemoryHints::Performance,
            trace: Trace::default(),
        })
        .await
        .expect("Failed to request device");
    info!(
        "Device ready: backend={:?} name={:?}",
        adapter.get_info().backend,
        adapter.get_info().name
    );

    let model = Qwen35GpuModel::new(&device, &queue, &tensors, &config.text_config);
    info!("Model constructed");

    // Stop on chat-tuned EOS (`<|im_end|>`) or base-model EOS
    // (`<|endoftext|>`); the latter is a safety net in case the chat
    // tuning regresses.
    let eos_ids: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|s| tokenizer.token_to_id(s))
        .collect();
    let stopping = [StoppingCriteria::Eos(eos_ids.clone())];

    let system_prompt =
        std::env::var("CHAT_SYSTEM_PROMPT").unwrap_or_else(|_| DEFAULT_SYSTEM_PROMPT.to_string());
    let max_new_tokens = std::env::var("CHAT_MAX_NEW_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_NEW_TOKENS);
    let debug_render = std::env::var("CHAT_DEBUG_RENDER")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false);

    let mut messages: Vec<Message> = vec![Message {
        role: "system".into(),
        content: system_prompt,
    }];

    println!("\nChat with Qwen3.5-0.8B. Commands: /reset, /exit. Ctrl-D / Ctrl-Z+Enter also exits.");

    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    loop {
        write!(stdout, "\nUser> ").ok();
        stdout.flush().ok();

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(e) => {
                eprintln!("stdin error: {e}");
                break;
            }
        }
        let user_input = line.trim();
        if user_input.is_empty() {
            continue;
        }
        match user_input {
            "/exit" | "/quit" => break,
            "/reset" => {
                messages.truncate(1);
                println!("(history reset)");
                continue;
            }
            _ => {}
        }
        messages.push(Message {
            role: "user".into(),
            content: user_input.to_string(),
        });

        // Render the entire conversation through the chat template.
        let tmpl = env.get_template("chat").unwrap();
        let rendered = match tmpl.render(context! {
            messages => &messages,
            add_generation_prompt => true,
            // `false` makes the template emit an empty `<think></think>` so
            // the model can produce the answer directly. Flip to true for
            // self-thinking; the response stream will then start with the
            // thinking trace before `</think>`.
            enable_thinking => false,
        }) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("template render error: {e}");
                messages.pop();
                continue;
            }
        };
        if debug_render {
            debug!("rendered prompt:\n{rendered}");
        }

        let encoded = tokenizer
            .encode(rendered.as_str(), false)
            .expect("Failed to tokenize prompt");
        let prompt_ids = encoded.get_ids();
        let max_seq_len = prompt_ids.len() + max_new_tokens;

        let mut session = Qwen35GpuSession::new(&model, &device, &queue, max_seq_len);
        let params = SamplingParams::default();

        write!(stdout, "Assistant> ").ok();
        stdout.flush().ok();
        let stream = session.generate(
            &device,
            &queue,
            prompt_ids,
            &params,
            max_new_tokens,
            &stopping,
        );
        tokio::pin!(stream);
        let mut response = String::new();
        while let Some(tok) = stream.next().await {
            if eos_ids.contains(&tok.id) {
                break;
            }
            let piece = tokenizer.decode(&[tok.id], false).unwrap_or_default();
            response.push_str(&piece);
            write!(stdout, "{piece}").ok();
            stdout.flush().ok();
        }
        writeln!(stdout).ok();

        messages.push(Message {
            role: "assistant".into(),
            content: response,
        });
    }

    println!("\nGoodbye.");
}
