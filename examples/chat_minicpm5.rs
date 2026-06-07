//! Multi-turn REPL chat using the MiniCPM5 Jinja chat template.
//!
//! Run with:
//!     cargo run --release --example chat_minicpm5
//!
//! Commands at the `User>` prompt:
//!     /reset   clear conversation history (keeps the system prompt)
//!     /exit    quit
//!
//! Env vars:
//!     CHAT_SYSTEM_PROMPT           override the default system prompt
//!     CHAT_MAX_NEW_TOKENS          cap per-turn generation (default 512)
//!     CHAT_MAX_CONTEXT_TOKENS      session KV-cache budget (default 8192)
//!     CHAT_DEBUG_RENDER=1          log the rendered template before each turn
//!
//! ## Token-stream multi-turn (same shape as `examples/chat.rs`)
//!
//! The MiniCPM5 chat template is structurally identical to Qwen's:
//! `<|im_start|>{role}\n{content}<|im_end|>\n` per turn, with an
//! `add_generation_prompt`-controlled `<|im_start|>assistant\n
//! <think>\n\n</think>\n\n` tail when `enable_thinking=false`. It also
//! strips `<think>` from past assistant turns when re-rendering, so the
//! same "render once, append per-turn deltas" trick that `chat.rs` uses
//! applies here verbatim.
//!
//! Key differences from `chat.rs`:
//!   - Template prefixes `bos_token` (`<s>`, id 0). Qwen's didn't.
//!   - EOS pair is `</s>` (id 1) + `<|im_end|>` (id 130073).
//!   - Weight prefix is `model.*` (not `model.language_model.*`) and the
//!     safetensors shard is `model-00000-of-00001.safetensors`.
//!   - LM head is untied — handled inside `MiniCPM5ModelCore`.

use std::io::{BufRead, Write};

use inferaived::language_model::{MiniCPM5Config, MiniCPM5GpuModel, MiniCPM5GpuSession};
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

const MODEL_SAFETENSORS: &str = "model/MiniCPM5-1B/model-00000-of-00001.safetensors";
const MODEL_TOKENIZER: &str = "model/MiniCPM5-1B/tokenizer.json";
const MODEL_CONFIG: &str = "model/MiniCPM5-1B/config.json";
const MODEL_CHAT_TEMPLATE: &str = "model/MiniCPM5-1B/chat_template.jinja";

const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful assistant.";
const DEFAULT_MAX_NEW_TOKENS: usize = 512;
const DEFAULT_MAX_CONTEXT_TOKENS: usize = 8192;

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

/// The literal turn-boundary string the MiniCPM5 chat template emits for
/// a new user/assistant pair when `add_generation_prompt=true` and
/// `enable_thinking=false`. Identical to Qwen3.5's because both use the
/// `<|im_start|>` / `<|im_end|>` framing and the same `<think>` tail.
fn render_turn_delta(user_input: &str) -> String {
    format!(
        "<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
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
    let config = MiniCPM5Config::from_json_file(MODEL_CONFIG).expect("Failed to load model config");
    let chat_template_src =
        std::fs::read_to_string(MODEL_CHAT_TEMPLATE).expect("Failed to read chat_template.jinja");

    // -- jinja env --
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

    let model = MiniCPM5GpuModel::new(&device, &queue, &tensors, &config);
    info!("Model constructed");

    // EOS: chat-tuned `<|im_end|>` (id 130073) + base `</s>` (id 1).
    // Match by token string so the example stays correct if IDs shift.
    let eos_ids: Vec<u32> = ["<|im_end|>", "</s>"]
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
    let max_context_tokens = std::env::var("CHAT_MAX_CONTEXT_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_CONTEXT_TOKENS);
    let debug_render = std::env::var("CHAT_DEBUG_RENDER")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false);

    let mut session = model.new_session(&device, &queue, max_context_tokens);
    let params = SamplingParams::default();

    let mut messages: Vec<Message> = vec![Message {
        role: "system".into(),
        content: system_prompt.clone(),
    }];

    println!("\nChat with MiniCPM5-1B. Commands: /reset, /exit. Ctrl-D / Ctrl-Z+Enter also exits.");

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
                session.reset(&device, &queue);
                messages.truncate(1);
                println!("(history reset)");
                continue;
            }
            _ => {}
        }

        // First turn → full template render (which includes the BOS).
        // Subsequent turns → append-only delta. The template strips
        // `<think>` from past assistant turns, so a re-render would
        // diverge from what the KV cache holds.
        let prompt = if session.position() == 0 {
            messages.push(Message {
                role: "user".into(),
                content: user_input.to_string(),
            });
            let tmpl = env.get_template("chat").unwrap();
            // `bos_token` is referenced as a bare name (`{{- bos_token }}`),
            // so we must inject it via the context — minijinja doesn't
            // have HF's tokenizer-provided global.
            match tmpl.render(context! {
                messages => &messages,
                bos_token => "<s>",
                add_generation_prompt => true,
                enable_thinking => false,
            }) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("template render error: {e}");
                    messages.pop();
                    continue;
                }
            }
        } else {
            messages.push(Message {
                role: "user".into(),
                content: user_input.to_string(),
            });
            render_turn_delta(user_input)
        };

        if debug_render {
            debug!("turn input:\n{prompt}");
        }

        let encoded = tokenizer
            .encode(prompt.as_str(), false)
            .expect("Failed to tokenize prompt");
        let prompt_ids = encoded.get_ids();

        if session.position() + prompt_ids.len() + max_new_tokens > session.max_seq_len() {
            eprintln!(
                "(context budget exhausted: position={} + prompt={} + max_new={} > max_context={}; \
                 use /reset or raise CHAT_MAX_CONTEXT_TOKENS)",
                session.position(),
                prompt_ids.len(),
                max_new_tokens,
                session.max_seq_len()
            );
            messages.pop();
            continue;
        }

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
