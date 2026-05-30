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
//!     CHAT_SYSTEM_PROMPT           override the default system prompt
//!     CHAT_MAX_NEW_TOKENS          cap per-turn generation (default 512)
//!     CHAT_MAX_CONTEXT_TOKENS      session KV-cache budget (default 8192)
//!     CHAT_DEBUG_RENDER=1          log the rendered template before each turn
//!
//! ## Why token-stream instead of "render each turn from scratch"
//!
//! The Qwen chat template strips `<think>` blocks from past assistant
//! turns when re-rendering. So a re-render-each-turn loop would have to
//! discard the entire KV cache between turns (because what the cache
//! holds for turn N-1 doesn't line up with what the new render of past
//! turns claims happened). Instead, we render the template **only on
//! the first turn (or right after `/reset`)**, then append per-turn
//! deltas — `<|im_end|>\n<|im_start|>user\n{u}<|im_end|>\n
//! <|im_start|>assistant\n<think>\n\n</think>\n\n` — verbatim. This is
//! how production engines (vLLM, llama.cpp, TGI) all handle multi-turn
//! chat: the token stream is the source of truth, not the message list.
//!
//! `Qwen35GpuSession::reset(&device, &queue)` exists for the `/reset`
//! command — it zeros the recurrent state in every linear-attention
//! layer (full-attention KV cache is implicitly invalidated by setting
//! `position = 0`). See `Qwen35GpuSession::reset` docs for why
//! "rewind to position N > 0" is not supported on this hybrid stack.

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

/// The literal turn-boundary string the chat template emits for a new
/// user/assistant pair when `add_generation_prompt=true` and
/// `enable_thinking=false`. Used to build per-turn deltas without
/// re-rendering the whole conversation.
///
/// Built from the template's literal output:
///   - close-out of the previous assistant turn: `<|im_end|>\n`
///   - new user message: `<|im_start|>user\n{u}<|im_end|>\n`
///   - generation prompt: `<|im_start|>assistant\n<think>\n\n</think>\n\n`
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
    let max_context_tokens = std::env::var("CHAT_MAX_CONTEXT_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_CONTEXT_TOKENS);
    let debug_render = std::env::var("CHAT_DEBUG_RENDER")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false);

    // One session for the entire process lifetime. Sized to a chat
    // budget rather than the model's full max_position_embeddings
    // (262,144 here) — the latter would cost ~12 GB just for KV cache.
    let mut session = Qwen35GpuSession::new(&model, &device, &queue, max_context_tokens);
    let params = SamplingParams::default();

    let mut messages: Vec<Message> = vec![Message {
        role: "system".into(),
        content: system_prompt.clone(),
    }];

    println!(
        "\nChat with Qwen3.5-0.8B. Commands: /reset, /exit. Ctrl-D / Ctrl-Z+Enter also exits."
    );

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

        // First turn (session is empty) → render full template. Subsequent
        // turns → tokenize just the user/assistant delta and append. This
        // avoids the template's past-assistant-turn `<think>` stripping
        // which would otherwise invalidate the KV cache.
        let prompt = if session.position() == 0 {
            messages.push(Message {
                role: "user".into(),
                content: user_input.to_string(),
            });
            let tmpl = env.get_template("chat").unwrap();
            match tmpl.render(context! {
                messages => &messages,
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

        // Bound check before we feed the delta. If this turn would
        // exceed the session budget, force a reset so the user gets a
        // useful error rather than a panic deep in the kernel.
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
