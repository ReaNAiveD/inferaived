//! Interactive REPL over a precompiled parallel-context MiniCPM5 session.
//!
//! Run with:
//!     cargo run --release --example parallel_minicpm5 -- \
//!         --prefix "You answer using only the provided facts." \
//!         --context "Fact A: The Eiffel Tower is in Paris." \
//!         --context "Fact B: The Colosseum is in Rome."
//!
//! The `--prefix` text is rendered through MiniCPM5's jinja chat template as
//! the **system** message (with BOS) and becomes the always-visible prefix
//! tokens of the parallel context namespace. Each `--context` text is rendered
//! as an independent user-role chunk (no BOS) and registered as one context
//! window; the windows are encoded once into the shared KV cache.
//!
//! A generation session is opened over the context with all windows visible.
//! Commands:
//!   - `/exit` (Ctrl-D / Ctrl-Z+Enter) — quit.
//!   - `/reset` — drop the session and re-open with no contexts visible.
//!   - `/reset 0 2` — drop the session and re-open with only contexts `0`
//!     and `2` visible (indices match `--context` declaration order).

use std::io::{BufRead, Write};

use clap::Parser;
use inferaived::language_model::{MiniCPM5Config, MiniCPM5GpuModel};
use inferaived::sampling::{SamplingParams, StoppingCriteria};
use minijinja::value::Value;
use minijinja::{Environment, ErrorKind, context};
use minijinja_contrib::pycompat::unknown_method_callback;
use safetensors::SafeTensors;
use serde::Serialize;
use tokenizers::Tokenizer;
use tokio_stream::StreamExt;
use tracing::info;
use wgpu::{
    BackendOptions, Backends, DeviceDescriptor, ExperimentalFeatures, Features, Instance,
    InstanceDescriptor, InstanceFlags, MemoryBudgetThresholds, MemoryHints, PowerPreference,
    RequestAdapterOptions, Trace,
};

const MODEL_SAFETENSORS: &str = "model/MiniCPM5-1B/model-00000-of-00001.safetensors";
const MODEL_TOKENIZER: &str = "model/MiniCPM5-1B/tokenizer.json";
const MODEL_CONFIG: &str = "model/MiniCPM5-1B/config.json";
const MODEL_CHAT_TEMPLATE: &str = "model/MiniCPM5-1B/chat_template.jinja";

const BOS: &str = "<s>";

const DEFAULT_PREFIX: &str = "You are a helpful assistant.";
const DEFAULT_MAX_NEW_TOKENS: usize = 256;
const DEFAULT_MAX_SEQ_LEN: usize = 4096;

#[derive(Parser, Debug)]
#[command(about = "Interactive REPL over a parallel-context MiniCPM5 session")]
struct Args {
    /// System prompt rendered ahead of the context blocks. Becomes the
    /// always-visible prefix of the compiled context.
    #[arg(long, default_value = DEFAULT_PREFIX)]
    prefix: String,
    /// Context block to register. Pass multiple times. Each block is rendered
    /// as an independent user-role chunk and assigned an index (in declaration
    /// order) usable with `/reset N..`.
    #[arg(long = "context")]
    contexts: Vec<String>,
    /// Maximum tokens generated per assistant turn.
    #[arg(long, default_value_t = DEFAULT_MAX_NEW_TOKENS)]
    max_new_tokens: usize,
    /// KV cache capacity (per layer). Must exceed prefix + sum(context lengths).
    #[arg(long, default_value_t = DEFAULT_MAX_SEQ_LEN)]
    max_seq_len: usize,
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

fn encode(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
    tokenizer
        .encode(text, false)
        .expect("tokenization failed")
        .get_ids()
        .to_vec()
}

/// HF chat templates call a Python global `raise_exception(...)`; surface it
/// as a template error so the user sees a meaningful message.
fn raise_exception(msg: String) -> Result<Value, minijinja::Error> {
    Err(minijinja::Error::new(ErrorKind::InvalidOperation, msg))
}

#[derive(Serialize)]
struct Message<'a> {
    role: &'a str,
    content: &'a str,
}

/// Render one independent chat chunk via the MiniCPM5 jinja template.
/// `with_bos` keeps the leading `<s>` (true for the very first piece, i.e. the
/// system prefix); `add_generation_prompt` appends the
/// `<|im_start|>assistant\n<think>...` suffix (true for user queries).
fn render_chunk(
    env: &Environment<'_>,
    role: &str,
    content: &str,
    with_bos: bool,
    add_generation_prompt: bool,
) -> String {
    let tmpl = env.get_template("chat").expect("chat template missing");
    let rendered = tmpl
        .render(context! {
            messages => vec![Message { role, content }],
            bos_token => BOS,
            add_generation_prompt => add_generation_prompt,
            enable_thinking => false,
        })
        .expect("chat template render failed");
    if with_bos {
        rendered
    } else {
        rendered
            .strip_prefix(BOS)
            .map(str::to_owned)
            .unwrap_or(rendered)
    }
}

/// Parse a `/reset [idx...]` command body into a visibility list. Empty body
/// → no contexts visible (just the prefix).
fn parse_reset_indices(body: &str, num_contexts: usize) -> Result<Vec<usize>, String> {
    body.split_whitespace()
        .map(|tok| {
            let n: usize = tok
                .parse()
                .map_err(|_| format!("not a non-negative integer: {tok:?}"))?;
            if n >= num_contexts {
                Err(format!(
                    "context index {n} out of range (have {num_contexts})"
                ))
            } else {
                Ok(n)
            }
        })
        .collect()
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

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

    // EOS: chat-tuned `<|im_end|>` + base `</s>`.
    let eos_ids: Vec<u32> = ["<|im_end|>", "</s>"]
        .iter()
        .filter_map(|s| tokenizer.token_to_id(s))
        .collect();
    let stop = StoppingCriteria::Eos(eos_ids.clone());

    // Render prefix (system + BOS) and each context (user, no BOS) through the
    // chat template, then build the parallel-context namespace.
    let prefix = encode(
        &tokenizer,
        &render_chunk(&env, "system", &args.prefix, true, false),
    );
    let num_contexts = args.contexts.len();
    let mut namespace = model.new_context_namespace(args.max_seq_len, &prefix);
    let context_ids: Vec<_> = args
        .contexts
        .iter()
        .map(|ctx| {
            let chunk = render_chunk(&env, "user", ctx, false, false);
            namespace.add_context(&encode(&tokenizer, &chunk))
        })
        .collect();
    let mut compiled = namespace.compile(&device, &queue);
    info!("Compiled context with {} blocks.", num_contexts);

    // Start with all contexts visible.
    let mut visible: Vec<usize> = (0..num_contexts).collect();
    let mut session = {
        let v: Vec<_> = visible.iter().map(|&i| context_ids[i]).collect();
        compiled.begin(&device, &queue, &v)
    };
    let params = SamplingParams::default();

    println!(
        "\nParallel-context REPL over MiniCPM5. {} context block(s) compiled.\n\
         Commands: /reset, /reset <idx..>, /exit. Ctrl-D / Ctrl-Z+Enter also exits.\n\
         Currently visible contexts: {:?}",
        num_contexts, visible,
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
        if matches!(user_input, "/exit" | "/quit") {
            break;
        }
        if let Some(body) = user_input.strip_prefix("/reset") {
            match parse_reset_indices(body.trim(), num_contexts) {
                Ok(new_visible) => {
                    visible = new_visible;
                    let v: Vec<_> = visible.iter().map(|&i| context_ids[i]).collect();
                    // Drop the old session before opening a new one (compiled is &mut-borrowed).
                    drop(session);
                    session = compiled.begin(&device, &queue, &v);
                    println!("(session reset; visible contexts: {visible:?})");
                }
                Err(msg) => eprintln!("/reset: {msg}"),
            }
            continue;
        }

        // Render the user query as an independent chunk (no BOS, with the
        // assistant generation prompt), tokenize, and stream the response.
        let prompt = render_chunk(&env, "user", user_input, false, true);
        let prompt_ids = encode(&tokenizer, &prompt);

        write!(stdout, "Assistant> ").ok();
        stdout.flush().ok();
        let stream = session.generate(
            &device,
            &queue,
            &prompt_ids,
            &params,
            args.max_new_tokens,
            std::slice::from_ref(&stop),
        );
        tokio::pin!(stream);
        while let Some(tok) = stream.next().await {
            if eos_ids.contains(&tok.id) {
                break;
            }
            let piece = tokenizer.decode(&[tok.id], false).unwrap_or_default();
            write!(stdout, "{piece}").ok();
            stdout.flush().ok();
        }
        writeln!(stdout).ok();
    }
}
