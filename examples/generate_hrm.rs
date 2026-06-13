//! Greedy completion with HRM-Text-1B (Sapient's hierarchical-reasoning model).
//!
//! Run with:
//!     cargo run --release --example generate_hrm
//!
//! Env vars:
//!     HRM_PROMPT             user text (default: "Explain why the sky is blue.")
//!     HRM_MAX_NEW_TOKENS     generation cap (default 256)
//!
//! ## Prompt format
//!
//! HRM-Text-1B is a pre-alignment base checkpoint, not a chat model. The card
//! recommends wrapping the prompt in an `<|im_start|> … <|im_end|>` envelope
//! with a leading *condition* prefix. We use the composite `synth,cot`
//! condition (`<|quad_end|><|object_ref_end|>`), which elicits a step-by-step
//! style for reasoning prompts.
//!
//! ## Caveats (this engine)
//!
//!   * Attention is **causal only**. HRM-Text trained with a PrefixLM mask
//!     (bidirectional prompt, causal completion); matching it is a follow-up.
//!     Causal is the documented fallback — coherent but off-distribution.
//!   * Greedy (argmax) decoding only.
//!   * Not yet numerically verified against the HF reference.

use std::io::Write;

use inferaived::language_model::{HrmTextConfig, HrmTextGpuModel, HrmTextGpuSession};
use inferaived::sampling::{SamplingParams, StoppingCriteria};
use safetensors::SafeTensors;
use tokenizers::Tokenizer;
use tokio_stream::StreamExt;
use tracing::{debug, info};
use wgpu::{
    BackendOptions, Backends, DeviceDescriptor, ExperimentalFeatures, Features, Instance,
    InstanceDescriptor, InstanceFlags, MemoryBudgetThresholds, MemoryHints, PowerPreference,
    RequestAdapterOptions, Trace,
};

const MODEL_SAFETENSORS: &str = "model/HRM-Text-1B/model.safetensors";
const MODEL_TOKENIZER: &str = "model/HRM-Text-1B/tokenizer.json";
const MODEL_CONFIG: &str = "model/HRM-Text-1B/config.json";

/// `synth,cot` composite condition prefix (see the model card): synth
/// (`<|quad_end|>`) then cot (`<|object_ref_end|>`).
const CONDITION: &str = "<|quad_end|><|object_ref_end|>";
const DEFAULT_PROMPT: &str = "Explain why the sky is blue.";
const DEFAULT_MAX_NEW_TOKENS: usize = 256;

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

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let buffer = std::fs::read(MODEL_SAFETENSORS).expect("Failed to read safetensors");
    let tensors = SafeTensors::deserialize(&buffer[..]).expect("Failed to deserialize tensors");
    let tokenizer = Tokenizer::from_file(MODEL_TOKENIZER).expect("Failed to load tokenizer");
    let config = HrmTextConfig::from_json_file(MODEL_CONFIG).expect("Failed to load model config");

    let user_prompt = std::env::var("HRM_PROMPT").unwrap_or_else(|_| DEFAULT_PROMPT.to_string());
    let max_new_tokens = std::env::var("HRM_MAX_NEW_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_NEW_TOKENS);

    let prompt = format!("<|im_start|>{CONDITION}{user_prompt}<|im_end|>");
    let encoded = tokenizer
        .encode(prompt.as_str(), false)
        .expect("Failed to encode input");
    info!("Prompt: {:?}", prompt);
    info!("Encoded {} tokens", encoded.get_ids().len());

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
    debug!("Adapter: {:?}", adapter.get_info());
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

    let model = HrmTextGpuModel::new(&device, &queue, &tensors, &config);
    info!("Model constructed");

    let max_seq_len = encoded.get_ids().len() + max_new_tokens;
    let mut session = HrmTextGpuSession::new(&model, &device, &queue, max_seq_len);
    let params = SamplingParams::default();

    // EOS: config `eos_token_id` (id 11) plus `<|im_end|>` by string.
    let mut eos_ids: Vec<u32> = vec![config.eos_token_id];
    if let Some(id) = tokenizer.token_to_id("<|im_end|>") {
        if !eos_ids.contains(&id) {
            eos_ids.push(id);
        }
    }
    let stopping = [StoppingCriteria::Eos(eos_ids)];

    print!("{user_prompt}");
    std::io::stdout().flush().ok();
    let stream = session.generate(
        &device,
        &queue,
        encoded.get_ids(),
        &params,
        max_new_tokens,
        &stopping,
    );
    tokio::pin!(stream);
    let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
    while let Some(tok) = stream.next().await {
        let piece = tokenizer.decode(&[tok.id], false).unwrap_or_default();
        print!("{}", piece);
        std::io::stdout().flush().ok();
        generated.push(tok.id);
    }
    println!();

    let hit_eos = generated
        .last()
        .map(|id| matches!(&stopping[0], StoppingCriteria::Eos(eos) if eos.contains(id)))
        .unwrap_or(false);
    println!(
        "--- generated {} tokens (stopped on {}) ---",
        generated.len(),
        if hit_eos { "EOS" } else { "max_tokens" },
    );
}
