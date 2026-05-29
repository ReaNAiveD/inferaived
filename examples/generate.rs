use std::io::Write;

use inferaived::language_model::{Qwen35Config, Qwen35GpuModel, Qwen35GpuSession};
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

    let buffer = std::fs::read("model/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors")
        .expect("Failed to read file");
    let tensors = SafeTensors::deserialize(&buffer[..]).expect("Failed to deserialize tensors");
    let tokenizer = Tokenizer::from_file("model/Qwen3.5-0.8B/tokenizer.json")
        .expect("Failed to load tokenizer");
    let prompt = "Inferaived is a Rust library for running transformer-based language models on consumer GPUs using WebGPU.";
    let encoded = tokenizer
        .encode(prompt, false)
        .expect("Failed to encode input");
    info!("Prompt: {:?}", prompt);
    info!("Encoded IDs: {:?}", encoded.get_ids());

    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::PRIMARY,
        flags: InstanceFlags::default(),
        memory_budget_thresholds: MemoryBudgetThresholds::default(),
        backend_options: BackendOptions::default(),
        display: None,
    });
    debug!("WGPU Instance created successfully: {:?}", instance);
    debug!(
        "Available WGSL features: {:?}",
        instance.wgsl_language_features()
    );
    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .expect("Failed to request adapter");
    debug!("Adapter requested successfully: {:?}", adapter);
    debug!("Available adapter features: {:?}", adapter.features());
    debug!("Adapter limits: {:?}", adapter.limits());
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
    info!("Device requested successfully");

    let config = Qwen35Config::from_json_file("model/Qwen3.5-0.8B/config.json")
        .expect("Failed to load model config");
    let model = Qwen35GpuModel::new(&device, &queue, &tensors, &config.text_config);
    info!("Model constructed");

    // Cap generation at 1000 tokens, or stop early when the model
    // emits any of the configured EOS tokens. `<|im_end|>` is Qwen's
    // chat-tuned EOS; `<|endoftext|>` is the base-model end-of-document
    // marker and is the one most likely to fire for a non-instruct
    // continuation prompt.
    let max_tokens = 1000;
    let max_seq_len = encoded.get_ids().len() + max_tokens;
    let mut session = Qwen35GpuSession::new(&model, &device, &queue, max_seq_len);
    let params = SamplingParams::default();
    let eos_ids: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|s| tokenizer.token_to_id(s))
        .collect();
    let stopping = [StoppingCriteria::Eos(eos_ids)];

    // Stream tokens as they arrive so the user gets visible progress
    // instead of waiting ~30 s for 1000 tokens at ~30 tok/s.
    print!("{}", prompt);
    std::io::stdout().flush().ok();
    let stream = session.generate(
        &device,
        &queue,
        encoded.get_ids(),
        &params,
        max_tokens,
        &stopping,
    );
    tokio::pin!(stream);
    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
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
    let reason = if hit_eos { "EOS" } else { "max_tokens" };
    println!("--- generated {} tokens (stopped on {}) ---", generated.len(), reason);
}
