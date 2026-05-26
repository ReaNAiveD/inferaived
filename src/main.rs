use inferaived::language_model::{LayerType, Qwen35Config, Qwen35GpuModel, Qwen35GpuSession};
use inferaived::sampling::SamplingParams;
use safetensors::SafeTensors;
use tokenizers::Tokenizer;
use tokio;
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

/// Hardcoded Qwen 3.5 0.8B model config. TODO: load from `config.json`.
fn qwen35_0_8b_config() -> Qwen35Config {
    // 24 layers; full attention every 4th layer (indices 3, 7, 11, 15, 19, 23).
    let layer_types: Vec<LayerType> = (0..24)
        .map(|i| {
            if (i + 1) % 4 == 0 {
                LayerType::Full
            } else {
                LayerType::Linear
            }
        })
        .collect();
    Qwen35Config {
        hidden_size: 1024,
        layer_types,
        num_attention_heads: 8,
        num_key_value_heads: 2,
        head_dim: 256,
        rope_theta: 10_000_000.0,
        partial_rotary_factor: 0.25,
        linear_num_key_heads: 16,
        linear_num_value_heads: 16,
        linear_key_head_dim: 128,
        linear_value_head_dim: 128,
        intermediate_size: 3584,
    }
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

    let config = qwen35_0_8b_config();
    let model = Qwen35GpuModel::new(&device, &queue, &tensors, &config);
    info!("Model constructed");

    // Small max_seq_len is enough for this smoke test (prompt + a few
    // generated tokens). Bump this up for longer generation or multi-turn
    // chat.
    let max_seq_len = 32;
    let num_generated = 5;
    let mut session = Qwen35GpuSession::new(&model, &device, &queue, max_seq_len);
    let params = SamplingParams::default();

    // One call drives prefill + the whole decode loop. No EOS stop here
    // (the base model isn't instruct-tuned); we cap by `num_generated`.
    let sampled = session
        .generate(
            &device,
            &queue,
            encoded.get_ids(),
            &params,
            num_generated,
            &[],
        )
        .await;
    for (i, tok) in sampled.iter().enumerate() {
        let label = if i == 0 { "Prefill" } else { "Step" };
        println!(
            "{} picked token {}: {:?} (logprob {:.4})",
            label,
            tok.id,
            tokenizer.decode(&[tok.id], false).unwrap_or_default(),
            tok.logprob,
        );
    }
    let generated: Vec<u32> = sampled.iter().map(|t| t.id).collect();

    let generated_text = tokenizer
        .decode(&generated, false)
        .expect("Failed to decode generated tokens");
    println!("Prompt + generated: {:?}{:?}", prompt, generated_text);
}
