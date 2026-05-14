use inferaived::language_model::{LayerType, Qwen35Config, Qwen35Model};
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
    let prompt = "Hello ";
    let encoded = tokenizer
        .encode(prompt, false)
        .expect("Failed to encode input");
    info!("Prompt: {:?}", prompt);
    info!("Encoded IDs: {:?}", encoded.get_ids());

    let instance = Instance::new(&InstanceDescriptor {
        backends: Backends::PRIMARY,
        flags: InstanceFlags::default(),
        memory_budget_thresholds: MemoryBudgetThresholds::default(),
        backend_options: BackendOptions::default(),
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
    let model = Qwen35Model::new(&device, &queue, &tensors, &config);
    info!("Model constructed");

    let top_candidates = model.compute(&device, &queue, encoded.get_ids(), 5).await;
    let next_token_id = top_candidates[0].0 as u32;
    let next_token_text = tokenizer
        .decode(&[next_token_id], false)
        .expect("Failed to decode next token");
    println!(
        "Next token id: {}, text: {:?}, top candidates: {:?}",
        next_token_id, next_token_text, top_candidates
    );
}
