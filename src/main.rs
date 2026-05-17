use inferaived::language_model::{LayerType, Qwen35Config, Qwen35Model, Qwen35Session};
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
    let model = Qwen35Model::new(&device, &queue, &tensors, &config);
    info!("Model constructed");

    // Small max_seq_len is enough for this smoke test (prompt + a few
    // generated tokens). Bump this up for longer generation or multi-turn
    // chat.
    let max_seq_len = 32;
    let num_generated = 5;
    let mut session = Qwen35Session::new(&model, &device, max_seq_len);

    // Cold prefill of the prompt; the top candidates are for the FIRST
    // token immediately after the prompt.
    let prompt_top = session
        .forward(&device, &queue, encoded.get_ids(), 5)
        .await;
    let first_token = prompt_top[0].0 as u32;
    println!(
        "Prefill picked token {}: {:?} (top 5: {:?})",
        first_token,
        tokenizer
            .decode(&[first_token], false)
            .unwrap_or_default(),
        prompt_top,
    );

    // Decode loop: feed the previously-sampled token back into the session
    // one at a time and take the greedy pick each step.
    let mut generated: Vec<u32> = vec![first_token];
    let mut next_input = first_token;
    for _ in 0..(num_generated - 1) {
        let top = session.forward(&device, &queue, &[next_input], 5).await;
        let tok = top[0].0 as u32;
        println!(
            "Step picked token {}: {:?} (top 5: {:?})",
            tok,
            tokenizer.decode(&[tok], false).unwrap_or_default(),
            top,
        );
        generated.push(tok);
        next_input = tok;
    }

    let generated_text = tokenizer
        .decode(&generated, false)
        .expect("Failed to decode generated tokens");
    println!("Prompt + generated: {:?}{:?}", prompt, generated_text);
}
