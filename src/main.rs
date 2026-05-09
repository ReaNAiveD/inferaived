use inferaived::{
    embedding_lookup::EmbeddingLookupCpu,
    layer_loop::{LinearLayer, LinearLayerConfig},
    log_tensor,
};
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

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // Should extract from config file in the future
    let hidden_size = 1024usize;
    let linear_num_key_heads = 16usize;
    let linear_key_head_dim = 128usize;
    let linear_num_value_heads = 16usize;
    let linear_value_head_dim = 128usize;
    let rope_theta = 10_000_000f32;
    let partial_rotary_factor = 0.25f32;
    let buffer = std::fs::read("model/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors")
        .expect("Failed to read file");
    let tensors = SafeTensors::deserialize(&buffer[..]).expect("Failed to deserialize tensors");
    let embeddings = tensors
        .tensor("model.language_model.embed_tokens.weight")
        .expect("Failed to get tensor: model.language_model.embed_tokens.weight");
    log_tensor("model.language_model.embed_tokens.weight", &embeddings);
    let tokenizer = Tokenizer::from_file("model/Qwen3.5-0.8B/tokenizer.json")
        .expect("Failed to load tokenizer");
    let encoded = tokenizer
        .encode("你好，Hello World", false)
        .expect("Failed to encode input");
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

    // CPU-based embedding lookup (avoids OOM on limited VRAM GPUs)
    let embedding_lookup = EmbeddingLookupCpu::new(embeddings, hidden_size);
    let result = embedding_lookup.lookup(&encoded);
    let embeddings = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("embeddings_buffer"),
        size: (result.len() * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&embeddings, 0, bytemuck::cast_slice(&result));

    let seq_len = encoded.get_ids().len();
    let config = LinearLayerConfig {
        linear_num_key_heads,
        linear_num_value_heads,
        linear_key_head_dim,
        linear_value_head_dim,
    };

    // Layer 0 (linear attention)
    let layer0 = LinearLayer::new(
        &device,
        &queue,
        &tensors,
        "model.language_model.layers.0".to_string(),
        hidden_size,
        &config,
        seq_len,
    );
    layer0.compute(&device, &queue, &embeddings, seq_len);
    // embeddings now holds the final output of layer 0
}
