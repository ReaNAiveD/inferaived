use inferaived::{embedding_lookup::EmbeddingLookupCpu, norm::RmsNormWebgpu};
use safetensors::SafeTensors;
use tokenizers::Tokenizer;
use tokio;
use wgpu::{
    BackendOptions, Backends,
    DeviceDescriptor, ExperimentalFeatures, Features, Instance, InstanceDescriptor, InstanceFlags,
    MemoryBudgetThresholds, MemoryHints,
    PowerPreference, RequestAdapterOptions, Trace,
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
    // Should extract from config file in the future
    let hidden_size = 1024usize;
    let buffer = std::fs::read("model/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors")
        .expect("Failed to read file");
    let tensors = SafeTensors::deserialize(&buffer[..]).expect("Failed to deserialize tensors");
    let embeddings = tensors
        .tensor("model.language_model.embed_tokens.weight")
        .expect("Failed to get tensor");
    println!("Tensors: {:?}", tensors.names());
    let tokenizer = Tokenizer::from_file("model/Qwen3.5-0.8B/tokenizer.json")
        .expect("Failed to load tokenizer");
    let encoded = tokenizer
        .encode("你好，Hello World", false)
        .expect("Failed to encode input");
    println!("Encoded IDs: {:?}", encoded.get_ids());

    let instance = Instance::new(&InstanceDescriptor {
        backends: Backends::PRIMARY,
        flags: InstanceFlags::default(),
        memory_budget_thresholds: MemoryBudgetThresholds::default(),
        backend_options: BackendOptions::default(),
    });
    println!("WGPU Instance created successfully: {:?}", instance);
    println!(
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
    println!("Adapter requested successfully: {:?}", adapter);
    println!("Available adapter features: {:?}", adapter.features());
    println!("Adapter limits: {:?}", adapter.limits());
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
    println!("Device requested successfully: {:?}", device);

    // CPU-based embedding lookup (avoids OOM on limited VRAM GPUs)
    let embedding_lookup = EmbeddingLookupCpu::new(embeddings, hidden_size);
    let result = embedding_lookup.lookup(&encoded);
    println!(
        "Embedding lookup result: {} floats (expected {})",
        result.len(),
        4 * hidden_size
    );
    // Print first few values of each token's embedding for verification
    for (i, chunk) in result.chunks(hidden_size).enumerate() {
        println!(
            "Token {} embedding[0..8]: {:?}",
            i,
            &chunk[..8.min(chunk.len())]
        );
    }
    let embeddings = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("embeddings_buffer"),
        size: (result.len() * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&embeddings, 0, bytemuck::cast_slice(&result));
    let rms_norm = RmsNormWebgpu::new(&device, hidden_size);
    rms_norm.compute(&device, &queue, &embeddings, encoded.get_ids().len());

}
