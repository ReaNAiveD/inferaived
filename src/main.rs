use inferaived::{
    conv_silu::{ChannelMode, ConvSiluWebgpu},
    delta_rule::DeltaRuleWebgpu,
    embedding_lookup::EmbeddingLookupCpu,
    mul_mat::MulMatWebgpu,
    norm::{NormScaleWebgpu, RmsNormWebgpu},
};
use safetensors::SafeTensors;
use tokenizers::Tokenizer;
use tokio;
use wgpu::{
    BackendOptions, Backends, DeviceDescriptor, ExperimentalFeatures, Features, Instance,
    InstanceDescriptor, InstanceFlags, MemoryBudgetThresholds, MemoryHints, PowerPreference,
    RequestAdapterOptions, Trace,
};

fn print_tensor(name: &str, tensor: &safetensors::tensor::TensorView) {
    let num_elements: usize = tensor.shape().iter().product();
    if num_elements <= 64 {
        println!("Tensor {}: {:?}", name, tensor);
    } else {
        println!(
            "Tensor {}: dtype={:?}, shape={:?}",
            name,
            tensor.dtype(),
            tensor.shape()
        );
    }
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

#[tokio::main]
async fn main() {
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
    print_tensor("model.language_model.embed_tokens.weight", &embeddings);
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
    let embeddings = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("embeddings_buffer"),
        size: (result.len() * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&embeddings, 0, bytemuck::cast_slice(&result));
    let rms_norm = RmsNormWebgpu::new(&device, hidden_size);
    rms_norm.compute(&device, &queue, &embeddings, encoded.get_ids().len());
    let norm_weight0 = tensors
        .tensor("model.language_model.layers.0.input_layernorm.weight")
        .expect("Failed to get tensor: model.language_model.layers.0.input_layernorm.weight");
    print_tensor(
        "model.language_model.layers.0.input_layernorm.weight",
        &norm_weight0,
    );
    let norm_scale0 = NormScaleWebgpu::new(&device, &queue, norm_weight0, hidden_size);
    norm_scale0.compute(&device, &queue, &embeddings, encoded.get_ids().len());
    let qkv_weight0 = tensors
        .tensor("model.language_model.layers.0.linear_attn.in_proj_qkv.weight")
        .expect(
            "Failed to get tensor: model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        );
    print_tensor(
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        &qkv_weight0,
    );
    let qkv_weight0_height = qkv_weight0.shape()[0] as usize;
    let qkv_mul_mat = MulMatWebgpu::new(&device, &queue, qkv_weight0, hidden_size);
    let qkv_dst_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("qkv_dst_buffer"),
        size: (encoded.get_ids().len() * qkv_weight0_height * std::mem::size_of::<f32>())
            as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    qkv_mul_mat.execute(
        &device,
        &queue,
        &embeddings,
        &qkv_dst_buffer,
        encoded.get_ids().len(),
    );
    let z_weight0 = tensors
        .tensor("model.language_model.layers.0.linear_attn.in_proj_z.weight")
        .expect("Failed to get tensor: model.language_model.layers.0.linear_attn.in_proj_z.weight");
    print_tensor(
        "model.language_model.layers.0.linear_attn.in_proj_z.weight",
        &z_weight0,
    );
    let z_weight0_height = z_weight0.shape()[0] as usize;
    let z_mul_mat = MulMatWebgpu::new(&device, &queue, z_weight0, hidden_size);
    let z_dst_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("z_dst_buffer"),
        size: (encoded.get_ids().len() * z_weight0_height * std::mem::size_of::<f32>())
            as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    z_mul_mat.execute(
        &device,
        &queue,
        &embeddings,
        &z_dst_buffer,
        encoded.get_ids().len(),
    );
    let proj_a0 = tensors
        .tensor("model.language_model.layers.0.linear_attn.in_proj_a.weight")
        .expect("Failed to get tensor: model.language_model.layers.0.linear_attn.in_proj_a.weight");
    print_tensor(
        "model.language_model.layers.0.linear_attn.in_proj_a.weight",
        &proj_a0,
    );
    let proj_a0_height = proj_a0.shape()[0] as usize;
    let proj_a_mul_mat = MulMatWebgpu::new(&device, &queue, proj_a0, hidden_size);
    let proj_a_dst_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("proj_a_dst_buffer"),
        size: (encoded.get_ids().len() * proj_a0_height * std::mem::size_of::<f32>())
            as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    proj_a_mul_mat.execute(
        &device,
        &queue,
        &embeddings,
        &proj_a_dst_buffer,
        encoded.get_ids().len(),
    );
    let proj_b0 = tensors
        .tensor("model.language_model.layers.0.linear_attn.in_proj_b.weight")
        .expect("Failed to get tensor: model.language_model.layers.0.linear_attn.in_proj_b.weight");
    print_tensor(
        "model.language_model.layers.0.linear_attn.in_proj_b.weight",
        &proj_b0,
    );
    let proj_b0_height = proj_b0.shape()[0] as usize;
    let proj_b_mul_mat = MulMatWebgpu::new(&device, &queue, proj_b0, hidden_size);
    let proj_b_dst_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("proj_b_dst_buffer"),
        size: (encoded.get_ids().len() * proj_b0_height * std::mem::size_of::<f32>())
            as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    proj_b_mul_mat.execute(
        &device,
        &queue,
        &embeddings,
        &proj_b_dst_buffer,
        encoded.get_ids().len(),
    );
    let q_dim = linear_num_key_heads * linear_key_head_dim;
    let k_dim = linear_num_key_heads * linear_key_head_dim;
    let v_dim = linear_num_value_heads * linear_value_head_dim;
    let qkv_dim = q_dim + k_dim + v_dim;
    let conv_weights0 = tensors
        .tensor("model.language_model.layers.0.linear_attn.conv1d.weight")
        .expect("Failed to get tensor: model.language_model.layers.0.linear_attn.conv1d.weight");
    print_tensor(
        "model.language_model.layers.0.linear_attn.conv1d.weight",
        &conv_weights0,
    );
    let kernel_size = conv_weights0.shape()[2] as usize;
    let conv_silu = ConvSiluWebgpu::new(
        &device,
        conv_weights0,
        q_dim,
        k_dim,
        v_dim,
        kernel_size,
        ChannelMode::ConvSilu,
        ChannelMode::ConvSilu,
        ChannelMode::ConvSilu,
    );
    let mixed_qkv_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mixed_qkv_buffer"),
        size: (encoded.get_ids().len() * qkv_dim * std::mem::size_of::<f32>())
            as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    conv_silu.compute(
        &device,
        &queue,
        &qkv_dst_buffer,
        &mixed_qkv_buffer,
        encoded.get_ids().len(),
    );
    // let rope = RopeWebgpu::new(&device, linear_key_head_dim, linear_num_key_heads, qkv_dim, rope_theta, partial_rotary_factor);
    // rope.compute(&device, &queue, &mixed_qkv_buffer, encoded.get_ids().iter().len(), 0);
    let dt_bias0 = tensors
        .tensor("model.language_model.layers.0.linear_attn.dt_bias")
        .expect("Failed to get tensor: model.language_model.layers.0.linear_attn.dt_bias");
    print_tensor(
        "model.language_model.layers.0.linear_attn.dt_bias",
        &dt_bias0,
    );
    let a_log0 = tensors
        .tensor("model.language_model.layers.0.linear_attn.A_log")
        .expect("Failed to get tensor: model.language_model.layers.0.linear_attn.A_log");
    print_tensor("model.language_model.layers.0.linear_attn.A_log", &a_log0);
    let delta_rule = DeltaRuleWebgpu::new(
        &device,
        dt_bias0,
        a_log0,
        linear_num_key_heads as u32,
        linear_key_head_dim as u32,
        linear_value_head_dim as u32,
    );
    let state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("state_buffer"),
        size: (linear_num_key_heads
            * linear_key_head_dim
            * linear_value_head_dim
            * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mamba_dst_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mamba_dst_buffer"),
        size: (encoded.get_ids().len() * v_dim * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    delta_rule.compute(
        &device,
        &queue,
        &mixed_qkv_buffer,
        &proj_a_dst_buffer,
        &proj_b_dst_buffer,
        &state_buffer,
        &mamba_dst_buffer,
        encoded.get_ids().iter().len(),
    );
}
