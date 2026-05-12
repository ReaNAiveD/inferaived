use safetensors::SafeTensors;

use crate::{
    attention::CausalGqaNaiveAttentionWebgpu,
    binary::ElementwiseAddInplaceWebgpu,
    conv_silu::{ChannelMode, ConvSiluWebgpu},
    delta_rule::DeltaRuleWebgpu,
    gated_rms_norm::GatedRmsNormInplaceWebgpu,
    log_tensor,
    mlp::MultiLayerPerceptron,
    mul_mat::MulMatWebgpu,
    norm::{RmsNormInplaceWebgpu, RmsNormWebgpu},
    rope::RopeInplaceWebgpu,
    sigmoid_mul::SigmoidMulInplaceWebgpu,
    slice_copy::SliceCopyWebgpu,
};

#[derive(Debug, Clone, Copy)]
pub struct LinearAttentionConfig {
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
}

pub struct LinearAttentionLayer {
    input_layernorm: RmsNormWebgpu,
    normed_embedding_buffer: wgpu::Buffer,
    in_proj_qkv_mul_mat: MulMatWebgpu,
    in_proj_qkv_buffer: wgpu::Buffer,
    in_proj_z_mul_mat: MulMatWebgpu,
    in_proj_z_buffer: wgpu::Buffer,
    in_proj_a_mul_mat: MulMatWebgpu,
    in_proj_a_buffer: wgpu::Buffer,
    in_proj_b_mul_mat: MulMatWebgpu,
    in_proj_b_buffer: wgpu::Buffer,
    conv_silu: ConvSiluWebgpu,
    conv_qkv_buffer: wgpu::Buffer,
    delta_rule: DeltaRuleWebgpu,
    recurrent_state_buffer: wgpu::Buffer,
    attn_output_buffer: wgpu::Buffer,
    gated_norm: GatedRmsNormInplaceWebgpu,
    out_proj_mat_mul: MulMatWebgpu,
    out_proj_buffer: wgpu::Buffer,
    attn_residual_add: ElementwiseAddInplaceWebgpu,
    post_attention_layernorm: RmsNormWebgpu,
    mlp: MultiLayerPerceptron,
    mlp_output_buffer: wgpu::Buffer,
    mlp_residual_add: ElementwiseAddInplaceWebgpu,
}

impl LinearAttentionLayer {
    pub fn new<'data>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tensor: &SafeTensors<'data>,
        weight_prefix: &str,
        hidden_size: usize,
        config: &LinearAttentionConfig,
        seq_len: usize,
    ) -> Self {
        let input_layernorm_weight_name = format!("{}.input_layernorm.weight", weight_prefix);
        let input_layernorm_weight = tensor.tensor(&input_layernorm_weight_name).expect(&format!(
            "Failed to get tensor for {}",
            input_layernorm_weight_name
        ));
        log_tensor(&input_layernorm_weight_name, &input_layernorm_weight);
        let input_layernorm =
            RmsNormWebgpu::new(device, queue, input_layernorm_weight, hidden_size);
        let normed_embedding_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/normed_embedding_buffer"),
            size: (seq_len * hidden_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let qkv_weight_name = format!("{}.linear_attn.in_proj_qkv.weight", weight_prefix);
        let qkv_weight = tensor
            .tensor(&qkv_weight_name)
            .expect(&format!("Failed to get tensor for {}", qkv_weight_name));
        log_tensor(&qkv_weight_name, &qkv_weight);
        let qkv_weight_height = qkv_weight.shape()[0] as usize;
        let in_proj_qkv_mul_mat = MulMatWebgpu::new(device, queue, qkv_weight, hidden_size);
        let in_proj_qkv_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/in_proj_qkv_buffer"),
            size: (seq_len * qkv_weight_height * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let in_proj_z_weight_name = format!("{}.linear_attn.in_proj_z.weight", weight_prefix);
        let in_proj_z_weight = tensor.tensor(&in_proj_z_weight_name).expect(&format!(
            "Failed to get tensor for {}",
            in_proj_z_weight_name
        ));
        log_tensor(&in_proj_z_weight_name, &in_proj_z_weight);
        let in_proj_z_weight_height = in_proj_z_weight.shape()[0] as usize;
        let in_proj_z_mul_mat = MulMatWebgpu::new(device, queue, in_proj_z_weight, hidden_size);
        let in_proj_z_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/in_proj_z_buffer"),
            size: (seq_len * in_proj_z_weight_height * std::mem::size_of::<f32>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let in_proj_a_weight_name = format!("{}.linear_attn.in_proj_a.weight", weight_prefix);
        let in_proj_a_weight = tensor.tensor(&in_proj_a_weight_name).expect(&format!(
            "Failed to get tensor for {}",
            in_proj_a_weight_name
        ));
        log_tensor(&in_proj_a_weight_name, &in_proj_a_weight);
        let in_proj_a_weight_height = in_proj_a_weight.shape()[0] as usize;
        let in_proj_a_mul_mat = MulMatWebgpu::new(device, queue, in_proj_a_weight, hidden_size);
        let in_proj_a_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/in_proj_a_buffer"),
            size: (seq_len * in_proj_a_weight_height * std::mem::size_of::<f32>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let in_proj_b_weight_name = format!("{}.linear_attn.in_proj_b.weight", weight_prefix);
        let in_proj_b_weight = tensor.tensor(&in_proj_b_weight_name).expect(&format!(
            "Failed to get tensor for {}",
            in_proj_b_weight_name
        ));
        log_tensor(&in_proj_b_weight_name, &in_proj_b_weight);
        let in_proj_b_weight_height = in_proj_b_weight.shape()[0] as usize;
        let in_proj_b_mul_mat = MulMatWebgpu::new(device, queue, in_proj_b_weight, hidden_size);
        let in_proj_b_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/in_proj_b_buffer"),
            size: (seq_len * in_proj_b_weight_height * std::mem::size_of::<f32>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let q_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let qkv_dim = q_dim + k_dim + v_dim;
        let conv1d_weight_name = format!("{}.linear_attn.conv1d.weight", weight_prefix);
        let conv1d_weight = tensor
            .tensor(&conv1d_weight_name)
            .expect(&format!("Failed to get tensor for {}", conv1d_weight_name));
        log_tensor(&conv1d_weight_name, &conv1d_weight);
        let kernel_size = conv1d_weight.shape()[2] as usize;
        let conv_silu = ConvSiluWebgpu::new(
            &device,
            conv1d_weight,
            q_dim,
            k_dim,
            v_dim,
            kernel_size,
            ChannelMode::ConvSilu,
            ChannelMode::ConvSilu,
            ChannelMode::ConvSilu,
        );
        let conv_qkv_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/conv_qkv_buffer"),
            size: (seq_len * qkv_dim * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let dt_bias_weight_name = format!("{}.linear_attn.dt_bias", weight_prefix);
        let dt_bias_weight = tensor
            .tensor(&dt_bias_weight_name)
            .expect(&format!("Failed to get tensor for {}", dt_bias_weight_name));
        log_tensor(&dt_bias_weight_name, &dt_bias_weight);
        let a_log_weight_name = format!("{}.linear_attn.A_log", weight_prefix);
        let a_log_weight = tensor
            .tensor(&a_log_weight_name)
            .expect(&format!("Failed to get tensor for {}", a_log_weight_name));
        log_tensor(&a_log_weight_name, &a_log_weight);
        let delta_rule = DeltaRuleWebgpu::new(
            &device,
            dt_bias_weight,
            a_log_weight,
            config.linear_num_key_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
        );
        let recurrent_state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/recurrent_state_buffer"),
            size: (config.linear_num_key_heads
                * config.linear_key_head_dim
                * config.linear_value_head_dim
                * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let attn_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/attn_output_buffer"),
            size: (seq_len * v_dim * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let norm_weight_name = format!("{}.linear_attn.norm.weight", weight_prefix);
        let norm_weight = tensor
            .tensor(&norm_weight_name)
            .expect(&format!("Failed to get tensor for {}", norm_weight_name));
        log_tensor(&norm_weight_name, &norm_weight);
        let gated_norm = GatedRmsNormInplaceWebgpu::new(
            &device,
            norm_weight,
            config.linear_num_value_heads,
            config.linear_value_head_dim,
            1e-6f32,
        );
        let out_proj_weight_name = format!("{}.linear_attn.out_proj.weight", weight_prefix);
        let out_proj_weight = tensor.tensor(&out_proj_weight_name).expect(&format!(
            "Failed to get tensor for {}",
            out_proj_weight_name
        ));
        log_tensor(&out_proj_weight_name, &out_proj_weight);
        let out_proj_mat_mul = MulMatWebgpu::new(&device, &queue, out_proj_weight, v_dim);
        let out_proj_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/out_proj_buffer"),
            size: (seq_len * hidden_size * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let attn_residual_add = ElementwiseAddInplaceWebgpu::new(&device, hidden_size);
        let post_attention_layernorm_weight_name =
            format!("{}.post_attention_layernorm.weight", weight_prefix);
        let post_attention_layernorm_weight = tensor
            .tensor(&post_attention_layernorm_weight_name)
            .expect(&format!(
                "Failed to get tensor for {}",
                post_attention_layernorm_weight_name
            ));
        let post_attention_layernorm = RmsNormWebgpu::new(
            &device,
            &queue,
            post_attention_layernorm_weight,
            hidden_size,
        );
        let mlp =
            MultiLayerPerceptron::new(device, queue, tensor, weight_prefix, hidden_size, seq_len);
        let mlp_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/mlp_down_proj_buffer"),
            size: (seq_len * hidden_size * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mlp_residual_add = ElementwiseAddInplaceWebgpu::new(&device, hidden_size);
        Self {
            input_layernorm,
            normed_embedding_buffer,
            in_proj_qkv_mul_mat,
            in_proj_qkv_buffer,
            in_proj_z_mul_mat,
            in_proj_z_buffer,
            in_proj_a_mul_mat,
            in_proj_a_buffer,
            in_proj_b_mul_mat,
            in_proj_b_buffer,
            conv_silu,
            conv_qkv_buffer,
            delta_rule,
            recurrent_state_buffer,
            attn_output_buffer,
            gated_norm,
            out_proj_mat_mul,
            out_proj_buffer,
            attn_residual_add,
            post_attention_layernorm,
            mlp,
            mlp_output_buffer,
            mlp_residual_add,
        }
    }

    pub fn compute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        embedding_buffer: &wgpu::Buffer,
        seq_len: usize,
    ) {
        self.input_layernorm.compute(
            device,
            queue,
            &embedding_buffer,
            &self.normed_embedding_buffer,
            seq_len,
        );
        self.in_proj_qkv_mul_mat.compute(
            device,
            queue,
            &self.normed_embedding_buffer,
            &self.in_proj_qkv_buffer,
            seq_len,
        );
        self.in_proj_a_mul_mat.compute(
            device,
            queue,
            &self.normed_embedding_buffer,
            &self.in_proj_a_buffer,
            seq_len,
        );
        self.in_proj_b_mul_mat.compute(
            device,
            queue,
            &self.normed_embedding_buffer,
            &self.in_proj_b_buffer,
            seq_len,
        );
        self.conv_silu.compute(
            device,
            queue,
            &self.in_proj_qkv_buffer,
            &self.conv_qkv_buffer,
            seq_len,
        );
        self.delta_rule.compute(
            device,
            queue,
            &self.conv_qkv_buffer,
            &self.in_proj_a_buffer,
            &self.in_proj_b_buffer,
            &self.recurrent_state_buffer,
            &self.attn_output_buffer,
            seq_len,
        );
        self.in_proj_z_mul_mat.compute(
            device,
            queue,
            &self.normed_embedding_buffer,
            &self.in_proj_z_buffer,
            seq_len,
        );
        self.gated_norm.compute(
            device,
            queue,
            &self.attn_output_buffer,
            &self.in_proj_z_buffer,
            seq_len,
        );
        self.out_proj_mat_mul.compute(
            device,
            queue,
            &self.attn_output_buffer,
            &self.out_proj_buffer,
            seq_len,
        );
        self.attn_residual_add.compute(
            device,
            queue,
            &embedding_buffer,
            &self.out_proj_buffer,
            seq_len,
        );
        self.post_attention_layernorm.compute(
            device,
            queue,
            &embedding_buffer,
            &self.normed_embedding_buffer,
            seq_len,
        );
        self.mlp.compute(
            device,
            queue,
            &self.normed_embedding_buffer,
            &self.mlp_output_buffer,
            seq_len,
        );
        self.mlp_residual_add.compute(
            device,
            queue,
            &embedding_buffer,
            &self.mlp_output_buffer,
            seq_len,
        );
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SelfAttentionConfig {
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
}

pub struct SelfAttentionLayer {
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,

    input_layernorm: RmsNormWebgpu,
    normed_embedding_buffer: wgpu::Buffer,
    q_proj_mul_mat: MulMatWebgpu,
    q_gate_proj_buffer: wgpu::Buffer,
    q_extract: SliceCopyWebgpu,
    q_proj_buffer: wgpu::Buffer,
    k_proj_mul_mat: MulMatWebgpu,
    k_proj_buffer: wgpu::Buffer,
    v_proj_mul_mat: MulMatWebgpu,
    v_proj_buffer: wgpu::Buffer,
    q_norm: RmsNormInplaceWebgpu,
    k_norm: RmsNormInplaceWebgpu,
    rope: RopeInplaceWebgpu,
    gqa_attention: CausalGqaNaiveAttentionWebgpu,
    attn_output_buffer: wgpu::Buffer,
    sigmoid_mul: SigmoidMulInplaceWebgpu,
    o_proj_mul_mat: MulMatWebgpu,
    o_proj_buffer: wgpu::Buffer,
    attn_residual_add: ElementwiseAddInplaceWebgpu,
    post_attention_layernorm: RmsNormWebgpu,
    mlp: MultiLayerPerceptron,
    mlp_output_buffer: wgpu::Buffer,
    mlp_residual_add: ElementwiseAddInplaceWebgpu,
}

impl SelfAttentionLayer {
    pub fn new<'data>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tensor: &SafeTensors<'data>,
        weight_prefix: &str,
        hidden_size: usize,
        config: &SelfAttentionConfig,
        seq_len: usize,
    ) -> Self {
        let input_layernorm_weight_name = format!("{}.input_layernorm.weight", weight_prefix);
        let input_layernorm_weight = tensor.tensor(&input_layernorm_weight_name).expect(&format!(
            "Failed to get tensor for {}",
            input_layernorm_weight_name
        ));
        log_tensor(&input_layernorm_weight_name, &input_layernorm_weight);
        let input_layernorm =
            RmsNormWebgpu::new(device, queue, input_layernorm_weight, hidden_size);
        let normed_embedding_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("self_attention_layer/normed_embedding_buffer"),
            size: (seq_len * hidden_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let q_proj_weight_name = format!("{}.self_attn.q_proj.weight", weight_prefix);
        let q_proj_weight = tensor
            .tensor(&q_proj_weight_name)
            .expect(&format!("Failed to get tensor for {}", q_proj_weight_name));
        let q_proj_weight_height = q_proj_weight.shape()[0] as usize;
        log_tensor(&q_proj_weight_name, &q_proj_weight);
        let q_proj_mul_mat = MulMatWebgpu::new(device, queue, q_proj_weight, hidden_size);
        let q_gate_proj_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("self_attention_layer/q_proj_buffer"),
            size: (seq_len * q_proj_weight_height * std::mem::size_of::<f32>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Tight buffer that holds just the Q half of `q_proj_buffer`, used as
        // input by q_norm / RoPE / attention. The gate half stays in
        // `q_proj_buffer` for the later sigmoid * gate step.
        let q_proj_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("self_attention_layer/q_buffer"),
            size: (seq_len
                * config.num_attention_heads
                * config.head_dim
                * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let q_extract = SliceCopyWebgpu::new(device);
        let k_proj_weight_name = format!("{}.self_attn.k_proj.weight", weight_prefix);
        let k_proj_weight = tensor
            .tensor(&k_proj_weight_name)
            .expect(&format!("Failed to get tensor for {}", k_proj_weight_name));
        let k_proj_weight_height = k_proj_weight.shape()[0] as usize;
        log_tensor(&k_proj_weight_name, &k_proj_weight);
        let k_proj_mul_mat = MulMatWebgpu::new(device, queue, k_proj_weight, hidden_size);
        let k_proj_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("self_attention_layer/k_proj_buffer"),
            size: (seq_len * k_proj_weight_height * std::mem::size_of::<f32>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let v_proj_weight_name = format!("{}.self_attn.v_proj.weight", weight_prefix);
        let v_proj_weight = tensor
            .tensor(&v_proj_weight_name)
            .expect(&format!("Failed to get tensor for {}", v_proj_weight_name));
        let v_proj_weight_height = v_proj_weight.shape()[0] as usize;
        log_tensor(&v_proj_weight_name, &v_proj_weight);
        let v_proj_mul_mat = MulMatWebgpu::new(device, queue, v_proj_weight, hidden_size);
        let v_proj_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("self_attention_layer/v_proj_buffer"),
            size: (seq_len * v_proj_weight_height * std::mem::size_of::<f32>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let q_norm_weight_name = format!("{}.self_attn.q_norm.weight", weight_prefix);
        let q_norm_weight = tensor
            .tensor(&q_norm_weight_name)
            .expect(&format!("Failed to get tensor for {}", q_norm_weight_name));
        log_tensor(&q_norm_weight_name, &q_norm_weight);
        let q_norm = RmsNormInplaceWebgpu::new(device, queue, q_norm_weight, config.head_dim);
        let k_norm_weight_name = format!("{}.self_attn.k_norm.weight", weight_prefix);
        let k_norm_weight = tensor
            .tensor(&k_norm_weight_name)
            .expect(&format!("Failed to get tensor for {}", k_norm_weight_name));
        log_tensor(&k_norm_weight_name, &k_norm_weight);
        let k_norm = RmsNormInplaceWebgpu::new(device, queue, k_norm_weight, config.head_dim);
        let rope = RopeInplaceWebgpu::new(
            device,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.rope_theta,
            config.partial_rotary_factor,
        );
        let gqa_attention = CausalGqaNaiveAttentionWebgpu::new(
            device,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.head_dim, // value head dim is typically the same as key head dim in GQA
        );
        let attn_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("self_attention_layer/attn_output_buffer"),
            size: (seq_len
                * config.head_dim
                * config.num_attention_heads
                * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let sigmoid_mul =
            SigmoidMulInplaceWebgpu::new(device, config.head_dim * config.num_attention_heads);
        let o_proj_weight_name = format!("{}.self_attn.o_proj.weight", weight_prefix);
        let o_proj_weight = tensor
            .tensor(&o_proj_weight_name)
            .expect(&format!("Failed to get tensor for {}", o_proj_weight_name));
        log_tensor(&o_proj_weight_name, &o_proj_weight);
        let o_proj_mul_mat = MulMatWebgpu::new(device, queue, o_proj_weight, hidden_size);
        let o_proj_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("self_attention_layer/o_proj_buffer"),
            size: (seq_len * hidden_size * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let attn_residual_add = ElementwiseAddInplaceWebgpu::new(&device, hidden_size);
        let post_attention_layernorm_weight_name =
            format!("{}.post_attention_layernorm.weight", weight_prefix);
        let post_attention_layernorm_weight = tensor
            .tensor(&post_attention_layernorm_weight_name)
            .expect(&format!(
                "Failed to get tensor for {}",
                post_attention_layernorm_weight_name
            ));
        let post_attention_layernorm = RmsNormWebgpu::new(
            &device,
            &queue,
            post_attention_layernorm_weight,
            hidden_size,
        );
        let mlp =
            MultiLayerPerceptron::new(device, queue, tensor, weight_prefix, hidden_size, seq_len);
        let mlp_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/mlp_down_proj_buffer"),
            size: (seq_len * hidden_size * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mlp_residual_add = ElementwiseAddInplaceWebgpu::new(&device, hidden_size);
        Self {
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            input_layernorm,
            normed_embedding_buffer,
            q_proj_mul_mat,
            q_gate_proj_buffer,
            q_extract,
            q_proj_buffer,
            k_proj_mul_mat,
            k_proj_buffer,
            v_proj_mul_mat,
            v_proj_buffer,
            q_norm,
            k_norm,
            rope,
            gqa_attention,
            attn_output_buffer,
            sigmoid_mul,
            o_proj_mul_mat,
            o_proj_buffer,
            attn_residual_add,
            post_attention_layernorm,
            mlp,
            mlp_output_buffer,
            mlp_residual_add,
        }
    }

    pub fn compute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        embedding_buffer: &wgpu::Buffer,
        seq_len: usize,
    ) {
        self.input_layernorm.compute(
            device,
            queue,
            &embedding_buffer,
            &self.normed_embedding_buffer,
            seq_len,
        );
        self.q_proj_mul_mat.compute(
            device,
            queue,
            &self.normed_embedding_buffer,
            &self.q_gate_proj_buffer,
            seq_len,
        );
        self.k_proj_mul_mat.compute(
            device,
            queue,
            &self.normed_embedding_buffer,
            &self.k_proj_buffer,
            seq_len,
        );
        self.v_proj_mul_mat.compute(
            device,
            queue,
            &self.normed_embedding_buffer,
            &self.v_proj_buffer,
            seq_len,
        );
        self.q_extract.compute(
            device,
            queue,
            &self.q_gate_proj_buffer,
            &self.q_proj_buffer,
            /* src_offset       */ 0,
            /* src_token_stride */ self.num_attention_heads * self.head_dim * 2,
            /* src_head_stride  */ self.head_dim * 2,
            /* dst_offset       */ 0,
            /* dst_token_stride */ self.num_attention_heads * self.head_dim,
            /* dst_head_stride  */ self.head_dim,
            self.num_attention_heads,
            self.head_dim,
            seq_len,
        );
        self.q_norm.compute(
            device,
            queue,
            &self.q_proj_buffer,
            seq_len * self.num_attention_heads,
        );
        self.k_norm.compute(
            device,
            queue,
            &self.k_proj_buffer,
            seq_len * self.num_key_value_heads,
        );
        self.rope.compute(
            device,
            queue,
            &self.q_proj_buffer,
            &self.k_proj_buffer,
            seq_len,
            0,
        );
        self.gqa_attention.compute(
            device,
            queue,
            &self.q_proj_buffer,
            &self.k_proj_buffer,
            &self.v_proj_buffer,
            &self.attn_output_buffer,
            seq_len,
        );
        self.sigmoid_mul.compute_strided(
            device,
            queue,
            &self.attn_output_buffer,
            &self.q_gate_proj_buffer,
            0,
            self.num_attention_heads * self.head_dim,
            self.head_dim,
            self.head_dim,
            self.num_attention_heads * self.head_dim * 2,
            self.head_dim * 2,
            self.num_attention_heads,
            self.head_dim,
            seq_len,
        );
        self.o_proj_mul_mat.compute(
            device,
            queue,
            &self.attn_output_buffer,
            &self.o_proj_buffer,
            seq_len,
        );
        self.attn_residual_add.compute(
            device,
            queue,
            &embedding_buffer,
            &self.o_proj_buffer,
            seq_len,
        );
        self.post_attention_layernorm.compute(
            device,
            queue,
            &embedding_buffer,
            &self.normed_embedding_buffer,
            seq_len,
        );
        self.mlp.compute(
            device,
            queue,
            &self.normed_embedding_buffer,
            &self.mlp_output_buffer,
            seq_len,
        );
        self.mlp_residual_add.compute(
            device,
            queue,
            &embedding_buffer,
            &self.mlp_output_buffer,
            seq_len,
        );
    }
}
