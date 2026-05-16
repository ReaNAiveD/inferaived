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
    pub intermediate_size: usize,
}

pub struct LinearAttentionLayer {
    hidden_size: usize,
    linear_num_value_heads: usize,
    qkv_dim: usize,
    v_dim: usize,
    recurrent_state_size: usize,
    conv_state_size: usize,
    input_layernorm: RmsNormWebgpu,
    in_proj_qkv_mul_mat: MulMatWebgpu,
    in_proj_z_mul_mat: MulMatWebgpu,
    in_proj_a_mul_mat: MulMatWebgpu,
    in_proj_b_mul_mat: MulMatWebgpu,
    conv_silu: ConvSiluWebgpu,
    delta_rule: DeltaRuleWebgpu,
    gated_norm: GatedRmsNormInplaceWebgpu,
    out_proj_mat_mul: MulMatWebgpu,
    attn_residual_add: ElementwiseAddInplaceWebgpu,
    post_attention_layernorm: RmsNormWebgpu,
    mlp: MultiLayerPerceptron,
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
    ) -> Self {
        let q_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let k_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let v_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let qkv_dim = q_dim + k_dim + v_dim;
        let input_layernorm_weight_name = format!("{}.input_layernorm.weight", weight_prefix);
        let input_layernorm_weight = tensor.tensor(&input_layernorm_weight_name).expect(&format!(
            "Failed to get tensor for {}",
            input_layernorm_weight_name
        ));
        log_tensor(&input_layernorm_weight_name, &input_layernorm_weight);
        let input_layernorm = RmsNormWebgpu::new(device, queue, input_layernorm_weight);
        let qkv_weight_name = format!("{}.linear_attn.in_proj_qkv.weight", weight_prefix);
        let qkv_weight = tensor
            .tensor(&qkv_weight_name)
            .expect(&format!("Failed to get tensor for {}", qkv_weight_name));
        log_tensor(&qkv_weight_name, &qkv_weight);
        debug_assert_eq!(
            qkv_weight.shape()[0] as usize,
            qkv_dim,
            "{} height does not match q_dim+k_dim+v_dim",
            qkv_weight_name
        );
        debug_assert_eq!(
            qkv_weight.shape()[1] as usize,
            hidden_size,
            "{} width does not match hidden_size",
            qkv_weight_name
        );
        let in_proj_qkv_mul_mat = MulMatWebgpu::new(device, queue, qkv_weight);
        let in_proj_z_weight_name = format!("{}.linear_attn.in_proj_z.weight", weight_prefix);
        let in_proj_z_weight = tensor.tensor(&in_proj_z_weight_name).expect(&format!(
            "Failed to get tensor for {}",
            in_proj_z_weight_name
        ));
        log_tensor(&in_proj_z_weight_name, &in_proj_z_weight);
        debug_assert_eq!(
            in_proj_z_weight.shape()[0] as usize,
            v_dim,
            "{} height does not match v_dim",
            in_proj_z_weight_name
        );
        debug_assert_eq!(
            in_proj_z_weight.shape()[1] as usize,
            hidden_size,
            "{} width does not match hidden_size",
            in_proj_z_weight_name
        );
        let in_proj_z_mul_mat = MulMatWebgpu::new(device, queue, in_proj_z_weight);
        let in_proj_a_weight_name = format!("{}.linear_attn.in_proj_a.weight", weight_prefix);
        let in_proj_a_weight = tensor.tensor(&in_proj_a_weight_name).expect(&format!(
            "Failed to get tensor for {}",
            in_proj_a_weight_name
        ));
        log_tensor(&in_proj_a_weight_name, &in_proj_a_weight);
        debug_assert_eq!(
            in_proj_a_weight.shape()[0] as usize,
            config.linear_num_value_heads,
            "{} height does not match linear_num_value_heads",
            in_proj_a_weight_name
        );
        debug_assert_eq!(
            in_proj_a_weight.shape()[1] as usize,
            hidden_size,
            "{} width does not match hidden_size",
            in_proj_a_weight_name
        );
        let in_proj_a_mul_mat = MulMatWebgpu::new(device, queue, in_proj_a_weight);
        let in_proj_b_weight_name = format!("{}.linear_attn.in_proj_b.weight", weight_prefix);
        let in_proj_b_weight = tensor.tensor(&in_proj_b_weight_name).expect(&format!(
            "Failed to get tensor for {}",
            in_proj_b_weight_name
        ));
        log_tensor(&in_proj_b_weight_name, &in_proj_b_weight);
        debug_assert_eq!(
            in_proj_b_weight.shape()[0] as usize,
            config.linear_num_value_heads,
            "{} height does not match linear_num_value_heads",
            in_proj_b_weight_name
        );
        debug_assert_eq!(
            in_proj_b_weight.shape()[1] as usize,
            hidden_size,
            "{} width does not match hidden_size",
            in_proj_b_weight_name
        );
        let in_proj_b_mul_mat = MulMatWebgpu::new(device, queue, in_proj_b_weight);
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
        let recurrent_state_size = config.linear_num_key_heads
            * config.linear_key_head_dim
            * config.linear_value_head_dim;
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
        debug_assert_eq!(
            out_proj_weight.shape()[0] as usize,
            hidden_size,
            "{} height does not match hidden_size",
            out_proj_weight_name
        );
        debug_assert_eq!(
            out_proj_weight.shape()[1] as usize,
            v_dim,
            "{} width does not match v_dim",
            out_proj_weight_name
        );
        let out_proj_mat_mul = MulMatWebgpu::new(&device, &queue, out_proj_weight);
        let attn_residual_add = ElementwiseAddInplaceWebgpu::new(&device, hidden_size);
        let post_attention_layernorm_weight_name =
            format!("{}.post_attention_layernorm.weight", weight_prefix);
        let post_attention_layernorm_weight = tensor
            .tensor(&post_attention_layernorm_weight_name)
            .expect(&format!(
                "Failed to get tensor for {}",
                post_attention_layernorm_weight_name
            ));
        let post_attention_layernorm =
            RmsNormWebgpu::new(&device, &queue, post_attention_layernorm_weight);
        let mlp = MultiLayerPerceptron::new(
            device,
            queue,
            tensor,
            weight_prefix,
            hidden_size,
            config.intermediate_size,
        );
        let mlp_residual_add = ElementwiseAddInplaceWebgpu::new(&device, hidden_size);
        let conv_state_size = conv_silu.conv_state_size();
        Self {
            hidden_size,
            linear_num_value_heads: config.linear_num_value_heads,
            qkv_dim,
            v_dim,
            recurrent_state_size,
            conv_state_size,
            input_layernorm,
            in_proj_qkv_mul_mat,
            in_proj_z_mul_mat,
            in_proj_a_mul_mat,
            in_proj_b_mul_mat,
            conv_silu,
            delta_rule,
            gated_norm,
            out_proj_mat_mul,
            attn_residual_add,
            post_attention_layernorm,
            mlp,
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
        let normed_embedding_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/normed_embedding_buffer"),
            size: (seq_len * self.hidden_size * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let in_proj_qkv_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/in_proj_qkv_buffer"),
            size: (seq_len * self.qkv_dim * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let in_proj_z_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/in_proj_z_buffer"),
            size: (seq_len * self.v_dim * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let in_proj_a_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/in_proj_a_buffer"),
            size: (seq_len * self.linear_num_value_heads * std::mem::size_of::<f32>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let in_proj_b_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/in_proj_b_buffer"),
            size: (seq_len * self.linear_num_value_heads * std::mem::size_of::<f32>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let conv_qkv_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/conv_qkv_buffer"),
            size: (seq_len * self.qkv_dim * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let attn_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/attn_output_buffer"),
            size: (seq_len * self.v_dim * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_proj_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/out_proj_buffer"),
            size: (seq_len * self.hidden_size * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mlp_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/mlp_output_buffer"),
            size: (seq_len * self.hidden_size * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let recurrent_state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/recurrent_state_buffer"),
            size: (self.recurrent_state_size * std::mem::size_of::<f32>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let conv_state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/conv_state_buffer"),
            size: (self.conv_state_size * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.input_layernorm.compute(
            device,
            queue,
            &embedding_buffer,
            &normed_embedding_buffer,
            seq_len,
        );
        self.in_proj_qkv_mul_mat.compute(
            device,
            queue,
            &normed_embedding_buffer,
            &in_proj_qkv_buffer,
            seq_len,
        );
        self.in_proj_a_mul_mat.compute(
            device,
            queue,
            &normed_embedding_buffer,
            &in_proj_a_buffer,
            seq_len,
        );
        self.in_proj_b_mul_mat.compute(
            device,
            queue,
            &normed_embedding_buffer,
            &in_proj_b_buffer,
            seq_len,
        );
        self.conv_silu.compute(
            device,
            queue,
            &in_proj_qkv_buffer,
            &conv_qkv_buffer,
            &conv_state_buffer,
            seq_len,
        );
        self.delta_rule.compute(
            device,
            queue,
            &conv_qkv_buffer,
            &in_proj_a_buffer,
            &in_proj_b_buffer,
            &recurrent_state_buffer,
            &attn_output_buffer,
            seq_len,
        );
        self.in_proj_z_mul_mat.compute(
            device,
            queue,
            &normed_embedding_buffer,
            &in_proj_z_buffer,
            seq_len,
        );
        self.gated_norm.compute(
            device,
            queue,
            &attn_output_buffer,
            &in_proj_z_buffer,
            seq_len,
        );
        self.out_proj_mat_mul.compute(
            device,
            queue,
            &attn_output_buffer,
            &out_proj_buffer,
            seq_len,
        );
        self.attn_residual_add
            .compute(device, queue, &embedding_buffer, &out_proj_buffer, seq_len);
        self.post_attention_layernorm.compute(
            device,
            queue,
            &embedding_buffer,
            &normed_embedding_buffer,
            seq_len,
        );
        self.mlp.compute(
            device,
            queue,
            &normed_embedding_buffer,
            &mlp_output_buffer,
            seq_len,
        );
        self.mlp_residual_add.compute(
            device,
            queue,
            &embedding_buffer,
            &mlp_output_buffer,
            seq_len,
        );
    }

    pub fn compute_decode(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        embedding_buffer: &wgpu::Buffer,
        conv_state_buffer: &wgpu::Buffer,
        recurrent_state_buffer: &wgpu::Buffer,
        seq_len: usize,
    ) {
        debug_assert!(
            seq_len >= 1,
            "compute_decode requires seq_len >= 1, got {}",
            seq_len,
        );
        let _ = self.conv_state_size;
        let _ = self.recurrent_state_size;
        let position = seq_len - 1;
        let residual_offset = position * self.hidden_size;
        let normed_embedding_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/normed_embedding_buffer"),
            size: (self.hidden_size * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let in_proj_qkv_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/in_proj_qkv_buffer"),
            size: (self.qkv_dim * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let in_proj_z_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/in_proj_z_buffer"),
            size: (self.v_dim * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let in_proj_a_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/in_proj_a_buffer"),
            size: (self.linear_num_value_heads * std::mem::size_of::<f32>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let in_proj_b_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/in_proj_b_buffer"),
            size: (self.linear_num_value_heads * std::mem::size_of::<f32>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let conv_qkv_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/conv_qkv_buffer"),
            size: (self.qkv_dim * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let attn_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/attn_output_buffer"),
            size: (self.v_dim * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_proj_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/out_proj_buffer"),
            size: (self.hidden_size * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mlp_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/mlp_output_buffer"),
            size: (self.hidden_size * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.input_layernorm.compute_last_row(
            device,
            queue,
            embedding_buffer,
            &normed_embedding_buffer,
            seq_len,
        );
        self.in_proj_qkv_mul_mat.compute(
            device,
            queue,
            &normed_embedding_buffer,
            &in_proj_qkv_buffer,
            1,
        );
        self.in_proj_a_mul_mat.compute(
            device,
            queue,
            &normed_embedding_buffer,
            &in_proj_a_buffer,
            1,
        );
        self.in_proj_b_mul_mat.compute(
            device,
            queue,
            &normed_embedding_buffer,
            &in_proj_b_buffer,
            1,
        );
        self.conv_silu.compute(
            device,
            queue,
            &in_proj_qkv_buffer,
            &conv_qkv_buffer,
            conv_state_buffer,
            1,
        );
        self.delta_rule.compute(
            device,
            queue,
            &conv_qkv_buffer,
            &in_proj_a_buffer,
            &in_proj_b_buffer,
            recurrent_state_buffer,
            &attn_output_buffer,
            1,
        );
        self.in_proj_z_mul_mat.compute(
            device,
            queue,
            &normed_embedding_buffer,
            &in_proj_z_buffer,
            1,
        );
        self.gated_norm
            .compute(device, queue, &attn_output_buffer, &in_proj_z_buffer, 1);
        self.out_proj_mat_mul.compute(
            device,
            queue,
            &attn_output_buffer,
            &out_proj_buffer,
            1,
        );
        self.attn_residual_add.compute_strided(
            device,
            queue,
            embedding_buffer,
            &out_proj_buffer,
            residual_offset,
            0,
            1,
        );
        self.post_attention_layernorm.compute_last_row(
            device,
            queue,
            embedding_buffer,
            &normed_embedding_buffer,
            seq_len,
        );
        self.mlp.compute_decode(
            device,
            queue,
            &normed_embedding_buffer,
            &mlp_output_buffer,
        );
        self.mlp_residual_add.compute_strided(
            device,
            queue,
            embedding_buffer,
            &mlp_output_buffer,
            residual_offset,
            0,
            1,
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
    pub intermediate_size: usize,
}

pub struct SelfAttentionLayer {
    hidden_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,

    input_layernorm: RmsNormWebgpu,
    q_proj_mul_mat: MulMatWebgpu,
    q_extract: SliceCopyWebgpu,
    k_proj_mul_mat: MulMatWebgpu,
    v_proj_mul_mat: MulMatWebgpu,
    q_norm: RmsNormInplaceWebgpu,
    k_norm: RmsNormInplaceWebgpu,
    rope: RopeInplaceWebgpu,
    gqa_attention: CausalGqaNaiveAttentionWebgpu,
    sigmoid_mul: SigmoidMulInplaceWebgpu,
    o_proj_mul_mat: MulMatWebgpu,
    attn_residual_add: ElementwiseAddInplaceWebgpu,
    post_attention_layernorm: RmsNormWebgpu,
    mlp: MultiLayerPerceptron,
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
    ) -> Self {
        let q_dim = config.num_attention_heads * config.head_dim;
        let q_gate_dim = q_dim * 2;
        let kv_dim = config.num_key_value_heads * config.head_dim;
        let input_layernorm_weight_name = format!("{}.input_layernorm.weight", weight_prefix);
        let input_layernorm_weight = tensor.tensor(&input_layernorm_weight_name).expect(&format!(
            "Failed to get tensor for {}",
            input_layernorm_weight_name
        ));
        log_tensor(&input_layernorm_weight_name, &input_layernorm_weight);
        let input_layernorm = RmsNormWebgpu::new(device, queue, input_layernorm_weight);
        let q_proj_weight_name = format!("{}.self_attn.q_proj.weight", weight_prefix);
        let q_proj_weight = tensor
            .tensor(&q_proj_weight_name)
            .expect(&format!("Failed to get tensor for {}", q_proj_weight_name));
        log_tensor(&q_proj_weight_name, &q_proj_weight);
        debug_assert_eq!(
            q_proj_weight.shape()[0] as usize,
            q_gate_dim,
            "{} height does not match num_attention_heads * head_dim * 2 (output gate)",
            q_proj_weight_name
        );
        debug_assert_eq!(
            q_proj_weight.shape()[1] as usize,
            hidden_size,
            "{} width does not match hidden_size",
            q_proj_weight_name
        );
        let q_proj_mul_mat = MulMatWebgpu::new(device, queue, q_proj_weight);
        let q_extract = SliceCopyWebgpu::new(device);
        let k_proj_weight_name = format!("{}.self_attn.k_proj.weight", weight_prefix);
        let k_proj_weight = tensor
            .tensor(&k_proj_weight_name)
            .expect(&format!("Failed to get tensor for {}", k_proj_weight_name));
        log_tensor(&k_proj_weight_name, &k_proj_weight);
        debug_assert_eq!(
            k_proj_weight.shape()[0] as usize,
            kv_dim,
            "{} height does not match num_key_value_heads * head_dim",
            k_proj_weight_name
        );
        debug_assert_eq!(
            k_proj_weight.shape()[1] as usize,
            hidden_size,
            "{} width does not match hidden_size",
            k_proj_weight_name
        );
        let k_proj_mul_mat = MulMatWebgpu::new(device, queue, k_proj_weight);
        let v_proj_weight_name = format!("{}.self_attn.v_proj.weight", weight_prefix);
        let v_proj_weight = tensor
            .tensor(&v_proj_weight_name)
            .expect(&format!("Failed to get tensor for {}", v_proj_weight_name));
        log_tensor(&v_proj_weight_name, &v_proj_weight);
        debug_assert_eq!(
            v_proj_weight.shape()[0] as usize,
            kv_dim,
            "{} height does not match num_key_value_heads * head_dim",
            v_proj_weight_name
        );
        debug_assert_eq!(
            v_proj_weight.shape()[1] as usize,
            hidden_size,
            "{} width does not match hidden_size",
            v_proj_weight_name
        );
        let v_proj_mul_mat = MulMatWebgpu::new(device, queue, v_proj_weight);
        let q_norm_weight_name = format!("{}.self_attn.q_norm.weight", weight_prefix);
        let q_norm_weight = tensor
            .tensor(&q_norm_weight_name)
            .expect(&format!("Failed to get tensor for {}", q_norm_weight_name));
        log_tensor(&q_norm_weight_name, &q_norm_weight);
        let q_norm = RmsNormInplaceWebgpu::new(device, queue, q_norm_weight);
        let k_norm_weight_name = format!("{}.self_attn.k_norm.weight", weight_prefix);
        let k_norm_weight = tensor
            .tensor(&k_norm_weight_name)
            .expect(&format!("Failed to get tensor for {}", k_norm_weight_name));
        log_tensor(&k_norm_weight_name, &k_norm_weight);
        let k_norm = RmsNormInplaceWebgpu::new(device, queue, k_norm_weight);
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
        let sigmoid_mul =
            SigmoidMulInplaceWebgpu::new(device, config.head_dim * config.num_attention_heads);
        let o_proj_weight_name = format!("{}.self_attn.o_proj.weight", weight_prefix);
        let o_proj_weight = tensor
            .tensor(&o_proj_weight_name)
            .expect(&format!("Failed to get tensor for {}", o_proj_weight_name));
        log_tensor(&o_proj_weight_name, &o_proj_weight);
        debug_assert_eq!(
            o_proj_weight.shape()[0] as usize,
            hidden_size,
            "{} height does not match hidden_size",
            o_proj_weight_name
        );
        debug_assert_eq!(
            o_proj_weight.shape()[1] as usize,
            q_dim,
            "{} width does not match num_attention_heads * head_dim",
            o_proj_weight_name
        );
        let o_proj_mul_mat = MulMatWebgpu::new(device, queue, o_proj_weight);
        let attn_residual_add = ElementwiseAddInplaceWebgpu::new(&device, hidden_size);
        let post_attention_layernorm_weight_name =
            format!("{}.post_attention_layernorm.weight", weight_prefix);
        let post_attention_layernorm_weight = tensor
            .tensor(&post_attention_layernorm_weight_name)
            .expect(&format!(
                "Failed to get tensor for {}",
                post_attention_layernorm_weight_name
            ));
        let post_attention_layernorm =
            RmsNormWebgpu::new(&device, &queue, post_attention_layernorm_weight);
        let mlp = MultiLayerPerceptron::new(
            device,
            queue,
            tensor,
            weight_prefix,
            hidden_size,
            config.intermediate_size,
        );
        let mlp_residual_add = ElementwiseAddInplaceWebgpu::new(&device, hidden_size);
        Self {
            hidden_size,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            input_layernorm,
            q_proj_mul_mat,
            q_extract,
            k_proj_mul_mat,
            v_proj_mul_mat,
            q_norm,
            k_norm,
            rope,
            gqa_attention,
            sigmoid_mul,
            o_proj_mul_mat,
            attn_residual_add,
            post_attention_layernorm,
            mlp,
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
        let q_dim = self.num_attention_heads * self.head_dim;
        let q_gate_dim = q_dim * 2;
        let kv_dim = self.num_key_value_heads * self.head_dim;
        let normed_embedding_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("self_attention_layer/normed_embedding_buffer"),
            size: (seq_len * self.hidden_size * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let q_gate_proj_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("self_attention_layer/q_gate_proj_buffer"),
            size: (seq_len * q_gate_dim * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Tight buffer that holds just the Q half of `q_gate_proj_buffer`, used as
        // input by q_norm / RoPE / attention. The gate half stays in
        // `q_gate_proj_buffer` for the later sigmoid * gate step.
        let q_proj_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("self_attention_layer/q_proj_buffer"),
            size: (seq_len * q_dim * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let k_proj_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("self_attention_layer/k_proj_buffer"),
            size: (seq_len * kv_dim * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let v_proj_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("self_attention_layer/v_proj_buffer"),
            size: (seq_len * kv_dim * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let attn_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("self_attention_layer/attn_output_buffer"),
            size: (seq_len * q_dim * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let o_proj_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("self_attention_layer/o_proj_buffer"),
            size: (seq_len * self.hidden_size * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mlp_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("self_attention_layer/mlp_output_buffer"),
            size: (seq_len * self.hidden_size * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.input_layernorm.compute(
            device,
            queue,
            &embedding_buffer,
            &normed_embedding_buffer,
            seq_len,
        );
        self.q_proj_mul_mat.compute(
            device,
            queue,
            &normed_embedding_buffer,
            &q_gate_proj_buffer,
            seq_len,
        );
        self.k_proj_mul_mat.compute(
            device,
            queue,
            &normed_embedding_buffer,
            &k_proj_buffer,
            seq_len,
        );
        self.v_proj_mul_mat.compute(
            device,
            queue,
            &normed_embedding_buffer,
            &v_proj_buffer,
            seq_len,
        );
        self.q_extract.compute(
            device,
            queue,
            &q_gate_proj_buffer,
            &q_proj_buffer,
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
            &q_proj_buffer,
            seq_len * self.num_attention_heads,
        );
        self.k_norm.compute(
            device,
            queue,
            &k_proj_buffer,
            seq_len * self.num_key_value_heads,
        );
        self.rope
            .compute(device, queue, &q_proj_buffer, &k_proj_buffer, seq_len, 0);
        self.gqa_attention.compute(
            device,
            queue,
            &q_proj_buffer,
            &k_proj_buffer,
            &v_proj_buffer,
            &attn_output_buffer,
            seq_len,
        );
        self.sigmoid_mul.compute_strided(
            device,
            queue,
            &attn_output_buffer,
            &q_gate_proj_buffer,
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
        self.o_proj_mul_mat
            .compute(device, queue, &attn_output_buffer, &o_proj_buffer, seq_len);
        self.attn_residual_add
            .compute(device, queue, &embedding_buffer, &o_proj_buffer, seq_len);
        self.post_attention_layernorm.compute(
            device,
            queue,
            &embedding_buffer,
            &normed_embedding_buffer,
            seq_len,
        );
        self.mlp.compute(
            device,
            queue,
            &normed_embedding_buffer,
            &mlp_output_buffer,
            seq_len,
        );
        self.mlp_residual_add.compute(
            device,
            queue,
            &embedding_buffer,
            &mlp_output_buffer,
            seq_len,
        );
    }

    pub fn compute_decode(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        embedding_buffer: &wgpu::Buffer,
        k_buffer: &wgpu::Buffer,
        v_buffer: &wgpu::Buffer,
        seq_len: usize,
    ) {
        debug_assert!(
            seq_len >= 1,
            "compute_decoder requires seq_len >= 1, got {}",
            seq_len,
        );
        let q_dim = self.num_attention_heads * self.head_dim;
        let q_gate_dim = q_dim * 2;
        let kv_dim = self.num_key_value_heads * self.head_dim;
        let position = seq_len - 1;
        let kv_cache_slot_offset = position * kv_dim;
        let residual_offset = position * self.hidden_size;
        let normed_embedding_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("self_attention_layer/normed_embedding_buffer"),
            size: (self.hidden_size * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let q_gate_proj_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("self_attention_layer/q_gate_proj_buffer"),
            size: (q_gate_dim * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let q_proj_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("self_attention_layer/q_proj_buffer"),
            size: (q_dim * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let attn_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("self_attention_layer/attn_output_buffer"),
            size: (q_dim * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let o_proj_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("self_attention_layer/o_proj_buffer"),
            size: (self.hidden_size * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mlp_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("self_attention_layer/mlp_output_buffer"),
            size: (self.hidden_size * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.input_layernorm.compute_last_row(
            device,
            queue,
            embedding_buffer,
            &normed_embedding_buffer,
            seq_len,
        );
        self.q_proj_mul_mat.compute(
            device,
            queue,
            &normed_embedding_buffer,
            &q_gate_proj_buffer,
            1,
        );
        self.k_proj_mul_mat.compute_strided(
            device,
            queue,
            &normed_embedding_buffer,
            k_buffer,
            0,
            kv_cache_slot_offset,
            1,
        );
        self.v_proj_mul_mat.compute_strided(
            device,
            queue,
            &normed_embedding_buffer,
            v_buffer,
            0,
            kv_cache_slot_offset,
            1,
        );
        self.q_extract.compute(
            device,
            queue,
            &q_gate_proj_buffer,
            &q_proj_buffer,
            0,
            self.num_attention_heads * self.head_dim * 2,
            self.head_dim * 2,
            0,
            self.num_attention_heads * self.head_dim,
            self.head_dim,
            self.num_attention_heads,
            self.head_dim,
            1,
        );
        self.q_norm
            .compute(device, queue, &q_proj_buffer, self.num_attention_heads);
        self.k_norm.compute_strided(
            device,
            queue,
            k_buffer,
            kv_cache_slot_offset,
            self.num_key_value_heads,
            self.head_dim,
        );
        self.rope.compute_strided(
            device,
            queue,
            &q_proj_buffer,
            k_buffer,
            0,
            kv_cache_slot_offset,
            1,
            position,
        );
        self.gqa_attention.compute_strided(
            device,
            queue,
            &q_proj_buffer,
            k_buffer,
            v_buffer,
            &attn_output_buffer,
            0,
            0,
            0,
            0,
            1,
            position,
        );
        self.sigmoid_mul.compute_strided(
            device,
            queue,
            &attn_output_buffer,
            &q_gate_proj_buffer,
            0,
            self.num_attention_heads * self.head_dim,
            self.head_dim,
            self.head_dim,
            self.num_attention_heads * self.head_dim * 2,
            self.head_dim * 2,
            self.num_attention_heads,
            self.head_dim,
            1,
        );
        self.o_proj_mul_mat
            .compute(device, queue, &attn_output_buffer, &o_proj_buffer, 1);
        self.attn_residual_add.compute_strided(
            device,
            queue,
            embedding_buffer,
            &o_proj_buffer,
            residual_offset,
            0,
            1,
        );
        self.post_attention_layernorm.compute_last_row(
            device,
            queue,
            embedding_buffer,
            &normed_embedding_buffer,
            seq_len,
        );
        self.mlp.compute_decode(
            device,
            queue,
            &normed_embedding_buffer,
            &mlp_output_buffer,
        );
        self.mlp_residual_add.compute_strided(
            device,
            queue,
            embedding_buffer,
            &mlp_output_buffer,
            residual_offset,
            0,
            1,
        );
    }
}

// TODO: consider unifying LinearAttentionLayer and SelfAttentionLayer into a single generic AttentionLayer with generic parameters for the various components
pub enum AttentionLayer {
    Linear(LinearAttentionLayer),
    Full(SelfAttentionLayer),
}

impl AttentionLayer {
    pub fn compute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        embedding_buffer: &wgpu::Buffer,
        seq_len: usize,
    ) {
        match self {
            AttentionLayer::Linear(layer) => {
                layer.compute(device, queue, embedding_buffer, seq_len)
            }
            AttentionLayer::Full(layer) => layer.compute(device, queue, embedding_buffer, seq_len),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum LayerConfig {
    Linear(LinearAttentionConfig),
    Full(SelfAttentionConfig),
}

#[derive(Debug, Clone)]
pub struct LayerStackConfig {
    pub layers: Vec<LayerConfig>,
}

pub struct LayerStack {
    layers: Vec<AttentionLayer>,
}

impl LayerStack {
    pub fn new<'data>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tensor: &SafeTensors<'data>,
        weight_prefix: &str,
        config: &LayerStackConfig,
        hidden_size: usize,
    ) -> Self {
        let mut layers = Vec::with_capacity(config.layers.len());
        for (i, layer_config) in config.layers.iter().enumerate() {
            let layer_weight_prefix = format!("{}.layers.{}", weight_prefix, i);
            let layer = match layer_config {
                LayerConfig::Linear(linear_config) => {
                    AttentionLayer::Linear(LinearAttentionLayer::new(
                        device,
                        queue,
                        tensor,
                        &layer_weight_prefix,
                        hidden_size,
                        linear_config,
                    ))
                }
                LayerConfig::Full(full_config) => AttentionLayer::Full(SelfAttentionLayer::new(
                    device,
                    queue,
                    tensor,
                    &layer_weight_prefix,
                    hidden_size,
                    full_config,
                )),
            };
            layers.push(layer);
        }
        Self { layers }
    }

    pub fn compute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        embedding_buffer: &wgpu::Buffer,
        seq_len: usize,
    ) {
        for layer in &self.layers {
            layer.compute(device, queue, embedding_buffer, seq_len);
        }
    }
}
