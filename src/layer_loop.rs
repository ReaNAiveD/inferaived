use safetensors::SafeTensors;

use crate::{
    binary::ElementwiseAddWebgpu,
    conv_silu::{ChannelMode, ConvSiluWebgpu},
    delta_rule::DeltaRuleWebgpu,
    gated_rms_norm::GatedRmsNormWebgpu,
    mul_mat::MulMatWebgpu,
    norm::RmsNormWebgpu,
    log_tensor,
    silu_mul::SiluMulWebgpu,
};

#[derive(Debug, Clone, Copy)]
pub struct LinearLayerConfig {
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
}

pub struct LinearLayer {
    hidden_size: usize,

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
    gated_norm: GatedRmsNormWebgpu,
    out_proj_mat_mul: MulMatWebgpu,
    out_proj_buffer: wgpu::Buffer,
    attn_residual_add: ElementwiseAddWebgpu,
    post_attention_layernorm: RmsNormWebgpu,
    mlp_gate_proj_mul_mat: MulMatWebgpu,
    mlp_gate_proj_buffer: wgpu::Buffer,
    mlp_up_proj_mul_mat: MulMatWebgpu,
    mlp_up_proj_buffer: wgpu::Buffer,
    mlp_silu_mul: SiluMulWebgpu,
    mlp_down_proj_mul_mat: MulMatWebgpu,
    mlp_down_proj_buffer: wgpu::Buffer,
    mlp_residual_add: ElementwiseAddWebgpu,
}

impl LinearLayer {
    pub fn new<'data>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tensor: &SafeTensors<'data>,
        weight_prefix: String,
        hidden_size: usize,
        config: &LinearLayerConfig,
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
            label: Some("linear_layer/normed_embedding_buffer"),
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
            label: Some("linear_layer/in_proj_qkv_buffer"),
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
            label: Some("linear_layer/in_proj_z_buffer"),
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
            label: Some("linear_layer/in_proj_a_buffer"),
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
            label: Some("linear_layer/in_proj_b_buffer"),
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
            label: Some("linear_layer/conv_qkv_buffer"),
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
            label: Some("linear_layer/recurrent_state_buffer"),
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
            label: Some("linear_layer/attn_output_buffer"),
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
        let gated_norm = GatedRmsNormWebgpu::new(
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
            label: Some("linear_layer/out_proj_buffer"),
            size: (seq_len * hidden_size * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let attn_residual_add = ElementwiseAddWebgpu::new(&device, hidden_size);
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
        let mlp_gate_proj_weight_name = format!("{}.mlp.gate_proj.weight", weight_prefix);
        let mlp_gate_proj_weight = tensor.tensor(&mlp_gate_proj_weight_name).expect(&format!(
            "Failed to get tensor for {}",
            mlp_gate_proj_weight_name
        ));
        log_tensor(&mlp_gate_proj_weight_name, &mlp_gate_proj_weight);
        let mlp_gate_dim = mlp_gate_proj_weight.shape()[0] as usize;
        let mlp_gate_proj_mul_mat =
            MulMatWebgpu::new(&device, &queue, mlp_gate_proj_weight, hidden_size);
        let mlp_gate_proj_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_layer/mlp_gate_proj_buffer"),
            size: (seq_len * mlp_gate_dim * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mlp_up_proj_weight_name = format!("{}.mlp.up_proj.weight", weight_prefix);
        let mlp_up_proj_weight = tensor.tensor(&mlp_up_proj_weight_name).expect(&format!(
            "Failed to get tensor for {}",
            mlp_up_proj_weight_name
        ));
        log_tensor(&mlp_up_proj_weight_name, &mlp_up_proj_weight);
        let mlp_up_proj_mul_mat =
            MulMatWebgpu::new(&device, &queue, mlp_up_proj_weight, hidden_size);
        let mlp_up_proj_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_layer/mlp_up_proj_buffer"),
            size: (seq_len * mlp_gate_dim * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mlp_silu_mul = SiluMulWebgpu::new(&device, mlp_gate_dim);
        let mlp_down_proj_weight_name = format!("{}.mlp.down_proj.weight", weight_prefix);
        let mlp_down_proj_weight = tensor.tensor(&mlp_down_proj_weight_name).expect(&format!(
            "Failed to get tensor for {}",
            mlp_down_proj_weight_name
        ));
        log_tensor(&mlp_down_proj_weight_name, &mlp_down_proj_weight);
        let mlp_down_proj_mul_mat =
            MulMatWebgpu::new(&device, &queue, mlp_down_proj_weight, mlp_gate_dim);
        let mlp_down_proj_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_layer/mlp_down_proj_buffer"),
            size: (seq_len * hidden_size * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mlp_residual_add = ElementwiseAddWebgpu::new(&device, hidden_size);
        Self {
            hidden_size,
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
            mlp_gate_proj_mul_mat,
            mlp_gate_proj_buffer,
            mlp_up_proj_mul_mat,
            mlp_up_proj_buffer,
            mlp_silu_mul,
            mlp_down_proj_mul_mat,
            mlp_down_proj_buffer,
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
        self.mlp_gate_proj_mul_mat.compute(
            device,
            queue,
            &self.normed_embedding_buffer,
            &self.mlp_gate_proj_buffer,
            seq_len,
        );
        self.mlp_up_proj_mul_mat.compute(
            device,
            queue,
            &self.normed_embedding_buffer,
            &self.mlp_up_proj_buffer,
            seq_len,
        );
        self.mlp_silu_mul.compute(
            device,
            queue,
            &self.mlp_up_proj_buffer,
            &self.mlp_gate_proj_buffer,
            seq_len,
        );
        self.mlp_down_proj_mul_mat.compute(
            device,
            queue,
            &self.mlp_up_proj_buffer,
            &self.mlp_down_proj_buffer,
            seq_len,
        );
        self.mlp_residual_add.compute(
            device,
            queue,
            &embedding_buffer,
            &self.mlp_down_proj_buffer,
            seq_len,
        );
    }
}
