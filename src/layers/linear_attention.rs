use safetensors::SafeTensors;

use crate::{
    buffer_view::BufferView,
    kernels::{
        binary::ElementwiseAddInplaceWebgpu,
        conv_silu::{ChannelMode, ConvSiluWebgpu},
        delta_rule::DeltaRuleWebgpu,
        gated_rms_norm::GatedRmsNormInplaceWebgpu,
        mul_mat::MulMatWebgpu,
        norm::RmsNormWebgpu,
    },
    layers::{layer_session::LayerSession, mlp::MultiLayerPerceptron},
    log_tensor,
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
        let recurrent_state_size =
            config.linear_num_key_heads * config.linear_key_head_dim * config.linear_value_head_dim;
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

    /// Run the linear-attention block (in-norm → in_proj_qkv/a/b/z →
    /// conv-silu → delta-rule → gated-norm → out_proj → residual +
    /// post-norm → MLP → residual) over the `[num_new, hidden_size]`
    /// `residual_slot`, updating it in place.
    pub fn forward(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        residual_slot: BufferView<'_>,
        conv_state_buffer: &wgpu::Buffer,
        recurrent_state_buffer: &wgpu::Buffer,
    ) {
        let num_new_tokens = residual_slot.shape[0] as usize;
        debug_assert!(
            num_new_tokens >= 1,
            "forward requires residual_slot.shape[0] >= 1, got {}",
            num_new_tokens,
        );
        let sz = std::mem::size_of::<f32>() as u32;
        let num_new_u32 = num_new_tokens as u32;
        let hidden_size = self.hidden_size as u32;
        let qkv_dim = self.qkv_dim as u32;
        let v_dim = self.v_dim as u32;
        let num_v_heads = self.linear_num_value_heads as u32;

        // Each per-forward scratch is paired with its canonical view at
        // the point of creation. conv_silu / delta_rule / gated_norm
        // still consume raw `&wgpu::Buffer`s, so `conv_qkv_buffer`
        // intentionally has no paired view.
        let normed_embedding_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/normed_embedding_buffer"),
            size: (num_new_tokens * self.hidden_size * std::mem::size_of::<f32>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let normed_view =
            BufferView::new_2d_tight(&normed_embedding_buffer, num_new_u32, hidden_size, sz);

        let in_proj_qkv_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/in_proj_qkv_buffer"),
            size: (num_new_tokens * self.qkv_dim * std::mem::size_of::<f32>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let qkv_view = BufferView::new_2d_tight(&in_proj_qkv_buffer, num_new_u32, qkv_dim, sz);

        let in_proj_a_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/in_proj_a_buffer"),
            size: (num_new_tokens * self.linear_num_value_heads * std::mem::size_of::<f32>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let proj_a_view = BufferView::new_2d_tight(&in_proj_a_buffer, num_new_u32, num_v_heads, sz);

        let in_proj_b_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/in_proj_b_buffer"),
            size: (num_new_tokens * self.linear_num_value_heads * std::mem::size_of::<f32>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let proj_b_view = BufferView::new_2d_tight(&in_proj_b_buffer, num_new_u32, num_v_heads, sz);

        let in_proj_z_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/in_proj_z_buffer"),
            size: (num_new_tokens * self.v_dim * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let proj_z_view = BufferView::new_2d_tight(&in_proj_z_buffer, num_new_u32, v_dim, sz);

        // conv1d output: consumed by delta_rule via raw `&wgpu::Buffer`,
        // so no view is built for this scratch.
        let conv_qkv_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/conv_qkv_buffer"),
            size: (num_new_tokens * self.qkv_dim * std::mem::size_of::<f32>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let attn_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/attn_output_buffer"),
            size: (num_new_tokens * self.v_dim * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let attn_out_v_view = BufferView::new_2d_tight(&attn_output_buffer, num_new_u32, v_dim, sz);

        let out_proj_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/out_proj_buffer"),
            size: (num_new_tokens * self.hidden_size * std::mem::size_of::<f32>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let out_proj_view =
            BufferView::new_2d_tight(&out_proj_buffer, num_new_u32, hidden_size, sz);

        let mlp_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/mlp_output_buffer"),
            size: (num_new_tokens * self.hidden_size * std::mem::size_of::<f32>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mlp_out_view =
            BufferView::new_2d_tight(&mlp_output_buffer, num_new_u32, hidden_size, sz);

        self.input_layernorm
            .forward(device, queue, residual_slot, normed_view);
        self.in_proj_qkv_mul_mat
            .forward(device, queue, normed_view, qkv_view);
        self.in_proj_a_mul_mat
            .forward(device, queue, normed_view, proj_a_view);
        self.in_proj_b_mul_mat
            .forward(device, queue, normed_view, proj_b_view);
        self.conv_silu.forward(
            device,
            queue,
            &in_proj_qkv_buffer,
            &conv_qkv_buffer,
            conv_state_buffer,
            num_new_tokens,
        );
        self.delta_rule.forward(
            device,
            queue,
            &conv_qkv_buffer,
            &in_proj_a_buffer,
            &in_proj_b_buffer,
            recurrent_state_buffer,
            &attn_output_buffer,
            num_new_tokens,
        );
        self.in_proj_z_mul_mat
            .forward(device, queue, normed_view, proj_z_view);
        self.gated_norm.forward(
            device,
            queue,
            &attn_output_buffer,
            &in_proj_z_buffer,
            num_new_tokens,
        );
        self.out_proj_mat_mul
            .forward(device, queue, attn_out_v_view, out_proj_view);
        self.attn_residual_add
            .forward(device, queue, residual_slot, out_proj_view);
        self.post_attention_layernorm
            .forward(device, queue, residual_slot, normed_view);
        self.mlp.forward(device, queue, normed_view, mlp_out_view);
        self.mlp_residual_add
            .forward(device, queue, residual_slot, mlp_out_view);
    }
}

/// Per-sequence state for one linear-attention layer. Pairs a borrow of
/// the immutable model layer with its conv and recurrent state buffers.
pub struct LinearAttentionLayerSession<'m> {
    layer: &'m LinearAttentionLayer,
    conv_state_buffer: wgpu::Buffer,
    recurrent_state_buffer: wgpu::Buffer,
}

impl<'m> LinearAttentionLayerSession<'m> {
    pub fn new(
        layer: &'m LinearAttentionLayer,
        device: &wgpu::Device,
        _max_seq_len: usize,
    ) -> Self {
        let conv_state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/session/conv_state_buffer"),
            size: (layer.conv_state_size * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let recurrent_state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/session/recurrent_state_buffer"),
            size: (layer.recurrent_state_size * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self {
            layer,
            conv_state_buffer,
            recurrent_state_buffer,
        }
    }
}

impl<'m> LayerSession for LinearAttentionLayerSession<'m> {
    fn forward(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        residual_slot: BufferView<'_>,
        _prev_position: usize,
    ) {
        self.layer.forward(
            device,
            queue,
            residual_slot,
            &self.conv_state_buffer,
            &self.recurrent_state_buffer,
        );
    }
}
