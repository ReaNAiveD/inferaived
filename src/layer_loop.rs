use safetensors::SafeTensors;

use crate::{
    attention::CausalGqaNaiveAttentionWebgpu,
    binary::ElementwiseAddInplaceWebgpu,
    buffer_view::BufferView,
    conv_silu::{ChannelMode, ConvSiluWebgpu},
    delta_rule::DeltaRuleWebgpu,
    gated_rms_norm::GatedRmsNormInplaceWebgpu,
    log_tensor,
    mlp::MultiLayerPerceptron,
    mul_mat::MulMatWebgpu,
    norm::{RmsNormInplaceWebgpu, RmsNormWebgpu},
    rope::RopeInplaceWebgpu,
    sigmoid_mul::SigmoidMulInplaceWebgpu,
};

/// Per-sequence forward interface for a transformer layer.
pub trait LayerSession {
    /// Run this layer over `residual_slot.shape[0]` new tokens starting
    /// at absolute position `prev_position`. `self`'s state at entry
    /// must reflect everything before `prev_position`; at exit it
    /// reflects everything before
    /// `prev_position + residual_slot.shape[0]`.
    ///
    /// * Cold prefill of an `N`-token prompt: `forward(slot, 0)` with
    ///   `slot.shape[0] == N`.
    /// * Single-token decode at position `P`: `forward(slot, P)` with
    ///   `slot.shape[0] == 1`.
    /// * Continued prefill (appending `M` tokens to a session of
    ///   length `K`): `forward(slot, K)` with `slot.shape[0] == M`.
    fn forward(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        residual_slot: BufferView<'_>,
        prev_position: usize,
    );
}

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
            config.head_dim,
        );
        let sigmoid_mul =
            SigmoidMulInplaceWebgpu::new(device, config.num_attention_heads, config.head_dim);
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

    /// Run the full self-attention block (in-norm → q/k/v_proj →
    /// q/k_norm → RoPE → causal GQA → output gate → o_proj → residual
    /// + post-norm → MLP → residual) over the `[num_new, hidden_size]`
    /// `residual_slot`, updating it in place.
    pub fn forward(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        residual_slot: BufferView<'_>,
        prev_position: usize,
        k_cache_buffer: &wgpu::Buffer,
        v_cache_buffer: &wgpu::Buffer,
    ) {
        let num_new_tokens = residual_slot.shape[0] as usize;
        debug_assert!(
            num_new_tokens >= 1,
            "forward requires residual_slot.shape[0] >= 1, got {}",
            num_new_tokens,
        );
        let q_dim = self.num_attention_heads * self.head_dim;
        let q_gate_dim = q_dim * 2;
        let sz = std::mem::size_of::<f32>() as u32;
        let num_new_u32 = num_new_tokens as u32;
        let hidden_size = self.hidden_size as u32;
        let kv_dim = self.num_key_value_heads * self.head_dim;
        let kv_prefix_rows = (prev_position + num_new_tokens) as u32;

        let normed_embedding_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("self_attention_layer/normed_embedding_buffer"),
            size: (num_new_tokens * self.hidden_size * std::mem::size_of::<f32>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let normed =
            BufferView::new_2d_tight(&normed_embedding_buffer, num_new_u32, hidden_size, sz);

        let q_gate_proj_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("self_attention_layer/q_gate_proj_buffer"),
            size: (num_new_tokens * q_gate_dim * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let q_gate_view = BufferView::new_4d_tight(
            &q_gate_proj_buffer,
            num_new_u32,
            self.num_attention_heads as u32,
            2,
            self.head_dim as u32,
            sz,
        );

        let attn_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("self_attention_layer/attn_output_buffer"),
            size: (num_new_tokens * q_dim * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let attn_out_view = BufferView::new_3d_tight(
            &attn_output_buffer,
            num_new_u32,
            self.num_attention_heads as u32,
            self.head_dim as u32,
            sz,
        );

        let o_proj_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("self_attention_layer/o_proj_buffer"),
            size: (num_new_tokens * self.hidden_size * std::mem::size_of::<f32>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let o_proj_view = BufferView::new_2d_tight(&o_proj_buffer, num_new_u32, hidden_size, sz);

        let mlp_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("self_attention_layer/mlp_output_buffer"),
            size: (num_new_tokens * self.hidden_size * std::mem::size_of::<f32>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mlp_out_view =
            BufferView::new_2d_tight(&mlp_output_buffer, num_new_u32, hidden_size, sz);

        let max_seq_in_cache = (k_cache_buffer.size()
            / (kv_dim as wgpu::BufferAddress * sz as wgpu::BufferAddress))
            as u32;
        let k_cache_view = BufferView::new_3d_tight(
            k_cache_buffer,
            max_seq_in_cache,
            self.num_key_value_heads as u32,
            self.head_dim as u32,
            sz,
        );
        let v_cache_view = BufferView::new_3d_tight(
            v_cache_buffer,
            max_seq_in_cache,
            self.num_key_value_heads as u32,
            self.head_dim as u32,
            sz,
        );
        let k_new = k_cache_view.narrow(0, prev_position as u32, num_new_u32);
        let v_new = v_cache_view.narrow(0, prev_position as u32, num_new_u32);
        let k_full_prefix = k_cache_view.narrow(0, 0, kv_prefix_rows);
        let v_full_prefix = v_cache_view.narrow(0, 0, kv_prefix_rows);
        let k_new_heads = k_new.flatten_outer(2);

        self.input_layernorm
            .forward(device, queue, residual_slot, normed);
        self.q_proj_mul_mat
            .forward(device, queue, normed, q_gate_view);
        let q_view = q_gate_view.select(2, 0);
        let gate_view = q_gate_view.select(2, 1);
        let q_heads_flat_view = q_view.flatten_outer(2);
        self.k_proj_mul_mat.forward(device, queue, normed, k_new);
        self.v_proj_mul_mat.forward(device, queue, normed, v_new);
        self.q_norm.forward(device, queue, q_heads_flat_view);
        self.k_norm.forward(device, queue, k_new_heads);
        self.rope
            .forward(device, queue, q_view, k_new, prev_position);
        self.gqa_attention.forward(
            device,
            queue,
            q_view,
            k_full_prefix,
            v_full_prefix,
            attn_out_view,
            prev_position,
        );
        self.sigmoid_mul
            .forward(device, queue, attn_out_view, gate_view);
        self.o_proj_mul_mat
            .forward(device, queue, attn_out_view, o_proj_view);
        self.attn_residual_add
            .forward(device, queue, residual_slot, o_proj_view);
        self.post_attention_layernorm
            .forward(device, queue, residual_slot, normed);
        self.mlp.forward(device, queue, normed, mlp_out_view);
        self.mlp_residual_add
            .forward(device, queue, residual_slot, mlp_out_view);
    }
}

pub struct SelfAttentionLayerSession<'m> {
    layer: &'m SelfAttentionLayer,
    k_cache_buffer: wgpu::Buffer,
    v_cache_buffer: wgpu::Buffer,
}

impl<'m> SelfAttentionLayerSession<'m> {
    pub fn new(layer: &'m SelfAttentionLayer, device: &wgpu::Device, max_seq_len: usize) -> Self {
        debug_assert!(max_seq_len >= 1, "max_seq_len must be >= 1");
        let kv_dim = layer.num_key_value_heads * layer.head_dim;
        let cache_bytes =
            (max_seq_len * kv_dim * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
        let k_cache_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("self_attention_layer/session/k_cache_buffer"),
            size: cache_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let v_cache_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("self_attention_layer/session/v_cache_buffer"),
            size: cache_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self {
            layer,
            k_cache_buffer,
            v_cache_buffer,
        }
    }
}

impl<'m> LayerSession for SelfAttentionLayerSession<'m> {
    fn forward(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        residual_slot: BufferView<'_>,
        prev_position: usize,
    ) {
        self.layer.forward(
            device,
            queue,
            residual_slot,
            prev_position,
            &self.k_cache_buffer,
            &self.v_cache_buffer,
        );
    }
}

// TODO: consider unifying LinearAttentionLayer and SelfAttentionLayer into a single generic AttentionLayer with generic parameters for the various components
pub enum AttentionLayer {
    Linear(LinearAttentionLayer),
    Full(SelfAttentionLayer),
}

impl AttentionLayer {
    pub fn new_session(
        &self,
        device: &wgpu::Device,
        max_seq_len: usize,
    ) -> AttentionLayerSession<'_> {
        match self {
            AttentionLayer::Linear(layer) => AttentionLayerSession::Linear(
                LinearAttentionLayerSession::new(layer, device, max_seq_len),
            ),
            AttentionLayer::Full(layer) => AttentionLayerSession::Full(
                SelfAttentionLayerSession::new(layer, device, max_seq_len),
            ),
        }
    }
}

/// Per-layer, per-sequence state for one full transformer block. The enum
/// carries the model↔state pairing in the type system: a `Linear` layer
/// can only be paired with linear-attention state, and a `Full` layer can
/// only be paired with KV-cache buffers. This makes type-mismatched cache
/// access unrepresentable.
pub enum AttentionLayerSession<'m> {
    Linear(LinearAttentionLayerSession<'m>),
    Full(SelfAttentionLayerSession<'m>),
}

impl<'m> LayerSession for AttentionLayerSession<'m> {
    fn forward(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        residual_slot: BufferView<'_>,
        prev_position: usize,
    ) {
        match self {
            Self::Linear(session) => session.forward(device, queue, residual_slot, prev_position),
            Self::Full(session) => session.forward(device, queue, residual_slot, prev_position),
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

    pub fn layers(&self) -> &[AttentionLayer] {
        &self.layers
    }
}

/// Stack-wide per-sequence state: one `AttentionLayerSession` per layer in
/// the underlying `LayerStack`, in the same order. Borrows the stack
/// immutably for the lifetime of the session.
pub struct LayerStackSession<'m> {
    sessions: Vec<AttentionLayerSession<'m>>,
}

impl<'m> LayerStackSession<'m> {
    pub fn new(stack: &'m LayerStack, device: &wgpu::Device, max_seq_len: usize) -> Self {
        let sessions = stack
            .layers()
            .iter()
            .map(|layer| layer.new_session(device, max_seq_len))
            .collect();
        Self { sessions }
    }
}

impl<'m> LayerSession for LayerStackSession<'m> {
    fn forward(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        residual_slot: BufferView<'_>,
        prev_position: usize,
    ) {
        for session in &mut self.sessions {
            session.forward(device, queue, residual_slot, prev_position);
        }
    }
}
