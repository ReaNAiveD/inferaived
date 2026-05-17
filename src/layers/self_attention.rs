use safetensors::SafeTensors;

use crate::{
    buffer_view::BufferView,
    kernels::{
        attention::CausalGqaNaiveAttentionWebgpu,
        binary::ElementwiseAddInplaceWebgpu,
        mul_mat::MulMatWebgpu,
        norm::{RmsNormInplaceWebgpu, RmsNormWebgpu},
        rope::RopeInplaceWebgpu,
        sigmoid_mul::SigmoidMulInplaceWebgpu,
    },
    layers::{layer_session::LayerSession, mlp::MultiLayerPerceptron},
    log_tensor,
};

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
