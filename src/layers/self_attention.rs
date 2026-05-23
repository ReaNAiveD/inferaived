use safetensors::SafeTensors;

use crate::{
    buffer_view::BufferView,
    kernels::{
        attention::{CausalGqaNaiveAttentionWebgpu, CausalGqaNaiveAttentionWebgpuRunner},
        binary::{ElementwiseAddInplaceWebgpu, ElementwiseAddInplaceWebgpuRunner},
        mul_mat::{MulMatWebgpu, MulMatWebgpuRunner},
        norm::{
            RmsNormInplaceWebgpu, RmsNormInplaceWebgpuRunner, RmsNormWebgpu, RmsNormWebgpuRunner,
        },
        rope::{RopeInplaceWebgpu, RopeInplaceWebgpuRunner},
        scatter_row::{ScatterRowWebgpu, ScatterRowWebgpuRunner},
        sigmoid_mul::{SigmoidMulInplaceWebgpu, SigmoidMulInplaceWebgpuRunner},
    },
    layers::mlp::{MlpRunners, MultiLayerPerceptron},
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
    kv_scatter: ScatterRowWebgpu,
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
        let kv_scatter = ScatterRowWebgpu::new(device, kv_dim);
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
            kv_scatter,
            gqa_attention,
            sigmoid_mul,
            o_proj_mul_mat,
            attn_residual_add,
            post_attention_layernorm,
            mlp,
            mlp_residual_add,
        }
    }

    /// Build a [`SelfAttentionLayerRunner`] that records this block's
    /// dispatches into a caller-owned compute pass.
    pub fn plan(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        residual_slot: BufferView<'_>,
        k_cache_view: BufferView<'_>,
        v_cache_view: BufferView<'_>,
        position_buffer: &wgpu::Buffer,
    ) -> SelfAttentionLayerRunner {
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
        debug_assert_eq!(
            k_cache_view.rank, 3,
            "self_attention: k_cache must be rank-3"
        );
        debug_assert_eq!(
            v_cache_view.rank, 3,
            "self_attention: v_cache must be rank-3"
        );
        debug_assert_eq!(
            k_cache_view.shape[0], v_cache_view.shape[0],
            "self_attention: k/v cache max_seq mismatch (k={}, v={})",
            k_cache_view.shape[0], v_cache_view.shape[0],
        );
        debug_assert_eq!(
            k_cache_view.shape[1] as usize, self.num_key_value_heads,
            "self_attention: k_cache shape[1] ({}) must equal num_key_value_heads ({})",
            k_cache_view.shape[1], self.num_key_value_heads,
        );
        debug_assert_eq!(
            v_cache_view.shape[1] as usize, self.num_key_value_heads,
            "self_attention: v_cache shape[1] ({}) must equal num_key_value_heads ({})",
            v_cache_view.shape[1], self.num_key_value_heads,
        );
        debug_assert_eq!(
            k_cache_view.shape[2] as usize, self.head_dim,
            "self_attention: k_cache shape[2] ({}) must equal head_dim ({})",
            k_cache_view.shape[2], self.head_dim,
        );
        debug_assert_eq!(
            v_cache_view.shape[2] as usize, self.head_dim,
            "self_attention: v_cache shape[2] ({}) must equal head_dim ({})",
            v_cache_view.shape[2], self.head_dim,
        );

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

        let decode_kv_bytes =
            (num_new_tokens * kv_dim * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
        let decode_k_new_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("self_attention_layer/decode_k_new_buffer"),
            size: decode_kv_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let decode_v_new_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("self_attention_layer/decode_v_new_buffer"),
            size: decode_kv_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let decode_k_new_view = BufferView::new_3d_tight(
            &decode_k_new_buffer,
            num_new_u32,
            self.num_key_value_heads as u32,
            self.head_dim as u32,
            sz,
        );
        let decode_v_new_view = BufferView::new_3d_tight(
            &decode_v_new_buffer,
            num_new_u32,
            self.num_key_value_heads as u32,
            self.head_dim as u32,
            sz,
        );

        let k_new_heads = decode_k_new_view.flatten_outer(2);

        let q_view = q_gate_view.select(2, 0);
        let gate_view = q_gate_view.select(2, 1);
        let q_heads_flat_view = q_view.flatten_outer(2);

        let input_layernorm_runner =
            self.input_layernorm
                .plan(device, queue, residual_slot, normed);
        let q_proj_runner = self.q_proj_mul_mat.plan(device, queue, normed, q_gate_view);
        let k_proj_runner = self
            .k_proj_mul_mat
            .plan(device, queue, normed, decode_k_new_view);
        let v_proj_runner = self
            .v_proj_mul_mat
            .plan(device, queue, normed, decode_v_new_view);
        let q_norm_runner = self.q_norm.plan(device, queue, q_heads_flat_view);
        let k_norm_runner = self.k_norm.plan(device, queue, k_new_heads);
        let rope_runner = self
            .rope
            .plan(device, queue, q_view, decode_k_new_view, position_buffer);
        let scatter_k_runner = self.kv_scatter.plan(
            device,
            queue,
            decode_k_new_view,
            k_cache_view,
            position_buffer,
        );
        let scatter_v_runner = self.kv_scatter.plan(
            device,
            queue,
            decode_v_new_view,
            v_cache_view,
            position_buffer,
        );
        let attn_runner = self.gqa_attention.plan(
            device,
            queue,
            q_view,
            k_cache_view,
            v_cache_view,
            attn_out_view,
            position_buffer,
        );
        let sigmoid_mul_runner = self
            .sigmoid_mul
            .plan(device, queue, attn_out_view, gate_view);
        let o_proj_runner = self
            .o_proj_mul_mat
            .plan(device, queue, attn_out_view, o_proj_view);
        let attn_residual_runner =
            self.attn_residual_add
                .plan(device, queue, residual_slot, o_proj_view);
        let post_attn_norm_runner =
            self.post_attention_layernorm
                .plan(device, queue, residual_slot, normed);
        let mlp_runners = self.mlp.plan(device, queue, normed, mlp_out_view);
        let mlp_residual_runner =
            self.mlp_residual_add
                .plan(device, queue, residual_slot, mlp_out_view);

        SelfAttentionLayerRunner {
            input_layernorm_runner,
            q_proj_runner,
            k_proj_runner,
            v_proj_runner,
            q_norm_runner,
            k_norm_runner,
            rope_runner,
            scatter_k_runner,
            scatter_v_runner,
            attn_runner,
            sigmoid_mul_runner,
            o_proj_runner,
            attn_residual_runner,
            post_attn_norm_runner,
            mlp_runners,
            mlp_residual_runner,
        }
    }

    pub fn forward(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        residual_slot: BufferView<'_>,
        k_cache_view: BufferView<'_>,
        v_cache_view: BufferView<'_>,
        position_buffer: &wgpu::Buffer,
    ) {
        let runner = self.plan(
            device,
            queue,
            residual_slot,
            k_cache_view,
            v_cache_view,
            position_buffer,
        );
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("self_attention_layer/command_encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("self_attention_layer/compute_pass"),
                timestamp_writes: None,
            });
            runner.forward(&mut cpass);
        }
        queue.submit(Some(encoder.finish()));
    }
}

/// Cached runners for one full-attention forward pass. Records its
/// dispatches into a caller-owned compute pass via
/// [`SelfAttentionLayerRunner::forward`].
pub struct SelfAttentionLayerRunner {
    input_layernorm_runner: RmsNormWebgpuRunner,
    q_proj_runner: MulMatWebgpuRunner,
    k_proj_runner: MulMatWebgpuRunner,
    v_proj_runner: MulMatWebgpuRunner,
    q_norm_runner: RmsNormInplaceWebgpuRunner,
    k_norm_runner: RmsNormInplaceWebgpuRunner,
    rope_runner: RopeInplaceWebgpuRunner,
    scatter_k_runner: ScatterRowWebgpuRunner,
    scatter_v_runner: ScatterRowWebgpuRunner,
    attn_runner: CausalGqaNaiveAttentionWebgpuRunner,
    sigmoid_mul_runner: SigmoidMulInplaceWebgpuRunner,
    o_proj_runner: MulMatWebgpuRunner,
    attn_residual_runner: ElementwiseAddInplaceWebgpuRunner,
    post_attn_norm_runner: RmsNormWebgpuRunner,
    mlp_runners: MlpRunners,
    mlp_residual_runner: ElementwiseAddInplaceWebgpuRunner,
}

impl SelfAttentionLayerRunner {
    pub fn forward(&self, cpass: &mut wgpu::ComputePass<'_>) {
        self.input_layernorm_runner.forward(cpass);
        self.q_proj_runner.forward(cpass);
        self.k_proj_runner.forward(cpass);
        self.v_proj_runner.forward(cpass);
        self.q_norm_runner.forward(cpass);
        self.k_norm_runner.forward(cpass);
        self.rope_runner.forward(cpass);
        self.scatter_k_runner.forward(cpass);
        self.scatter_v_runner.forward(cpass);
        self.attn_runner.forward(cpass);
        self.sigmoid_mul_runner.forward(cpass);
        self.o_proj_runner.forward(cpass);
        self.attn_residual_runner.forward(cpass);
        self.post_attn_norm_runner.forward(cpass);
        self.mlp_runners.forward(cpass);
        self.mlp_residual_runner.forward(cpass);
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

    pub fn plan(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        residual_slot: BufferView<'_>,
        position_buffer: &wgpu::Buffer,
    ) -> SelfAttentionLayerRunner {
        let sz = std::mem::size_of::<f32>() as u32;
        let kv_dim = self.layer.num_key_value_heads * self.layer.head_dim;
        let max_seq_in_cache = (self.k_cache_buffer.size()
            / (kv_dim as wgpu::BufferAddress * sz as wgpu::BufferAddress))
            as u32;
        let k_cache_view = BufferView::new_3d_tight(
            &self.k_cache_buffer,
            max_seq_in_cache,
            self.layer.num_key_value_heads as u32,
            self.layer.head_dim as u32,
            sz,
        );
        let v_cache_view = BufferView::new_3d_tight(
            &self.v_cache_buffer,
            max_seq_in_cache,
            self.layer.num_key_value_heads as u32,
            self.layer.head_dim as u32,
            sz,
        );
        self.layer.plan(
            device,
            queue,
            residual_slot,
            k_cache_view,
            v_cache_view,
            position_buffer,
        )
    }
}
