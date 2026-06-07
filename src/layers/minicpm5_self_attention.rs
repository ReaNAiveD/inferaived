use safetensors::SafeTensors;

use crate::{
    buffer_view::BufferView,
    kernels::{
        attention::{CausalGqaNaiveAttentionWebgpu, CausalGqaNaiveAttentionWebgpuRunner},
        elementwise_add::{ElementwiseAddInplaceWebgpu, ElementwiseAddInplaceWebgpuRunner},
        masked_block_attention::{MaskedBlockAttentionWebgpu, MaskedBlockAttentionWebgpuRunner},
        mul_mat::{MulMatWebgpu, MulMatWebgpuRunner},
        norm::{LlamaRmsNormWebgpu, RmsNormWebgpuRunner},
        rope::{RopeInplaceWebgpu, RopeInplaceWebgpuRunner},
        scatter_row::{ScatterRowWebgpu, ScatterRowWebgpuRunner},
    },
    layers::mlp::{MlpRunners, MultiLayerPerceptron},
    log_tensor,
};

#[derive(Clone, Copy)]
pub struct MaskedKvPool<'a> {
    /// Shared K pool, `[pool_rows, num_kv_heads, head_dim]`.
    pub k: BufferView<'a>,
    /// Shared V pool, `[pool_rows, num_kv_heads, head_dim]`.
    pub v: BufferView<'a>,
    /// `[1]` u32 base pool-row; new token `i` scatters into row `base + i`.
    pub scatter_position: &'a wgpu::Buffer,
    /// Flat visible-range list (see
    /// [`encode_visibility`](crate::kernels::masked_block_attention::encode_visibility))
    /// selecting which pool rows the new tokens attend.
    pub visibility: &'a wgpu::Buffer,
}

/// Configuration for one MiniCPM5 (vanilla `LlamaForCausalLM`)
/// full-attention block.
#[derive(Debug, Clone, Copy)]
pub struct MiniCPM5SelfAttentionConfig {
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub intermediate_size: usize,
}

/// One MiniCPM5 transformer block: input RMSNorm → Q + K + V projections →
/// RoPE → KV-cache scatter → causal GQA attention → `o_proj` → residual →
/// post-attention RMSNorm → SwiGLU MLP → residual.
pub struct MiniCPM5SelfAttentionLayer {
    hidden_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,

    input_layernorm: LlamaRmsNormWebgpu,
    q_proj_mul_mat: MulMatWebgpu,
    k_proj_mul_mat: MulMatWebgpu,
    v_proj_mul_mat: MulMatWebgpu,
    rope: RopeInplaceWebgpu,
    kv_scatter: ScatterRowWebgpu,
    gqa_attention: CausalGqaNaiveAttentionWebgpu,
    masked_attention: MaskedBlockAttentionWebgpu,
    o_proj_mul_mat: MulMatWebgpu,
    attn_residual_add: ElementwiseAddInplaceWebgpu,
    post_attention_layernorm: LlamaRmsNormWebgpu,
    mlp: MultiLayerPerceptron,
    mlp_residual_add: ElementwiseAddInplaceWebgpu,
}

impl MiniCPM5SelfAttentionLayer {
    pub fn new<'data>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tensor: &SafeTensors<'data>,
        weight_prefix: &str,
        hidden_size: usize,
        config: &MiniCPM5SelfAttentionConfig,
    ) -> Self {
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_key_value_heads * config.head_dim;
        let input_layernorm_weight_name = format!("{}.input_layernorm.weight", weight_prefix);
        let input_layernorm_weight = tensor.tensor(&input_layernorm_weight_name).expect(&format!(
            "Failed to get tensor for {}",
            input_layernorm_weight_name
        ));
        log_tensor(&input_layernorm_weight_name, &input_layernorm_weight);
        let input_layernorm = LlamaRmsNormWebgpu::new(device, queue, input_layernorm_weight);
        let q_proj_weight_name = format!("{}.self_attn.q_proj.weight", weight_prefix);
        let q_proj_weight = tensor
            .tensor(&q_proj_weight_name)
            .expect(&format!("Failed to get tensor for {}", q_proj_weight_name));
        log_tensor(&q_proj_weight_name, &q_proj_weight);
        debug_assert_eq!(
            q_proj_weight.shape()[0] as usize,
            q_dim,
            "{} height does not match num_attention_heads * head_dim",
            q_proj_weight_name,
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
        // MiniCPM5 uses full RoPE; `partial_rotary_factor == 1.0` rotates
        // every dimension of each head.
        let rope = RopeInplaceWebgpu::new(
            device,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.rope_theta,
            1.0,
        );
        let gqa_attention = CausalGqaNaiveAttentionWebgpu::new(
            device,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.head_dim,
        );
        let masked_attention = MaskedBlockAttentionWebgpu::new(
            device,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.head_dim,
        );
        let kv_scatter = ScatterRowWebgpu::new(device, kv_dim);
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
            LlamaRmsNormWebgpu::new(&device, &queue, post_attention_layernorm_weight);
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
            rope,
            kv_scatter,
            gqa_attention,
            masked_attention,
            o_proj_mul_mat,
            attn_residual_add,
            post_attention_layernorm,
            mlp,
            mlp_residual_add,
        }
    }

    /// Number of query heads in this attention block.
    pub fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }

    /// Number of key/value heads (GQA group count) in this attention block.
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads
    }

    /// Per-head dimension shared by Q, K and V.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Build a [`MiniCPM5SelfAttentionLayerRunner`] that records this
    /// block's dispatches into a caller-owned compute pass.
    pub fn plan(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        residual_slot: BufferView<'_>,
        k_cache_view: BufferView<'_>,
        v_cache_view: BufferView<'_>,
        position_buffer: &wgpu::Buffer,
    ) -> MiniCPM5SelfAttentionLayerRunner {
        let num_new_tokens = residual_slot.shape[0] as usize;
        debug_assert!(
            num_new_tokens >= 1,
            "forward requires residual_slot.shape[0] >= 1, got {}",
            num_new_tokens,
        );
        let q_dim = self.num_attention_heads * self.head_dim;
        let sz = std::mem::size_of::<f32>() as u32;
        let num_new_u32 = num_new_tokens as u32;
        let hidden_size = self.hidden_size as u32;
        let kv_dim = self.num_key_value_heads * self.head_dim;
        debug_assert_eq!(
            k_cache_view.rank, 3,
            "minicpm5_self_attention: k_cache must be rank-3"
        );
        debug_assert_eq!(
            v_cache_view.rank, 3,
            "minicpm5_self_attention: v_cache must be rank-3"
        );
        debug_assert_eq!(
            k_cache_view.shape[0], v_cache_view.shape[0],
            "minicpm5_self_attention: k/v cache max_seq mismatch (k={}, v={})",
            k_cache_view.shape[0], v_cache_view.shape[0],
        );
        debug_assert_eq!(
            k_cache_view.shape[1] as usize, self.num_key_value_heads,
            "minicpm5_self_attention: k_cache shape[1] ({}) must equal num_key_value_heads ({})",
            k_cache_view.shape[1], self.num_key_value_heads,
        );
        debug_assert_eq!(
            v_cache_view.shape[1] as usize, self.num_key_value_heads,
            "minicpm5_self_attention: v_cache shape[1] ({}) must equal num_key_value_heads ({})",
            v_cache_view.shape[1], self.num_key_value_heads,
        );
        debug_assert_eq!(
            k_cache_view.shape[2] as usize, self.head_dim,
            "minicpm5_self_attention: k_cache shape[2] ({}) must equal head_dim ({})",
            k_cache_view.shape[2], self.head_dim,
        );
        debug_assert_eq!(
            v_cache_view.shape[2] as usize, self.head_dim,
            "minicpm5_self_attention: v_cache shape[2] ({}) must equal head_dim ({})",
            v_cache_view.shape[2], self.head_dim,
        );

        let normed_embedding_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("minicpm5_self_attention_layer/normed_embedding_buffer"),
            size: (num_new_tokens * self.hidden_size * std::mem::size_of::<f32>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let normed =
            BufferView::new_2d_tight(&normed_embedding_buffer, num_new_u32, hidden_size, sz);

        let q_proj_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("minicpm5_self_attention_layer/q_proj_buffer"),
            size: (num_new_tokens * q_dim * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Plain Llama Q: a 3-D `(tokens, heads, head_dim)` tensor, no gate.
        let q_view = BufferView::new_3d_tight(
            &q_proj_buffer,
            num_new_u32,
            self.num_attention_heads as u32,
            self.head_dim as u32,
            sz,
        );

        let attn_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("minicpm5_self_attention_layer/attn_output_buffer"),
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
            label: Some("minicpm5_self_attention_layer/o_proj_buffer"),
            size: (num_new_tokens * self.hidden_size * std::mem::size_of::<f32>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let o_proj_view = BufferView::new_2d_tight(&o_proj_buffer, num_new_u32, hidden_size, sz);

        let mlp_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("minicpm5_self_attention_layer/mlp_output_buffer"),
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
            label: Some("minicpm5_self_attention_layer/decode_k_new_buffer"),
            size: decode_kv_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let decode_v_new_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("minicpm5_self_attention_layer/decode_v_new_buffer"),
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

        let input_layernorm_runner =
            self.input_layernorm
                .plan(device, queue, residual_slot, normed);
        let q_proj_runner = self.q_proj_mul_mat.plan(device, queue, normed, q_view);
        let k_proj_runner = self
            .k_proj_mul_mat
            .plan(device, queue, normed, decode_k_new_view);
        let v_proj_runner = self
            .v_proj_mul_mat
            .plan(device, queue, normed, decode_v_new_view);
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

        MiniCPM5SelfAttentionLayerRunner {
            input_layernorm_runner,
            q_proj_runner,
            k_proj_runner,
            v_proj_runner,
            rope_runner,
            scatter_k_runner,
            scatter_v_runner,
            attn_runner,
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
            label: Some("minicpm5_self_attention_layer/command_encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("minicpm5_self_attention_layer/compute_pass"),
                timestamp_writes: None,
            });
            runner.forward(&mut cpass);
        }
        queue.submit(Some(encoder.finish()));
    }

    pub fn plan_masked(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        residual_slot: BufferView<'_>,
        kv_pool: MaskedKvPool<'_>,
        rope_position_buffer: &wgpu::Buffer,
    ) -> MiniCPM5MaskedSelfAttentionLayerRunner {
        let MaskedKvPool {
            k: pool_k_view,
            v: pool_v_view,
            scatter_position: scatter_position_buffer,
            visibility,
        } = kv_pool;
        let num_new_tokens = residual_slot.shape[0] as usize;
        debug_assert!(
            num_new_tokens >= 1,
            "plan_masked requires residual_slot.shape[0] >= 1, got {}",
            num_new_tokens,
        );
        let q_dim = self.num_attention_heads * self.head_dim;
        let sz = std::mem::size_of::<f32>() as u32;
        let num_new_u32 = num_new_tokens as u32;
        let hidden_size = self.hidden_size as u32;
        let kv_dim = self.num_key_value_heads * self.head_dim;
        debug_assert_eq!(
            pool_k_view.rank, 3,
            "minicpm5_self_attention: pool_k must be rank-3"
        );
        debug_assert_eq!(
            pool_v_view.rank, 3,
            "minicpm5_self_attention: pool_v must be rank-3"
        );
        debug_assert_eq!(
            pool_k_view.shape[1] as usize, self.num_key_value_heads,
            "minicpm5_self_attention: pool_k shape[1] ({}) must equal num_key_value_heads ({})",
            pool_k_view.shape[1], self.num_key_value_heads,
        );
        debug_assert_eq!(
            pool_k_view.shape[2] as usize, self.head_dim,
            "minicpm5_self_attention: pool_k shape[2] ({}) must equal head_dim ({})",
            pool_k_view.shape[2], self.head_dim,
        );

        let normed_embedding_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("minicpm5_self_attention_layer/masked_normed_buffer"),
            size: (num_new_tokens * self.hidden_size * std::mem::size_of::<f32>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let normed =
            BufferView::new_2d_tight(&normed_embedding_buffer, num_new_u32, hidden_size, sz);

        let q_proj_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("minicpm5_self_attention_layer/masked_q_proj_buffer"),
            size: (num_new_tokens * q_dim * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let q_view = BufferView::new_3d_tight(
            &q_proj_buffer,
            num_new_u32,
            self.num_attention_heads as u32,
            self.head_dim as u32,
            sz,
        );

        let attn_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("minicpm5_self_attention_layer/masked_attn_output_buffer"),
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
            label: Some("minicpm5_self_attention_layer/masked_o_proj_buffer"),
            size: (num_new_tokens * self.hidden_size * std::mem::size_of::<f32>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let o_proj_view = BufferView::new_2d_tight(&o_proj_buffer, num_new_u32, hidden_size, sz);

        let mlp_output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("minicpm5_self_attention_layer/masked_mlp_output_buffer"),
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
            label: Some("minicpm5_self_attention_layer/masked_decode_k_new_buffer"),
            size: decode_kv_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let decode_v_new_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("minicpm5_self_attention_layer/masked_decode_v_new_buffer"),
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

        let input_layernorm_runner =
            self.input_layernorm
                .plan(device, queue, residual_slot, normed);
        let q_proj_runner = self.q_proj_mul_mat.plan(device, queue, normed, q_view);
        let k_proj_runner = self
            .k_proj_mul_mat
            .plan(device, queue, normed, decode_k_new_view);
        let v_proj_runner = self
            .v_proj_mul_mat
            .plan(device, queue, normed, decode_v_new_view);
        let rope_runner = self.rope.plan(
            device,
            queue,
            q_view,
            decode_k_new_view,
            rope_position_buffer,
        );
        let scatter_k_runner = self.kv_scatter.plan(
            device,
            queue,
            decode_k_new_view,
            pool_k_view,
            scatter_position_buffer,
        );
        let scatter_v_runner = self.kv_scatter.plan(
            device,
            queue,
            decode_v_new_view,
            pool_v_view,
            scatter_position_buffer,
        );
        let attn_runner = self.masked_attention.plan(
            device,
            queue,
            q_view,
            pool_k_view,
            pool_v_view,
            attn_out_view,
            visibility,
            scatter_position_buffer,
        );
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

        MiniCPM5MaskedSelfAttentionLayerRunner {
            input_layernorm_runner,
            q_proj_runner,
            k_proj_runner,
            v_proj_runner,
            rope_runner,
            scatter_k_runner,
            scatter_v_runner,
            attn_runner,
            o_proj_runner,
            attn_residual_runner,
            post_attn_norm_runner,
            mlp_runners,
            mlp_residual_runner,
        }
    }
}

/// Cached runners for one MiniCPM5 masked-attention forward pass over a
/// shared KV pool. Records its dispatches into a caller-owned compute
/// pass via [`MiniCPM5MaskedSelfAttentionLayerRunner::forward`].
pub struct MiniCPM5MaskedSelfAttentionLayerRunner {
    input_layernorm_runner: RmsNormWebgpuRunner,
    q_proj_runner: MulMatWebgpuRunner,
    k_proj_runner: MulMatWebgpuRunner,
    v_proj_runner: MulMatWebgpuRunner,
    rope_runner: RopeInplaceWebgpuRunner,
    scatter_k_runner: ScatterRowWebgpuRunner,
    scatter_v_runner: ScatterRowWebgpuRunner,
    attn_runner: MaskedBlockAttentionWebgpuRunner,
    o_proj_runner: MulMatWebgpuRunner,
    attn_residual_runner: ElementwiseAddInplaceWebgpuRunner,
    post_attn_norm_runner: RmsNormWebgpuRunner,
    mlp_runners: MlpRunners,
    mlp_residual_runner: ElementwiseAddInplaceWebgpuRunner,
}

impl MiniCPM5MaskedSelfAttentionLayerRunner {
    pub fn forward(&self, cpass: &mut wgpu::ComputePass<'_>) {
        self.input_layernorm_runner.forward(cpass);
        self.q_proj_runner.forward(cpass);
        self.k_proj_runner.forward(cpass);
        self.v_proj_runner.forward(cpass);
        self.rope_runner.forward(cpass);
        self.scatter_k_runner.forward(cpass);
        self.scatter_v_runner.forward(cpass);
        self.attn_runner.forward(cpass);
        self.o_proj_runner.forward(cpass);
        self.attn_residual_runner.forward(cpass);
        self.post_attn_norm_runner.forward(cpass);
        self.mlp_runners.forward(cpass);
        self.mlp_residual_runner.forward(cpass);
    }
}

/// Cached runners for one MiniCPM5 full-attention forward pass. Records
/// its dispatches into a caller-owned compute pass via
/// [`MiniCPM5SelfAttentionLayerRunner::forward`].
pub struct MiniCPM5SelfAttentionLayerRunner {
    input_layernorm_runner: RmsNormWebgpuRunner,
    q_proj_runner: MulMatWebgpuRunner,
    k_proj_runner: MulMatWebgpuRunner,
    v_proj_runner: MulMatWebgpuRunner,
    rope_runner: RopeInplaceWebgpuRunner,
    scatter_k_runner: ScatterRowWebgpuRunner,
    scatter_v_runner: ScatterRowWebgpuRunner,
    attn_runner: CausalGqaNaiveAttentionWebgpuRunner,
    o_proj_runner: MulMatWebgpuRunner,
    attn_residual_runner: ElementwiseAddInplaceWebgpuRunner,
    post_attn_norm_runner: RmsNormWebgpuRunner,
    mlp_runners: MlpRunners,
    mlp_residual_runner: ElementwiseAddInplaceWebgpuRunner,
}

impl MiniCPM5SelfAttentionLayerRunner {
    pub fn forward(&self, cpass: &mut wgpu::ComputePass<'_>) {
        self.input_layernorm_runner.forward(cpass);
        self.q_proj_runner.forward(cpass);
        self.k_proj_runner.forward(cpass);
        self.v_proj_runner.forward(cpass);
        self.rope_runner.forward(cpass);
        self.scatter_k_runner.forward(cpass);
        self.scatter_v_runner.forward(cpass);
        self.attn_runner.forward(cpass);
        self.o_proj_runner.forward(cpass);
        self.attn_residual_runner.forward(cpass);
        self.post_attn_norm_runner.forward(cpass);
        self.mlp_runners.forward(cpass);
        self.mlp_residual_runner.forward(cpass);
    }
}

pub struct MiniCPM5SelfAttentionLayerSession<'m> {
    layer: &'m MiniCPM5SelfAttentionLayer,
    k_cache_buffer: wgpu::Buffer,
    v_cache_buffer: wgpu::Buffer,
}

impl<'m> MiniCPM5SelfAttentionLayerSession<'m> {
    pub fn new(
        layer: &'m MiniCPM5SelfAttentionLayer,
        device: &wgpu::Device,
        max_seq_len: usize,
    ) -> Self {
        debug_assert!(max_seq_len >= 1, "max_seq_len must be >= 1");
        let kv_dim = layer.num_key_value_heads * layer.head_dim;
        let cache_bytes =
            (max_seq_len * kv_dim * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
        let k_cache_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("minicpm5_self_attention_layer/session/k_cache_buffer"),
            size: cache_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let v_cache_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("minicpm5_self_attention_layer/session/v_cache_buffer"),
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
    ) -> MiniCPM5SelfAttentionLayerRunner {
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
