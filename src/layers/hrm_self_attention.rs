use safetensors::{SafeTensors, tensor::TensorView};

use crate::{
    buffer_view::BufferView,
    kernels::{
        elementwise_add::{ElementwiseAddInplaceWebgpu, ElementwiseAddInplaceWebgpuRunner},
        mul_mat::{MulMatWebgpu, MulMatWebgpuRunner},
        norm::{ParameterlessRmsNormWebgpu, RmsNormWebgpuRunner},
        prefix_gqa_naive_attention::{
            PrefixGqaNaiveAttentionWebgpu, PrefixGqaNaiveAttentionWebgpuRunner,
        },
        rope::{RopeInplaceWebgpu, RopeInplaceWebgpuRunner},
        scatter_row::{ScatterRowWebgpu, ScatterRowWebgpuRunner},
        sigmoid_mul::{SigmoidMulInplaceWebgpu, SigmoidMulInplaceWebgpuRunner},
    },
    layers::mlp::{MlpRunners, MultiLayerPerceptron},
    log_tensor,
};

/// View a contiguous range of `num_rows` rows from a row-major bf16 2-D weight
/// tensor as its own [`TensorView`], without copying the underlying bytes.
///
/// HRM-Text fuses several projections into one checkpoint tensor:
///   * `attn.gqkv_proj` packs `[gate | query | key | value]` head-groups, and
///   * `mlp.gate_up_proj` packs `[gate | up]`.
/// Because the weight is row-major and each sub-projection occupies a
/// contiguous block of output rows, every split is a contiguous byte slice.
fn bf16_row_slice<'data>(
    weight: &TensorView<'data>,
    start_row: usize,
    num_rows: usize,
) -> TensorView<'data> {
    debug_assert_eq!(
        weight.dtype(),
        safetensors::Dtype::BF16,
        "bf16_row_slice: weight must be bf16",
    );
    let cols = weight.shape()[1];
    debug_assert!(
        start_row + num_rows <= weight.shape()[0],
        "bf16_row_slice: rows [{}, {}) exceed weight height {}",
        start_row,
        start_row + num_rows,
        weight.shape()[0],
    );
    let row_bytes = cols * std::mem::size_of::<u16>();
    let data: &'data [u8] = weight.data();
    let start = start_row * row_bytes;
    let end = (start_row + num_rows) * row_bytes;
    TensorView::new(
        safetensors::Dtype::BF16,
        vec![num_rows, cols],
        &data[start..end],
    )
    .expect("bf16_row_slice: valid sub-view")
}

/// Configuration for one HRM-Text transformer block (the shared design of both
/// the H and L recurrent stacks).
#[derive(Debug, Clone, Copy)]
pub struct HrmSelfAttentionConfig {
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub intermediate_size: usize,
}

/// One HRM-Text transformer block
pub struct HrmSelfAttentionLayer {
    hidden_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,

    rms_norm: ParameterlessRmsNormWebgpu,
    gate_proj_mul_mat: MulMatWebgpu,
    q_proj_mul_mat: MulMatWebgpu,
    k_proj_mul_mat: MulMatWebgpu,
    v_proj_mul_mat: MulMatWebgpu,
    rope: RopeInplaceWebgpu,
    kv_scatter: ScatterRowWebgpu,
    gqa_attention: PrefixGqaNaiveAttentionWebgpu,
    sigmoid_mul: SigmoidMulInplaceWebgpu,
    o_proj_mul_mat: MulMatWebgpu,
    attn_residual_add: ElementwiseAddInplaceWebgpu,
    mlp: MultiLayerPerceptron,
    mlp_residual_add: ElementwiseAddInplaceWebgpu,
}

impl HrmSelfAttentionLayer {
    pub fn new<'data>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tensor: &SafeTensors<'data>,
        weight_prefix: &str,
        hidden_size: usize,
        config: &HrmSelfAttentionConfig,
    ) -> Self {
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_key_value_heads * config.head_dim;
        // `gqkv_proj` rows are laid out as [gate | query | key | value]
        // head-groups (see the reference `gqkv.split`).
        let gqkv_out_dim = 2 * q_dim + 2 * kv_dim;

        let rms_norm = ParameterlessRmsNormWebgpu::new(device, queue, hidden_size);

        let gqkv_proj_weight_name = format!("{}.attn.gqkv_proj.weight", weight_prefix);
        let gqkv_proj_weight = tensor.tensor(&gqkv_proj_weight_name).expect(&format!(
            "Failed to get tensor for {}",
            gqkv_proj_weight_name
        ));
        log_tensor(&gqkv_proj_weight_name, &gqkv_proj_weight);
        debug_assert_eq!(
            gqkv_proj_weight.shape()[0],
            gqkv_out_dim,
            "{} height does not match 2*q_dim + 2*kv_dim",
            gqkv_proj_weight_name,
        );
        debug_assert_eq!(
            gqkv_proj_weight.shape()[1],
            hidden_size,
            "{} width does not match hidden_size",
            gqkv_proj_weight_name,
        );
        // Row groups: gate [0, q_dim), query [q_dim, 2*q_dim),
        // key [2*q_dim, 2*q_dim+kv_dim), value [.., 2*q_dim+2*kv_dim).
        let gate_w = bf16_row_slice(&gqkv_proj_weight, 0, q_dim);
        let q_w = bf16_row_slice(&gqkv_proj_weight, q_dim, q_dim);
        let k_w = bf16_row_slice(&gqkv_proj_weight, 2 * q_dim, kv_dim);
        let v_w = bf16_row_slice(&gqkv_proj_weight, 2 * q_dim + kv_dim, kv_dim);
        let gate_proj_mul_mat = MulMatWebgpu::new(device, queue, gate_w);
        let q_proj_mul_mat = MulMatWebgpu::new(device, queue, q_w);
        let k_proj_mul_mat = MulMatWebgpu::new(device, queue, k_w);
        let v_proj_mul_mat = MulMatWebgpu::new(device, queue, v_w);

        // HRM uses full RoPE (every head dimension rotated).
        let rope = RopeInplaceWebgpu::new(
            device,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.rope_theta,
            1.0,
        );
        let gqa_attention = PrefixGqaNaiveAttentionWebgpu::new(
            device,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.head_dim,
        );
        let kv_scatter = ScatterRowWebgpu::new(device, kv_dim);
        let sigmoid_mul =
            SigmoidMulInplaceWebgpu::new(device, config.num_attention_heads, config.head_dim);

        let o_proj_weight_name = format!("{}.attn.o_proj.weight", weight_prefix);
        let o_proj_weight = tensor
            .tensor(&o_proj_weight_name)
            .expect(&format!("Failed to get tensor for {}", o_proj_weight_name));
        log_tensor(&o_proj_weight_name, &o_proj_weight);
        debug_assert_eq!(
            o_proj_weight.shape()[0],
            hidden_size,
            "{} height does not match hidden_size",
            o_proj_weight_name,
        );
        debug_assert_eq!(
            o_proj_weight.shape()[1],
            q_dim,
            "{} width does not match num_attention_heads * head_dim",
            o_proj_weight_name,
        );
        let o_proj_mul_mat = MulMatWebgpu::new(device, queue, o_proj_weight);
        let attn_residual_add = ElementwiseAddInplaceWebgpu::new(device, hidden_size);

        // SwiGLU MLP with a fused `gate_up_proj` ([gate | up] row groups).
        let gate_up_proj_weight_name = format!("{}.mlp.gate_up_proj.weight", weight_prefix);
        let gate_up_proj_weight = tensor.tensor(&gate_up_proj_weight_name).expect(&format!(
            "Failed to get tensor for {}",
            gate_up_proj_weight_name
        ));
        log_tensor(&gate_up_proj_weight_name, &gate_up_proj_weight);
        debug_assert_eq!(
            gate_up_proj_weight.shape()[0],
            2 * config.intermediate_size,
            "{} height does not match 2 * intermediate_size",
            gate_up_proj_weight_name,
        );
        debug_assert_eq!(
            gate_up_proj_weight.shape()[1],
            hidden_size,
            "{} width does not match hidden_size",
            gate_up_proj_weight_name,
        );
        let mlp_gate_w = bf16_row_slice(&gate_up_proj_weight, 0, config.intermediate_size);
        let mlp_up_w = bf16_row_slice(
            &gate_up_proj_weight,
            config.intermediate_size,
            config.intermediate_size,
        );
        let down_proj_weight_name = format!("{}.mlp.down_proj.weight", weight_prefix);
        let down_proj_weight = tensor.tensor(&down_proj_weight_name).expect(&format!(
            "Failed to get tensor for {}",
            down_proj_weight_name
        ));
        log_tensor(&down_proj_weight_name, &down_proj_weight);
        let mlp = MultiLayerPerceptron::from_weights(
            device,
            queue,
            mlp_gate_w,
            mlp_up_w,
            down_proj_weight,
            hidden_size,
            config.intermediate_size,
        );
        let mlp_residual_add = ElementwiseAddInplaceWebgpu::new(device, hidden_size);

        Self {
            hidden_size,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            rms_norm,
            gate_proj_mul_mat,
            q_proj_mul_mat,
            k_proj_mul_mat,
            v_proj_mul_mat,
            rope,
            kv_scatter,
            gqa_attention,
            sigmoid_mul,
            o_proj_mul_mat,
            attn_residual_add,
            mlp,
            mlp_residual_add,
        }
    }

    /// Number of query heads in this attention block.
    pub fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }

    /// Number of key/value heads in this attention block.
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads
    }

    /// Per-head dimension shared by Q, K and V.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Build an [`HrmSelfAttentionLayerRunner`] that records this block's
    /// dispatches into a caller-owned compute pass.
    pub fn plan(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        residual_slot: BufferView<'_>,
        k_cache_view: BufferView<'_>,
        v_cache_view: BufferView<'_>,
        position_buffer: &wgpu::Buffer,
        prefix_buffer: &wgpu::Buffer,
    ) -> HrmSelfAttentionLayerRunner {
        let num_new_tokens = residual_slot.shape[0] as usize;
        debug_assert!(
            num_new_tokens >= 1,
            "forward requires residual_slot.shape[0] >= 1, got {}",
            num_new_tokens,
        );
        let q_dim = self.num_attention_heads * self.head_dim;
        let kv_dim = self.num_key_value_heads * self.head_dim;
        let sz = std::mem::size_of::<f32>() as u32;
        let num_new_u32 = num_new_tokens as u32;
        let hidden_size = self.hidden_size as u32;
        debug_assert_eq!(
            k_cache_view.rank, 3,
            "hrm_self_attention: k_cache must be rank-3"
        );
        debug_assert_eq!(
            v_cache_view.rank, 3,
            "hrm_self_attention: v_cache must be rank-3"
        );
        debug_assert_eq!(
            k_cache_view.shape[1] as usize, self.num_key_value_heads,
            "hrm_self_attention: k_cache shape[1] ({}) must equal num_key_value_heads ({})",
            k_cache_view.shape[1], self.num_key_value_heads,
        );
        debug_assert_eq!(
            v_cache_view.shape[1] as usize, self.num_key_value_heads,
            "hrm_self_attention: v_cache shape[1] ({}) must equal num_key_value_heads ({})",
            v_cache_view.shape[1], self.num_key_value_heads,
        );
        debug_assert_eq!(
            k_cache_view.shape[2] as usize, self.head_dim,
            "hrm_self_attention: k_cache shape[2] ({}) must equal head_dim ({})",
            k_cache_view.shape[2], self.head_dim,
        );
        debug_assert_eq!(
            v_cache_view.shape[2] as usize, self.head_dim,
            "hrm_self_attention: v_cache shape[2] ({}) must equal head_dim ({})",
            v_cache_view.shape[2], self.head_dim,
        );

        let make_storage = |label: &str, elems: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (elems * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };

        let normed_buffer = make_storage(
            "hrm_self_attention/normed_buffer",
            num_new_tokens * self.hidden_size,
        );
        let normed = BufferView::new_2d_tight(&normed_buffer, num_new_u32, hidden_size, sz);

        let gate_proj_buffer = make_storage(
            "hrm_self_attention/gate_proj_buffer",
            num_new_tokens * q_dim,
        );
        let gate_view = BufferView::new_3d_tight(
            &gate_proj_buffer,
            num_new_u32,
            self.num_attention_heads as u32,
            self.head_dim as u32,
            sz,
        );

        let q_proj_buffer =
            make_storage("hrm_self_attention/q_proj_buffer", num_new_tokens * q_dim);
        let q_view = BufferView::new_3d_tight(
            &q_proj_buffer,
            num_new_u32,
            self.num_attention_heads as u32,
            self.head_dim as u32,
            sz,
        );

        let attn_output_buffer = make_storage(
            "hrm_self_attention/attn_output_buffer",
            num_new_tokens * q_dim,
        );
        let attn_out_view = BufferView::new_3d_tight(
            &attn_output_buffer,
            num_new_u32,
            self.num_attention_heads as u32,
            self.head_dim as u32,
            sz,
        );

        let o_proj_buffer = make_storage(
            "hrm_self_attention/o_proj_buffer",
            num_new_tokens * self.hidden_size,
        );
        let o_proj_view = BufferView::new_2d_tight(&o_proj_buffer, num_new_u32, hidden_size, sz);

        let mlp_output_buffer = make_storage(
            "hrm_self_attention/mlp_output_buffer",
            num_new_tokens * self.hidden_size,
        );
        let mlp_out_view =
            BufferView::new_2d_tight(&mlp_output_buffer, num_new_u32, hidden_size, sz);

        let decode_k_new_buffer = make_storage(
            "hrm_self_attention/decode_k_new_buffer",
            num_new_tokens * kv_dim,
        );
        let decode_v_new_buffer = make_storage(
            "hrm_self_attention/decode_v_new_buffer",
            num_new_tokens * kv_dim,
        );
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

        // Attention sub-layer (pre-norm): normed = rms_norm(residual).
        let input_layernorm_runner = self.rms_norm.plan(device, queue, residual_slot, normed);
        let gate_proj_runner = self
            .gate_proj_mul_mat
            .plan(device, queue, normed, gate_view);
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
            prefix_buffer,
        );
        // Sigmoid output gate: attn_out *= sigmoid(gate).
        let sigmoid_mul_runner = self
            .sigmoid_mul
            .plan(device, queue, attn_out_view, gate_view);
        let o_proj_runner = self
            .o_proj_mul_mat
            .plan(device, queue, attn_out_view, o_proj_view);
        let attn_residual_runner =
            self.attn_residual_add
                .plan(device, queue, residual_slot, o_proj_view);

        // MLP sub-layer (pre-norm): normed = rms_norm(residual) reusing the
        // same shared parameterless norm and `normed` scratch buffer.
        let post_attn_norm_runner = self.rms_norm.plan(device, queue, residual_slot, normed);
        let mlp_runners = self.mlp.plan(device, queue, normed, mlp_out_view);
        let mlp_residual_runner =
            self.mlp_residual_add
                .plan(device, queue, residual_slot, mlp_out_view);

        HrmSelfAttentionLayerRunner {
            input_layernorm_runner,
            gate_proj_runner,
            q_proj_runner,
            k_proj_runner,
            v_proj_runner,
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
}

/// Cached runners for one HRM-Text block forward pass. Records its dispatches
/// into a caller-owned compute pass via [`HrmSelfAttentionLayerRunner::forward`].
pub struct HrmSelfAttentionLayerRunner {
    input_layernorm_runner: RmsNormWebgpuRunner,
    gate_proj_runner: MulMatWebgpuRunner,
    q_proj_runner: MulMatWebgpuRunner,
    k_proj_runner: MulMatWebgpuRunner,
    v_proj_runner: MulMatWebgpuRunner,
    rope_runner: RopeInplaceWebgpuRunner,
    scatter_k_runner: ScatterRowWebgpuRunner,
    scatter_v_runner: ScatterRowWebgpuRunner,
    attn_runner: PrefixGqaNaiveAttentionWebgpuRunner,
    sigmoid_mul_runner: SigmoidMulInplaceWebgpuRunner,
    o_proj_runner: MulMatWebgpuRunner,
    attn_residual_runner: ElementwiseAddInplaceWebgpuRunner,
    post_attn_norm_runner: RmsNormWebgpuRunner,
    mlp_runners: MlpRunners,
    mlp_residual_runner: ElementwiseAddInplaceWebgpuRunner,
}

impl HrmSelfAttentionLayerRunner {
    pub fn forward(&self, cpass: &mut wgpu::ComputePass<'_>) {
        self.input_layernorm_runner.forward(cpass);
        self.gate_proj_runner.forward(cpass);
        self.q_proj_runner.forward(cpass);
        self.k_proj_runner.forward(cpass);
        self.v_proj_runner.forward(cpass);
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

/// Per-layer, per-sequence state: the K/V cache pair for one HRM-Text block.
pub struct HrmSelfAttentionLayerSession<'m> {
    layer: &'m HrmSelfAttentionLayer,
    k_cache_buffer: wgpu::Buffer,
    v_cache_buffer: wgpu::Buffer,
    max_seq_len: usize,
}

impl<'m> HrmSelfAttentionLayerSession<'m> {
    pub fn new(
        layer: &'m HrmSelfAttentionLayer,
        device: &wgpu::Device,
        max_seq_len: usize,
    ) -> Self {
        debug_assert!(max_seq_len >= 1, "max_seq_len must be >= 1");
        let kv_dim = layer.num_key_value_heads * layer.head_dim;
        let cache_bytes =
            (max_seq_len * kv_dim * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
        let k_cache_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hrm_self_attention/session/k_cache_buffer"),
            size: cache_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let v_cache_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hrm_self_attention/session/v_cache_buffer"),
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
            max_seq_len,
        }
    }

    fn kv_cache_view<'a>(
        buffer: &'a wgpu::Buffer,
        max_seq_len: usize,
        layer: &HrmSelfAttentionLayer,
    ) -> BufferView<'a> {
        let sz = std::mem::size_of::<f32>() as u32;
        BufferView::new_3d_tight(
            buffer,
            max_seq_len as u32,
            layer.num_key_value_heads as u32,
            layer.head_dim as u32,
            sz,
        )
    }

    /// Build an [`HrmSelfAttentionLayerRunner`] over this session's K/V cache.
    pub fn plan(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        residual_slot: BufferView<'_>,
        position_buffer: &wgpu::Buffer,
        prefix_buffer: &wgpu::Buffer,
    ) -> HrmSelfAttentionLayerRunner {
        let k_cache_view = Self::kv_cache_view(&self.k_cache_buffer, self.max_seq_len, self.layer);
        let v_cache_view = Self::kv_cache_view(&self.v_cache_buffer, self.max_seq_len, self.layer);
        self.layer.plan(
            device,
            queue,
            residual_slot,
            k_cache_view,
            v_cache_view,
            position_buffer,
            prefix_buffer,
        )
    }
}
