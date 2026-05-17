//! Shared, per-session pool of GPU scratch buffers.
//!
//! Every `LayerSession::forward` used to allocate ~10 transient
//! `wgpu::Buffer`s per call (~240 / decode step across the 24-layer
//! stack). All of those scratches have the same lifetime — they're born
//! at the top of `forward` and dead at the bottom — and successive
//! layers' scratch lifetimes never overlap, so a single shared set of
//! buffers reused across every layer and every forward call is enough
//! to eliminate the per-step `device.create_buffer` cost.
//!
//! ## Why named slots, not a single bump arena
//!
//! `wgpu`'s validation layer tracks buffer state at whole-buffer
//! granularity (see `docs/wgpu-single-buffer-arena.md`), so binding two
//! disjoint byte ranges of the same buffer as `Storage(read_only=true)`
//! and `Storage(read_only=false)` in the same dispatch is rejected even
//! though the ranges don't physically overlap. That kills the obvious
//! "one buffer + bump cursor" arena.
//!
//! This pool sidesteps the constraint by giving each logical scratch
//! tensor its own dedicated `wgpu::Buffer`. Two scratches can never
//! alias because they live in different `wgpu::Buffer` objects, so the
//! tracker has nothing to merge. Memory cost at current scales is
//! trivial: at `max_seq_len = 32, hidden_size = 1024` the whole pool is
//! under 3 MB. A future upgrade to liveness-based coloring would
//! replace the `[wgpu::Buffer; NUM_SLOTS]` storage and the body of
//! `buffer()` while keeping the public API identical.
//!
//! ## Lifetime model
//!
//! One pool per `Qwen35Session`. Sized at construction for
//! `max_seq_len`; never resized. Each call to a layer's `forward`
//! borrows `&ScratchPool` and discards the borrow at the end of the
//! call. Layers never store the reference. This avoids self-referential
//! structs at the session level.

use crate::buffer_view::BufferView;

/// One variant per logical scratch tensor used anywhere in the forward
/// pass of any layer type. Variants are named by their **role** in the
/// computation, not by which layer type owns them — `MlpGate` is one
/// slot used by both self-attention and linear-attention blocks
/// because the two MLPs never run concurrently.
///
/// Adding a new scratch tensor: add a variant, update `NUM_SLOTS` and
/// `ALL_SLOTS`, and add its `feature_dim_for` arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ScratchSlot {
    /// Input-layernorm and post-attention-layernorm output. Consumed by
    /// q/k/v_proj (self) or in_proj_qkv/a/b/z (linear) and by the MLP
    /// gate/up projections. Feature dim = `hidden_size`.
    Normed = 0,
    /// Fused Q+gate projection output of self-attention. Feature dim =
    /// `num_attention_heads * 2 * head_dim`. Logical shape is 4-D
    /// `[T, heads, 2, head_dim]`; callers build that view themselves.
    QGate,
    /// Causal-GQA attention output before output gating and o_proj.
    /// Feature dim = `num_attention_heads * head_dim`.
    AttnOutputSelf,
    /// o_proj output of self-attention; consumed by attn-residual-add.
    /// Feature dim = `hidden_size`.
    OProj,
    /// Linear-attention fused QKV projection output. Feature dim =
    /// `qkv_dim` = q_dim + k_dim + v_dim.
    InProjQkv,
    /// Linear-attention "alpha" projection output. Feature dim =
    /// `linear_num_value_heads`.
    InProjA,
    /// Linear-attention "beta" projection output. Feature dim =
    /// `linear_num_value_heads`.
    InProjB,
    /// Linear-attention "z" gating projection output. Feature dim =
    /// `v_dim`.
    InProjZ,
    /// Output of the depthwise-causal-conv + SiLU stage; consumed by
    /// the delta-rule kernel. Feature dim = `qkv_dim`.
    ConvQkv,
    /// Delta-rule output, before gated-norm and out_proj. Feature dim =
    /// `v_dim`.
    AttnOutputLinear,
    /// out_proj output of linear-attention; consumed by attn-residual-add.
    /// Feature dim = `hidden_size`.
    OutProj,
    /// MLP gate projection output. Feature dim = `intermediate_size`.
    MlpGate,
    /// MLP up projection output; also receives the in-place SiLU-mul
    /// with the gate. Feature dim = `intermediate_size`.
    MlpUp,
    /// MLP down_proj output; consumed by mlp-residual-add. Feature dim
    /// = `hidden_size`.
    MlpOutput,
}

const NUM_SLOTS: usize = 14;

const ALL_SLOTS: [ScratchSlot; NUM_SLOTS] = [
    ScratchSlot::Normed,
    ScratchSlot::QGate,
    ScratchSlot::AttnOutputSelf,
    ScratchSlot::OProj,
    ScratchSlot::InProjQkv,
    ScratchSlot::InProjA,
    ScratchSlot::InProjB,
    ScratchSlot::InProjZ,
    ScratchSlot::ConvQkv,
    ScratchSlot::AttnOutputLinear,
    ScratchSlot::OutProj,
    ScratchSlot::MlpGate,
    ScratchSlot::MlpUp,
    ScratchSlot::MlpOutput,
];

impl ScratchSlot {
    #[inline]
    fn index(self) -> usize {
        self as usize
    }

    /// Stable, RenderDoc-friendly name used as the wgpu buffer label.
    fn label(self) -> &'static str {
        match self {
            ScratchSlot::Normed => "scratch_pool/normed",
            ScratchSlot::QGate => "scratch_pool/q_gate",
            ScratchSlot::AttnOutputSelf => "scratch_pool/attn_output_self",
            ScratchSlot::OProj => "scratch_pool/o_proj",
            ScratchSlot::InProjQkv => "scratch_pool/in_proj_qkv",
            ScratchSlot::InProjA => "scratch_pool/in_proj_a",
            ScratchSlot::InProjB => "scratch_pool/in_proj_b",
            ScratchSlot::InProjZ => "scratch_pool/in_proj_z",
            ScratchSlot::ConvQkv => "scratch_pool/conv_qkv",
            ScratchSlot::AttnOutputLinear => "scratch_pool/attn_output_linear",
            ScratchSlot::OutProj => "scratch_pool/out_proj",
            ScratchSlot::MlpGate => "scratch_pool/mlp_gate",
            ScratchSlot::MlpUp => "scratch_pool/mlp_up",
            ScratchSlot::MlpOutput => "scratch_pool/mlp_output",
        }
    }
}

/// Dimensions needed to size every scratch slot. Built once from the
/// model config in `Qwen35Session::new` and consumed by `ScratchPool::new`.
///
/// Every field is `u32` because it ultimately flows into `wgpu` buffer
/// sizes (which are `u64`) and tensor shape parameters (which are
/// `u32`); doing the cast at the boundary keeps the arithmetic below
/// from being littered with `as u32` / `as u64`.
#[derive(Debug, Clone, Copy)]
pub struct ScratchPoolConfig {
    /// Residual / norm-output / o_proj / down_proj feature dim.
    pub hidden_size: u32,
    /// Maximum number of tokens any single `forward()` can process.
    /// Every slot's buffer is sized to hold this many rows.
    pub max_seq_len: u32,

    // Self-attention dimensions.
    /// `num_attention_heads * 2 * head_dim`. Q and gate are fused into
    /// one tensor by the q_proj weight.
    pub q_gate_dim: u32,
    /// `num_attention_heads * head_dim`. Width of the attention block's
    /// per-token output before output gating.
    pub attn_output_self_dim: u32,

    // Linear-attention dimensions.
    /// `q_dim + k_dim + v_dim` of the fused linear-attention input
    /// projection.
    pub qkv_dim: u32,
    /// Width of `in_proj_a` / `in_proj_b` outputs (= number of value
    /// heads).
    pub linear_num_value_heads: u32,
    /// `linear_num_value_heads * linear_value_head_dim`. Width of the
    /// linear-attention block's value-space tensors.
    pub v_dim: u32,

    /// MLP intermediate feature dim.
    pub intermediate_size: u32,
}

impl ScratchPoolConfig {
    fn feature_dim_for(&self, slot: ScratchSlot) -> u32 {
        match slot {
            ScratchSlot::Normed => self.hidden_size,
            ScratchSlot::QGate => self.q_gate_dim,
            ScratchSlot::AttnOutputSelf => self.attn_output_self_dim,
            ScratchSlot::OProj => self.hidden_size,
            ScratchSlot::InProjQkv => self.qkv_dim,
            ScratchSlot::InProjA => self.linear_num_value_heads,
            ScratchSlot::InProjB => self.linear_num_value_heads,
            ScratchSlot::InProjZ => self.v_dim,
            ScratchSlot::ConvQkv => self.qkv_dim,
            ScratchSlot::AttnOutputLinear => self.v_dim,
            ScratchSlot::OutProj => self.hidden_size,
            ScratchSlot::MlpGate => self.intermediate_size,
            ScratchSlot::MlpUp => self.intermediate_size,
            ScratchSlot::MlpOutput => self.hidden_size,
        }
    }
}

/// One `wgpu::Buffer` per `ScratchSlot`, each sized to
/// `max_seq_len * feature_dim(slot) * 4` bytes. Shared across all
/// layers of a session.
pub struct ScratchPool {
    buffers: [wgpu::Buffer; NUM_SLOTS],
    feature_dims: [u32; NUM_SLOTS],
    max_seq_len: u32,
}

const ELEM_BYTES: u32 = std::mem::size_of::<f32>() as u32;

impl ScratchPool {
    pub fn new(device: &wgpu::Device, cfg: &ScratchPoolConfig) -> Self {
        debug_assert!(cfg.max_seq_len >= 1, "ScratchPool max_seq_len must be >= 1");
        let buffers: [wgpu::Buffer; NUM_SLOTS] = ALL_SLOTS.map(|slot| {
            let feature_dim = cfg.feature_dim_for(slot);
            debug_assert!(
                feature_dim >= 1,
                "ScratchPool slot {:?} has feature_dim 0",
                slot,
            );
            let size = (cfg.max_seq_len as u64) * (feature_dim as u64) * (ELEM_BYTES as u64);
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(slot.label()),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        });
        let feature_dims = ALL_SLOTS.map(|slot| cfg.feature_dim_for(slot));
        Self {
            buffers,
            feature_dims,
            max_seq_len: cfg.max_seq_len,
        }
    }

    pub fn max_seq_len(&self) -> u32 {
        self.max_seq_len
    }

    /// Feature dimension of `slot`, identical to the value supplied via
    /// `ScratchPoolConfig` at construction. Useful when a caller wants
    /// to build a non-2-D view (e.g. the fused 4-D q+gate view) and
    /// already knows the inner shape from the layer config.
    pub fn feature_dim(&self, slot: ScratchSlot) -> u32 {
        self.feature_dims[slot.index()]
    }

    /// Raw access to the slot's backing buffer. Use this when the
    /// downstream kernel expects a `&wgpu::Buffer` (e.g. `conv_silu`,
    /// `delta_rule`, `gated_norm`) or when the caller wants to build a
    /// non-2-D `BufferView` over the slot.
    ///
    /// The returned buffer is sized to `max_seq_len * feature_dim(slot)
    /// * 4` bytes; out-of-range views will trip wgpu validation, not
    /// this accessor.
    pub fn buffer(&self, slot: ScratchSlot) -> &wgpu::Buffer {
        &self.buffers[slot.index()]
    }

    /// Tight row-major 2-D view `[num_tokens, feature_dim(slot)]`. This
    /// is the common case: most kernels accept a 2-D `BufferView` and
    /// don't need any narrowing or striding.
    pub fn view_2d(&self, slot: ScratchSlot, num_tokens: u32) -> BufferView<'_> {
        debug_assert!(
            num_tokens >= 1 && num_tokens <= self.max_seq_len,
            "ScratchPool::view_2d({:?}, {}): num_tokens out of range [1, {}]",
            slot,
            num_tokens,
            self.max_seq_len,
        );
        BufferView::new_2d_tight(
            self.buffer(slot),
            num_tokens,
            self.feature_dim(slot),
            ELEM_BYTES,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_test_utils::gpu_or_skip;

    fn small_cfg() -> ScratchPoolConfig {
        ScratchPoolConfig {
            hidden_size: 8,
            max_seq_len: 4,
            q_gate_dim: 16,
            attn_output_self_dim: 8,
            qkv_dim: 12,
            linear_num_value_heads: 4,
            v_dim: 8,
            intermediate_size: 16,
        }
    }

    #[tokio::test]
    async fn buffer_sizes_match_max_seq_len_times_feature_dim() {
        let (device, _queue) = gpu_or_skip!();
        let cfg = small_cfg();
        let pool = ScratchPool::new(&device, &cfg);
        for slot in ALL_SLOTS {
            let expected =
                (cfg.max_seq_len as u64) * (cfg.feature_dim_for(slot) as u64) * (ELEM_BYTES as u64);
            assert_eq!(
                pool.buffer(slot).size(),
                expected,
                "slot {:?} buffer size mismatch",
                slot,
            );
            assert_eq!(pool.feature_dim(slot), cfg.feature_dim_for(slot));
        }
        assert_eq!(pool.max_seq_len(), cfg.max_seq_len);
    }

    #[tokio::test]
    async fn view_2d_has_expected_shape_and_byte_size() {
        let (device, _queue) = gpu_or_skip!();
        let pool = ScratchPool::new(&device, &small_cfg());
        let view = pool.view_2d(ScratchSlot::QGate, 3);
        assert_eq!(view.rank, 2);
        assert_eq!(view.shape[0], 3);
        assert_eq!(view.shape[1], 16);
        assert_eq!(view.elem_size, ELEM_BYTES);
        assert_eq!(view.byte_offset, 0);
        assert_eq!(view.total_byte_size(), 3 * 16 * 4);
    }

    #[tokio::test]
    #[should_panic(expected = "num_tokens out of range")]
    async fn view_2d_rejects_num_tokens_zero() {
        let (device, _queue) = gpu_or_skip!();
        let pool = ScratchPool::new(&device, &small_cfg());
        let _ = pool.view_2d(ScratchSlot::Normed, 0);
    }

    #[tokio::test]
    #[should_panic(expected = "num_tokens out of range")]
    async fn view_2d_rejects_num_tokens_above_max() {
        let (device, _queue) = gpu_or_skip!();
        let pool = ScratchPool::new(&device, &small_cfg());
        let _ = pool.view_2d(ScratchSlot::Normed, 5);
    }
}
