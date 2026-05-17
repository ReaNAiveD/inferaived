use crate::buffer_view::BufferView;

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
