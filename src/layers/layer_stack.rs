use safetensors::SafeTensors;

use crate::{
    buffer_view::BufferView,
    layers::{
        layer_session::LayerSession,
        linear_attention::{
            LinearAttentionConfig, LinearAttentionLayer, LinearAttentionLayerSession,
        },
        self_attention::{SelfAttentionConfig, SelfAttentionLayer, SelfAttentionLayerSession},
    },
    scratch_pool::ScratchPool,
};

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
        scratch: &ScratchPool,
        residual_slot: BufferView<'_>,
        prev_position: usize,
    ) {
        match self {
            Self::Linear(session) => {
                session.forward(device, queue, scratch, residual_slot, prev_position)
            }
            Self::Full(session) => {
                session.forward(device, queue, scratch, residual_slot, prev_position)
            }
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
        scratch: &ScratchPool,
        residual_slot: BufferView<'_>,
        prev_position: usize,
    ) {
        for session in &mut self.sessions {
            session.forward(device, queue, scratch, residual_slot, prev_position);
        }
    }
}
