use safetensors::SafeTensors;

use crate::{
    buffer_view::BufferView,
    layers::{
        linear_attention::{
            LinearAttentionConfig, LinearAttentionLayer, LinearAttentionLayerRunner,
            LinearAttentionLayerSession,
        },
        self_attention::{
            SelfAttentionConfig, SelfAttentionLayer, SelfAttentionLayerRunner,
            SelfAttentionLayerSession,
        },
    },
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

impl<'m> AttentionLayerSession<'m> {
    pub fn plan(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        residual_slot: BufferView<'_>,
        position_buffer: &wgpu::Buffer,
    ) -> AttentionLayerRunner {
        match self {
            Self::Linear(session) => {
                AttentionLayerRunner::Linear(session.plan(device, queue, residual_slot))
            }
            Self::Full(session) => AttentionLayerRunner::Full(session.plan(
                device,
                queue,
                residual_slot,
                position_buffer,
            )),
        }
    }
}

/// Cached runners for one forward pass on a single transformer layer.
pub enum AttentionLayerRunner {
    Linear(LinearAttentionLayerRunner),
    Full(SelfAttentionLayerRunner),
}

impl AttentionLayerRunner {
    pub fn forward(&self, cpass: &mut wgpu::ComputePass<'_>) {
        match self {
            Self::Linear(d) => d.forward(cpass),
            Self::Full(d) => d.forward(cpass),
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

    /// Build a [`LayerStackRunner`] holding one runner per layer.
    pub fn plan(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        residual_slot: BufferView<'_>,
        position_buffer: &wgpu::Buffer,
    ) -> LayerStackRunner {
        let runners = self
            .sessions
            .iter()
            .map(|session| session.plan(device, queue, residual_slot, position_buffer))
            .collect();
        LayerStackRunner { runners }
    }
}

pub struct LayerStackRunner {
    runners: Vec<AttentionLayerRunner>,
}

impl LayerStackRunner {
    pub fn forward(&self, cpass: &mut wgpu::ComputePass<'_>) {
        for runner in &self.runners {
            runner.forward(cpass);
        }
    }
}
