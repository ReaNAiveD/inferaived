use safetensors::SafeTensors;

use crate::{
    buffer_view::BufferView,
    layers::minicpm5_self_attention::{
        MiniCPM5SelfAttentionConfig, MiniCPM5SelfAttentionLayer, MiniCPM5SelfAttentionLayerRunner,
        MiniCPM5SelfAttentionLayerSession,
    },
};

/// Configuration for a homogeneous MiniCPM5 layer stack: every layer is a
/// full self-attention block.
#[derive(Debug, Clone, Copy)]
pub struct MiniCPM5LayerStackConfig {
    pub num_hidden_layers: usize,
    pub layer: MiniCPM5SelfAttentionConfig,
}

/// MiniCPM5's mid-stack: `num_hidden_layers` identical full self-attention
/// blocks.
pub struct MiniCPM5LayerStack {
    layers: Vec<MiniCPM5SelfAttentionLayer>,
}

impl MiniCPM5LayerStack {
    pub fn new<'data>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tensor: &SafeTensors<'data>,
        weight_prefix: &str,
        config: &MiniCPM5LayerStackConfig,
        hidden_size: usize,
    ) -> Self {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer_weight_prefix = format!("{}.layers.{}", weight_prefix, i);
            layers.push(MiniCPM5SelfAttentionLayer::new(
                device,
                queue,
                tensor,
                &layer_weight_prefix,
                hidden_size,
                &config.layer,
            ));
        }
        Self { layers }
    }

    pub fn layers(&self) -> &[MiniCPM5SelfAttentionLayer] {
        &self.layers
    }
}

/// Stack-wide per-sequence state: one [`MiniCPM5SelfAttentionLayerSession`]
/// per layer, in the same order.
pub struct MiniCPM5LayerStackSession<'m> {
    sessions: Vec<MiniCPM5SelfAttentionLayerSession<'m>>,
}

impl<'m> MiniCPM5LayerStackSession<'m> {
    pub fn new(stack: &'m MiniCPM5LayerStack, device: &wgpu::Device, max_seq_len: usize) -> Self {
        let sessions = stack
            .layers()
            .iter()
            .map(|layer| MiniCPM5SelfAttentionLayerSession::new(layer, device, max_seq_len))
            .collect();
        Self { sessions }
    }

    /// Build a [`MiniCPM5LayerStackRunner`] holding one runner per layer.
    pub fn plan(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        residual_slot: BufferView<'_>,
        position_buffer: &wgpu::Buffer,
    ) -> MiniCPM5LayerStackRunner {
        let runners = self
            .sessions
            .iter()
            .map(|session| session.plan(device, queue, residual_slot, position_buffer))
            .collect();
        MiniCPM5LayerStackRunner { runners }
    }

    /// Erase per-sequence cached state across every layer.
    pub fn reset(&self, _encoder: &mut wgpu::CommandEncoder) {}
}

pub struct MiniCPM5LayerStackRunner {
    runners: Vec<MiniCPM5SelfAttentionLayerRunner>,
}

impl MiniCPM5LayerStackRunner {
    pub fn forward(&self, cpass: &mut wgpu::ComputePass<'_>) {
        for runner in &self.runners {
            runner.forward(cpass);
        }
    }
}
