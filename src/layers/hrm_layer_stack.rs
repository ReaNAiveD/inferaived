use safetensors::SafeTensors;

use crate::{
    buffer_view::BufferView,
    layers::hrm_self_attention::{
        HrmSelfAttentionConfig, HrmSelfAttentionLayer, HrmSelfAttentionLayerRunner,
        HrmSelfAttentionLayerSession,
    },
};

/// Configuration for one homogeneous HRM-Text recurrent module (an `H` or `L`
/// stack): `num_hidden_layers` identical transformer blocks.
#[derive(Debug, Clone, Copy)]
pub struct HrmLayerStackConfig {
    pub num_hidden_layers: usize,
    pub layer: HrmSelfAttentionConfig,
}

/// One HRM-Text recurrent module: `num_hidden_layers` identical
/// [`HrmSelfAttentionLayer`] blocks.
pub struct HrmLayerStack {
    layers: Vec<HrmSelfAttentionLayer>,
}

impl HrmLayerStack {
    pub fn new<'data>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tensor: &SafeTensors<'data>,
        weight_prefix: &str,
        config: &HrmLayerStackConfig,
        hidden_size: usize,
    ) -> Self {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer_weight_prefix = format!("{}.layers.{}", weight_prefix, i);
            layers.push(HrmSelfAttentionLayer::new(
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

    pub fn layers(&self) -> &[HrmSelfAttentionLayer] {
        &self.layers
    }
}

/// Stack-wide per-sequence state: one [`HrmSelfAttentionLayerSession`] per
/// layer, in the same order.
pub struct HrmLayerStackSession<'m> {
    sessions: Vec<HrmSelfAttentionLayerSession<'m>>,
}

impl<'m> HrmLayerStackSession<'m> {
    pub fn new(stack: &'m HrmLayerStack, device: &wgpu::Device, max_seq_len: usize) -> Self {
        let sessions = stack
            .layers()
            .iter()
            .map(|layer| HrmSelfAttentionLayerSession::new(layer, device, max_seq_len))
            .collect();
        Self { sessions }
    }

    /// Build an [`HrmLayerStackRunner`] holding one runner per layer.
    pub fn plan(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        residual_slot: BufferView<'_>,
        position_buffer: &wgpu::Buffer,
        prefix_buffer: &wgpu::Buffer,
    ) -> HrmLayerStackRunner {
        let runners = self
            .sessions
            .iter()
            .map(|session| {
                session.plan(device, queue, residual_slot, position_buffer, prefix_buffer)
            })
            .collect();
        HrmLayerStackRunner { runners }
    }

    /// Erase per-sequence cached state across every layer. The HRM block holds
    /// no resettable scratch beyond its KV cache (overwritten on the next
    /// forward), so this is a no-op.
    pub fn reset(&self, _encoder: &mut wgpu::CommandEncoder) {}
}

pub struct HrmLayerStackRunner {
    runners: Vec<HrmSelfAttentionLayerRunner>,
}

impl HrmLayerStackRunner {
    pub fn forward(&self, cpass: &mut wgpu::ComputePass<'_>) {
        for runner in &self.runners {
            runner.forward(cpass);
        }
    }
}
