use crate::{
    buffer_view::BufferView,
    layers::minicpm5_layer_stack::MiniCPM5LayerStack,
    layers::minicpm5_self_attention::{
        MaskedKvPool, MiniCPM5MaskedSelfAttentionLayerRunner, MiniCPM5SelfAttentionLayer,
    },
};

/// One transformer layer bound to its own shared K/V cache.
pub struct MiniCPM5MaskedSelfAttentionLayerSession<'m> {
    layer: &'m MiniCPM5SelfAttentionLayer,
    /// Shared K cache, `[kv_capacity, num_kv_heads, head_dim]` f32, row-major.
    k_pool: wgpu::Buffer,
    /// Shared V cache, same shape/layout as `k_pool`.
    v_pool: wgpu::Buffer,
    /// Slot capacity of each cache (the visibility-mask length for one query).
    kv_capacity: usize,
}

impl<'m> MiniCPM5MaskedSelfAttentionLayerSession<'m> {
    pub fn new(
        device: &wgpu::Device,
        kv_capacity: usize,
        layer: &'m MiniCPM5SelfAttentionLayer,
    ) -> Self {
        let bytes = (kv_capacity
            * layer.num_key_value_heads()
            * layer.head_dim()
            * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
        let k_pool = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("minicpm5_masked_layer_session/k_pool"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let v_pool = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("minicpm5_masked_layer_session/v_pool"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self {
            layer,
            k_pool,
            v_pool,
            kv_capacity,
        }
    }

    /// Plan this layer's masked forward over its own shared K/V cache.
    pub fn plan(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        residual_slot: BufferView<'_>,
        rope_position: &wgpu::Buffer,
        scatter_position: &wgpu::Buffer,
        visibility: &wgpu::Buffer,
    ) -> MiniCPM5MaskedSelfAttentionLayerRunner {
        let k_view = BufferView::new_3d_tight(
            &self.k_pool,
            self.kv_capacity as u32,
            self.layer.num_key_value_heads() as u32,
            self.layer.head_dim() as u32,
            std::mem::size_of::<f32>() as u32,
        );
        let v_view = BufferView::new_3d_tight(
            &self.v_pool,
            self.kv_capacity as u32,
            self.layer.num_key_value_heads() as u32,
            self.layer.head_dim() as u32,
            std::mem::size_of::<f32>() as u32,
        );
        self.layer.plan_masked(
            device,
            queue,
            residual_slot,
            MaskedKvPool {
                k: k_view,
                v: v_view,
                scatter_position,
                visibility,
            },
            rope_position,
        )
    }
}

/// One masked self-attention session per layer.
pub struct MiniCPM5MaskedLayerStackSession<'m> {
    sessions: Vec<MiniCPM5MaskedSelfAttentionLayerSession<'m>>,
}

impl<'m> MiniCPM5MaskedLayerStackSession<'m> {
    /// Allocate one `kv_capacity`-slot masked layer session per layer of `stack`.
    pub fn new(stack: &'m MiniCPM5LayerStack, device: &wgpu::Device, kv_capacity: usize) -> Self {
        let sessions = stack
            .layers()
            .iter()
            .map(|layer| MiniCPM5MaskedSelfAttentionLayerSession::new(device, kv_capacity, layer))
            .collect();
        Self { sessions }
    }

    /// Build a [`MiniCPM5MaskedLayerStackRunner`] holding one runner per layer,
    /// each planned over the shared `residual_slot` and masked-attention scalars.
    pub fn plan(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        residual_slot: BufferView<'_>,
        rope_position: &wgpu::Buffer,
        scatter_position: &wgpu::Buffer,
        visibility: &wgpu::Buffer,
    ) -> MiniCPM5MaskedLayerStackRunner {
        let runners = self
            .sessions
            .iter()
            .map(|session| {
                session.plan(
                    device,
                    queue,
                    residual_slot,
                    rope_position,
                    scatter_position,
                    visibility,
                )
            })
            .collect();
        MiniCPM5MaskedLayerStackRunner { runners }
    }
}

/// Planned masked layer-stack forward: one masked self-attention runner per
/// layer, in stack order.
pub struct MiniCPM5MaskedLayerStackRunner {
    runners: Vec<MiniCPM5MaskedSelfAttentionLayerRunner>,
}

impl MiniCPM5MaskedLayerStackRunner {
    pub fn forward(&self, cpass: &mut wgpu::ComputePass<'_>) {
        for runner in &self.runners {
            runner.forward(cpass);
        }
    }
}
