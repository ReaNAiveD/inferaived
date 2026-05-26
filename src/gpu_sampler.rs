use wgpu::{Device, Queue};

use crate::buffer_view::BufferView;
use crate::kernels::argmax::{GpuArgmax, GpuArgmaxRunner};

pub struct GpuSampler {
    argmax: GpuArgmax,
}

impl GpuSampler {
    pub fn new(device: &Device) -> Self {
        Self {
            argmax: GpuArgmax::new(device),
        }
    }

    /// Bake bindings for one sampler dispatch.
    pub fn plan(
        &self,
        device: &Device,
        queue: &Queue,
        logits: BufferView<'_>,
        token_out: BufferView<'_>,
    ) -> GpuSamplerRunner {
        GpuSamplerRunner {
            argmax: self.argmax.plan(device, queue, logits, token_out),
        }
    }
}

pub struct GpuSamplerRunner {
    argmax: GpuArgmaxRunner,
}

impl GpuSamplerRunner {
    pub fn forward(&self, cpass: &mut wgpu::ComputePass<'_>) {
        self.argmax.forward(cpass);
    }
}
