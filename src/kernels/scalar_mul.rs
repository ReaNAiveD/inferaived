use crate::buffer_view::BufferView;

#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ScalarMulParams {
    row_stride: u32,
    hidden_size: u32,
    seq_len: u32,
    scalar: f32,
}

/// In-place scalar multiply: `hidden[t, i] *= scalar` over a `[seq, hidden]`
/// f32 view. Used by HRM-Text to apply `embedding_scale` to the token
/// embeddings before the recurrent core.
pub struct ScalarMulInplaceWebgpu {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    hidden_size: usize,
    scalar: f32,
}

impl ScalarMulInplaceWebgpu {
    pub fn new(device: &wgpu::Device, hidden_size: usize, scalar: f32) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scalar_mul/shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("wgsl-shaders/scalar_mul.wgsl").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scalar_mul/bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scalar_mul/pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("scalar_mul/pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("workgroup_size", 256.0f64)],
                ..Default::default()
            },
            cache: None,
        });
        Self {
            bind_group_layout,
            pipeline,
            hidden_size,
            scalar,
        }
    }

    /// Bake the per-buffer bindings into a [`ScalarMulInplaceWebgpuRunner`]
    /// for repeated dispatch into a caller-owned compute pass.
    pub fn plan(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        hidden: BufferView<'_>,
    ) -> ScalarMulInplaceWebgpuRunner {
        debug_assert_eq!(hidden.rank, 2, "scalar_mul: hidden must be rank-2");
        debug_assert_eq!(
            hidden.shape[1] as usize, self.hidden_size,
            "scalar_mul: hidden inner dim {} != hidden_size {}",
            hidden.shape[1], self.hidden_size,
        );
        let seq_len = hidden.shape[0];
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scalar_mul_runner/uniform_buffer"),
            size: std::mem::size_of::<ScalarMulParams>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform = ScalarMulParams {
            row_stride: hidden.stride[0],
            hidden_size: self.hidden_size as u32,
            seq_len,
            scalar: self.scalar,
        };
        queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&uniform));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scalar_mul_runner/bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: hidden.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });
        ScalarMulInplaceWebgpuRunner {
            pipeline: self.pipeline.clone(),
            bind_group,
            seq_len,
        }
    }
}

pub struct ScalarMulInplaceWebgpuRunner {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    seq_len: u32,
}

impl ScalarMulInplaceWebgpuRunner {
    pub fn forward(&self, cpass: &mut wgpu::ComputePass<'_>) {
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.dispatch_workgroups(self.seq_len, 1, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_test_utils::*;

    /// Multi-token prefill shape: a tight `[seq, hidden]` view scaled in place.
    #[tokio::test]
    async fn test_scalar_mul_prefill() {
        let (device, queue) = gpu_or_skip!();
        let seq_len = 3;
        let hidden_size = 16;
        let scalar = 39.191_837_f32;
        let data: Vec<f32> = (0..seq_len * hidden_size)
            .map(|i| (i as f32) * 0.1 - 1.0)
            .collect();
        let expected: Vec<f32> = data.iter().map(|x| x * scalar).collect();

        let gpu = ScalarMulInplaceWebgpu::new(&device, hidden_size, scalar);
        let buf = upload_f32(&device, &data);
        let elem_size = std::mem::size_of::<f32>() as u32;
        let view = BufferView::new_2d_tight(&buf, seq_len as u32, hidden_size as u32, elem_size);
        let runner = gpu.plan(&device, &queue, view);
        run_blocking_compute(&device, &queue, |cp| runner.forward(cp));
        let actual = download_f32(&device, &queue, &buf, seq_len * hidden_size);

        assert_approx_eq(&actual, &expected, 1e-3);
    }

    /// Single-token decode shape: a `[1, hidden]` view must scale identically.
    #[tokio::test]
    async fn test_scalar_mul_decode_single_row() {
        let (device, queue) = gpu_or_skip!();
        let hidden_size = 16;
        let scalar = -2.5f32;
        let data: Vec<f32> = (0..hidden_size).map(|i| (i as f32) * 0.25).collect();
        let expected: Vec<f32> = data.iter().map(|x| x * scalar).collect();

        let gpu = ScalarMulInplaceWebgpu::new(&device, hidden_size, scalar);
        let buf = upload_f32(&device, &data);
        let elem_size = std::mem::size_of::<f32>() as u32;
        let view = BufferView::new_2d_tight(&buf, 1, hidden_size as u32, elem_size);
        let runner = gpu.plan(&device, &queue, view);
        run_blocking_compute(&device, &queue, |cp| runner.forward(cp));
        let actual = download_f32(&device, &queue, &buf, hidden_size);

        assert_approx_eq(&actual, &expected, 1e-5);
    }
}
