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
