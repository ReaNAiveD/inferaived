use crate::buffer_view::BufferView;

#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ElementwiseAddParams {
    hidden_token_stride: u32,
    addend_token_stride: u32,

    hidden_size: u32,
    seq_len: u32,
}

pub struct ElementwiseAddInplaceWebgpu {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,

    vec_dim: usize,
}

impl ElementwiseAddInplaceWebgpu {
    pub fn new(device: &wgpu::Device, vec_dim: usize) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("elementwise_add/shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("wgsl-shaders/elementwise_add.wgsl").into(),
            ),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("elementwise_add/bind_group_layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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
            label: Some("elementwise_add/pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("elementwise_add/pipeline"),
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
            vec_dim,
        }
    }

    pub fn plan(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        hidden: BufferView<'_>,
        addend: BufferView<'_>,
    ) -> ElementwiseAddInplaceWebgpuRunner {
        debug_assert_eq!(hidden.shape[0], addend.shape[0]);
        debug_assert_eq!(hidden.shape[1], self.vec_dim as u32);
        debug_assert_eq!(addend.shape[1], self.vec_dim as u32);

        let seq_len = hidden.shape[0];

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("elementwise_add_runner/uniform_buffer"),
            size: std::mem::size_of::<ElementwiseAddParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform = ElementwiseAddParams {
            hidden_token_stride: hidden.stride[0],
            addend_token_stride: addend.stride[0],
            hidden_size: self.vec_dim as u32,
            seq_len,
        };
        queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&uniform));

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("elementwise_add/bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: hidden.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: addend.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });
        ElementwiseAddInplaceWebgpuRunner::new(self.pipeline.clone(), bind_group, seq_len)
    }
}

pub struct ElementwiseAddInplaceWebgpuRunner {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    seq_len: u32,
}

impl ElementwiseAddInplaceWebgpuRunner {
    pub fn new(pipeline: wgpu::ComputePipeline, bind_group: wgpu::BindGroup, seq_len: u32) -> Self {
        Self {
            pipeline,
            bind_group,
            seq_len,
        }
    }

    pub fn forward(&self, cpass: &mut wgpu::ComputePass<'_>) {
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.dispatch_workgroups(self.seq_len, 1, 1);
    }
}

#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ElementwiseAddOutParams {
    dst_token_stride: u32,
    a_token_stride: u32,
    b_token_stride: u32,
    hidden_size: u32,
    seq_len: u32,
}

/// Out-of-place elementwise add: `dst[t, i] = a[t, i] + b[t, i]`, honoring each
/// operand's per-token (outer) stride independently. A zero stride on an
/// operand broadcasts its single row across all tokens — HRM-Text uses this to
/// inject the `[hidden]` `z_L_init` vector across the whole sequence.
pub struct ElementwiseAddWebgpu {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    hidden_size: usize,
}

impl ElementwiseAddWebgpu {
    pub fn new(device: &wgpu::Device, hidden_size: usize) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("elementwise_add_out/shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("wgsl-shaders/elementwise_add_out.wgsl").into(),
            ),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("elementwise_add_out/bind_group_layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
            label: Some("elementwise_add_out/pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("elementwise_add_out/pipeline"),
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
        }
    }

    /// Bake the per-buffer bindings into an [`ElementwiseAddWebgpuRunner`].
    /// `a` and `b` may have a zero outer stride to broadcast a single row.
    pub fn plan(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dst: BufferView<'_>,
        a: BufferView<'_>,
        b: BufferView<'_>,
    ) -> ElementwiseAddWebgpuRunner {
        debug_assert_eq!(dst.shape[1] as usize, self.hidden_size);
        debug_assert_eq!(a.shape[1] as usize, self.hidden_size);
        debug_assert_eq!(b.shape[1] as usize, self.hidden_size);
        debug_assert_eq!(dst.shape[0], a.shape[0]);
        debug_assert_eq!(dst.shape[0], b.shape[0]);
        let seq_len = dst.shape[0];
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("elementwise_add_out_runner/uniform_buffer"),
            size: std::mem::size_of::<ElementwiseAddOutParams>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform = ElementwiseAddOutParams {
            dst_token_stride: dst.stride[0],
            a_token_stride: a.stride[0],
            b_token_stride: b.stride[0],
            hidden_size: self.hidden_size as u32,
            seq_len,
        };
        queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&uniform));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("elementwise_add_out_runner/bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: dst.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });
        ElementwiseAddWebgpuRunner {
            pipeline: self.pipeline.clone(),
            bind_group,
            seq_len,
        }
    }
}

pub struct ElementwiseAddWebgpuRunner {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    seq_len: u32,
}

impl ElementwiseAddWebgpuRunner {
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

    /// CPU reference: hidden[t,i] += addend[t,i]
    fn cpu_elementwise_add(hidden: &mut [f32], addend: &[f32], hidden_size: usize, seq_len: usize) {
        for t in 0..seq_len {
            for i in 0..hidden_size {
                hidden[t * hidden_size + i] += addend[t * hidden_size + i];
            }
        }
    }

    #[tokio::test]
    async fn test_elementwise_add() {
        let (device, queue) = gpu_or_skip!();
        let seq_len = 3;
        let hidden_size = 16;
        let hidden: Vec<f32> = (0..seq_len * hidden_size)
            .map(|i| (i as f32) * 0.1)
            .collect();
        let addend: Vec<f32> = (0..seq_len * hidden_size)
            .map(|i| (i as f32) * -0.05 + 1.0)
            .collect();

        let mut expected = hidden.clone();
        cpu_elementwise_add(&mut expected, &addend, hidden_size, seq_len);

        let gpu = ElementwiseAddInplaceWebgpu::new(&device, hidden_size);
        let h_buf = upload_f32(&device, &hidden);
        let a_buf = upload_f32(&device, &addend);
        let elem_size = std::mem::size_of::<f32>() as u32;
        let h_view =
            BufferView::new_2d_tight(&h_buf, seq_len as u32, hidden_size as u32, elem_size);
        let a_view =
            BufferView::new_2d_tight(&a_buf, seq_len as u32, hidden_size as u32, elem_size);
        let runner = gpu.plan(&device, &queue, h_view, a_view);
        run_blocking_compute(&device, &queue, |cp| runner.forward(cp));
        let actual = download_f32(&device, &queue, &h_buf, seq_len * hidden_size);

        assert_approx_eq(&actual, &expected, 1e-5);
    }
}
