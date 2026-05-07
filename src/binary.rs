#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ElementwiseAddUniform {
    stride_src_token: u32,
    stride_other_token: u32,
    src_offset: u32,
    other_offset: u32,

    hidden_size: u32,
    seq_len: u32,
}

pub struct ElementwiseAddWebgpu {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    uniform_buffer: wgpu::Buffer,

    hidden_size: usize,
}

impl ElementwiseAddWebgpu {
    pub fn new(device: &wgpu::Device, hidden_size: usize) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("elementwise_add/shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("wgsl-shaders/elementwise_add.wgsl").into()),
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
            bind_group_layouts: &[&bind_group_layout],
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
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("elementwise_add/uniform_buffer"),
            size: std::mem::size_of::<ElementwiseAddUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            bind_group_layout,
            pipeline,
            uniform_buffer,
            hidden_size,
        }
    }

    pub fn compute(&self, device: &wgpu::Device, queue: &wgpu::Queue, src_buffer: &wgpu::Buffer, other_buffer: &wgpu::Buffer, seq_len: usize) {
        let uniform = ElementwiseAddUniform {
            stride_src_token: self.hidden_size as u32,
            stride_other_token: self.hidden_size as u32,
            src_offset: 0,
            other_offset: 0,
            hidden_size: self.hidden_size as u32,
            seq_len: seq_len as u32,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("elementwise_add/bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: other_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("elementwise_add/command_encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("elementwise_add/compute_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(seq_len as u32, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }
}