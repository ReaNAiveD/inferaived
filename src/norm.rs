use safetensors::tensor::TensorView;
use wgpu::{BindGroupLayout, Buffer, ComputePipeline};

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RmsNormUniform {
    offset_src: u32,
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,
    ne0: u32,
    ne1: u32,
    ne2: u32,
    ne3: u32,
    epsilon: f32,
}

pub struct RmsNormWebgpu {
    bind_group_layout: BindGroupLayout,
    pipeline: ComputePipeline,
    uniform_buffer: Buffer,
    weight_buffer: Buffer,
    hidden_size: usize,
}

impl RmsNormWebgpu {
    pub fn new<'data>(device: &wgpu::Device, queue: &wgpu::Queue, weight: TensorView<'data>, hidden_size: usize) -> Self {
        let weight_f32: Vec<f32> = weight.data().chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                half::bf16::from_bits(bits).to_f32()
            })
            .collect();
        assert_eq!(weight_f32.len(), hidden_size);
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rms_norm/shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "wgsl-shaders/rms_norm.wgsl"
            ))),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rms_norm/bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
            label: Some("rms_norm/pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rms_norm/pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions { constants: &[("workgroup_size", 256f64)], zero_initialize_workgroup_memory: true },
            cache: None,
        });
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rms_norm/uniform_buffer"),
            size: std::mem::size_of::<RmsNormUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let weight_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rms_norm/weight_buffer"),
            size: (weight_f32.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&weight_buffer, 0, bytemuck::cast_slice(&weight_f32));
        Self {
            bind_group_layout,
            pipeline,
            uniform_buffer,
            weight_buffer,
            hidden_size,
        }
    }

    pub fn compute(&self, device: &wgpu::Device, queue: &wgpu::Queue, input_buffer: &Buffer, dst_buffer: &Buffer, n_rows: usize) {
        let uniform = RmsNormUniform {
            offset_src: 0,
            stride_src1: self.hidden_size as u32,
            stride_src2: 0,
            stride_src3: 0,
            ne0: self.hidden_size as u32,
            ne1: n_rows as u32,
            ne2: 1,
            ne3: 1,
            epsilon: 1e-6,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rms_norm/bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dst_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.weight_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rms_norm/command_encoder"),
        });
        let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rms_norm/compute_pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(n_rows as u32, 1, 1);
        drop(compute_pass);
        queue.submit(Some(command_encoder.finish()));
    }
}