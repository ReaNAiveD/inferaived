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
    hidden_size: usize,
}

impl RmsNormWebgpu {
    pub fn new(device: &wgpu::Device, hidden_size: usize) -> Self {
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
        Self {
            bind_group_layout,
            pipeline,
            uniform_buffer,
            hidden_size,
        }
    }

    pub fn compute(&self, device: &wgpu::Device, queue: &wgpu::Queue, input_buffer: &Buffer, n_rows: usize) {
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
        // The n_rows is the number of the tokens. For each token, we dispatch 256 threads to compute the RMS norm in parallel.
        compute_pass.dispatch_workgroups(n_rows as u32, 1, 1);
        drop(compute_pass);
        queue.submit(Some(command_encoder.finish()));
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NormScaleUniform {
    offset_src: u32,

    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    ne0: u32,
    ne1: u32,
    ne2: u32,
    ne3: u32,

    ne0_scale: u32,
    offset_scale: u32,
}

pub struct NormScaleWebgpu {
    bind_group_layout: BindGroupLayout,
    pipeline: ComputePipeline,
    scale_buffer: Buffer,
    uniform_buffer: Buffer,
    hidden_size: usize,
}

impl NormScaleWebgpu {
    pub fn new<'data>(device: &wgpu::Device, queue: &wgpu::Queue, norm_weight: TensorView<'data>, hidden_size: usize) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("norm_scale/shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "wgsl-shaders/norm_scale.wgsl"
            ))),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("norm_scale/bind_group_layout"),
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
            label: Some("norm_scale/pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("norm_scale/pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let weight_f32 = norm_weight.data().chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                half::bf16::from_bits(bits).to_f32()
            })
            .collect::<Vec<f32>>();
        let scale_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("norm_scale/scale_buffer"),
            size: (weight_f32.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&scale_buffer, 0, bytemuck::cast_slice(&weight_f32));
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("norm_scale/uniform_buffer"),
            size: std::mem::size_of::<NormScaleUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            bind_group_layout,
            pipeline,
            scale_buffer,
            uniform_buffer,
            hidden_size,
        }
    }

    pub fn compute(&self, device: &wgpu::Device, queue: &wgpu::Queue, input_buffer: &Buffer, n_rows: usize) {
        let uniform = NormScaleUniform {
            offset_src: 0,
            stride_src1: self.hidden_size as u32,
            stride_src2: 0,
            stride_src3: 0,
            ne0: self.hidden_size as u32,
            ne1: n_rows as u32,
            ne2: 1,
            ne3: 1,
            ne0_scale: self.hidden_size as u32,
            offset_scale: 0,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("norm_scale/bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.scale_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("norm_scale/command_encoder"),
        });
        let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("norm_scale/compute_pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let total_elements = (self.hidden_size * n_rows) as u32;
        compute_pass.dispatch_workgroups((total_elements + 255) / 256, 1, 1);
        drop(compute_pass);
        queue.submit(Some(command_encoder.finish()));
    }
}