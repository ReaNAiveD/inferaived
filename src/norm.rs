use safetensors::tensor::TensorView;
use wgpu::{BindGroupLayout, Buffer, ComputePipeline};

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RmsNormParams {
    input_offset: u32,
    input_row_stride: u32,
    hidden_size: u32,
    seq_len: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RmsNormInplaceParams {
    hidden_offset: u32,
    hidden_row_stride: u32,
    hidden_size: u32,
    seq_len: u32,
    eps: f32,
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
        debug_assert_eq!(weight_f32.len(), hidden_size);
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
            size: std::mem::size_of::<RmsNormParams>() as u64,
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
        let uniform = RmsNormParams {
            input_offset: 0,
            input_row_stride: self.hidden_size as u32,
            hidden_size: self.hidden_size as u32,
            seq_len: n_rows as u32,
            eps: 1e-6,
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

pub struct RmsNormInplaceWebgpu {
    bind_group_layout: BindGroupLayout,
    pipeline: ComputePipeline,
    uniform_buffer: Buffer,
    weight_buffer: Buffer,
    hidden_size: usize,
}

impl RmsNormInplaceWebgpu {
    pub fn new<'data>(device: &wgpu::Device, queue: &wgpu::Queue, weight: TensorView<'data>, hidden_size: usize) -> Self {
        let weight_f32: Vec<f32> = weight.data().chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                half::bf16::from_bits(bits).to_f32()
            })
            .collect();
        debug_assert_eq!(weight_f32.len(), hidden_size);
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rms_norm_inplace/shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "wgsl-shaders/rms_norm_inplace.wgsl"
            ))),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rms_norm_inplace/bind_group_layout"),
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
            label: Some("rms_norm_inplace/pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rms_norm_inplace/pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions { constants: &[("workgroup_size", 256f64)], zero_initialize_workgroup_memory: true },
            cache: None,
        });
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rms_norm_inplace/uniform_buffer"),
            size: std::mem::size_of::<RmsNormInplaceParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let weight_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rms_norm_inplace/weight_buffer"),
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

    pub fn compute(&self, device: &wgpu::Device, queue: &wgpu::Queue, src_buffer: &Buffer, n_rows: usize) {
        self.compute_strided(device, queue, src_buffer, 0, n_rows, self.hidden_size);
    }

    /// Normalize `n_rows` rows that are spaced `row_stride` elements apart,
    /// starting at `offset` elements into `src_buffer`.
    pub fn compute_strided(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src_buffer: &Buffer,
        offset: usize,
        n_rows: usize,
        row_stride: usize,
    ) {
        let uniform = RmsNormInplaceParams {
            hidden_offset: offset as u32,
            hidden_row_stride: row_stride as u32,
            hidden_size: self.hidden_size as u32,
            seq_len: n_rows as u32,
            eps: 1e-6,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rms_norm_inplace/bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.weight_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rms_norm_inplace/command_encoder"),
        });
        let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rms_norm_inplace/compute_pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(n_rows as u32, 1, 1);
        drop(compute_pass);
        queue.submit(Some(command_encoder.finish()));
    }
}