use safetensors::tensor::TensorView;
use wgpu::util::DeviceExt;

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct GatedRmsNormParams {
    num_heads: u32,
    head_dim: u32,
    seq_len: u32,

    hidden_offset: u32,
    hidden_token_stride: u32,
    hidden_head_stride: u32,

    gate_offset: u32,
    gate_token_stride: u32,
    gate_head_stride: u32,

    weight_offset: u32,

    eps: f32,
}

pub struct GatedRmsNormInplaceWebgpu {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    weight_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,

    num_value_heads: usize,
    value_head_dim: usize,
    epsilon: f32,
}

impl GatedRmsNormInplaceWebgpu {
    pub fn new<'data>(
        device: &wgpu::Device,
        weight_tensor: TensorView<'data>,
        num_value_heads: usize,
        value_head_dim: usize,
        epsilon: f32,
    ) -> Self {
        let weight: Vec<f32> = weight_tensor
            .data()
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        assert_eq!(weight.len(), value_head_dim);
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gated_rms_norm/shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "wgsl-shaders/gated_rms_norm.wgsl"
            ))),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gated_rms_norm/bind_group_layout"),
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
            label: Some("gated_rms_norm/pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gated_rms_norm/pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("workgroup_size", value_head_dim as f64)],
                ..Default::default()
            },
            cache: None,
        });
        let weight_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gated_rms_norm/weight_buffer"),
            contents: bytemuck::cast_slice(&weight),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gated_rms_norm/uniform_buffer"),
            size: std::mem::size_of::<GatedRmsNormParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        return Self {
            bind_group_layout,
            pipeline,
            weight_buffer,
            uniform_buffer,
            num_value_heads,
            value_head_dim,
            epsilon,
        };
    }

    pub fn compute(&self, device: &wgpu::Device, queue: &wgpu::Queue, src_buffer: &wgpu::Buffer, gate_buffer: &wgpu::Buffer, seq_len: usize) {
        let uniform = GatedRmsNormParams {
            num_heads: self.num_value_heads as u32,
            head_dim: self.value_head_dim as u32,
            seq_len: seq_len as u32,
            hidden_offset: 0,
            hidden_token_stride: self.num_value_heads as u32 * self.value_head_dim as u32,
            hidden_head_stride: self.value_head_dim as u32,
            gate_offset: 0,
            gate_token_stride: self.num_value_heads as u32 * self.value_head_dim as u32,
            gate_head_stride: self.value_head_dim as u32,
            weight_offset: 0,
            eps: self.epsilon,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gated_rms_norm/bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gate_buffer.as_entire_binding(),
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
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gated_rms_norm/command_encoder"),
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gated_rms_norm/compute_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(self.num_value_heads as u32 * seq_len as u32, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }
}
