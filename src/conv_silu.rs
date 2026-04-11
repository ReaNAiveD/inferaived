use bytemuck;
use safetensors::tensor::TensorView;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ConvSiluUniform {
    offset_src: u32,
    offset_weights: u32,
    offset_dst: u32,
    stride_src1: u32,     // in elements
    stride_weights1: u32, // in bf16 elements
    stride_dst1: u32,     // in elements
    ne0: u32,
    n_rows: u32,
    kernel_size: u32,
}

pub struct ConvSiluWebgpu {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    weights_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    kernel_size: usize,
    v_dim: usize,      // num_value_heads × value_head_dim (2048)
    qkv_dim: usize,    // q_dim + k_dim + v_dim (6144)
    v_offset: usize,   // q_dim + k_dim (4096)
}

impl ConvSiluWebgpu {
    pub fn new<'data>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        weights: TensorView<'data>,
        v_dim: usize,       // num_value_heads × value_head_dim (2048)
        qkv_dim: usize,     // q_dim + k_dim + v_dim (6144)
        v_offset: usize,    // q_dim + k_dim (4096)
        kernel_size: usize,
    ) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("conv_silu/shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "wgsl-shaders/depthwise_causal_conv_silu.wgsl"
            ))),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("conv_silu/bind_group_layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("conv_silu/pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("conv1d_silu"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let weights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("conv_silu/weights_buffer"),
            contents: bytemuck::cast_slice(weights.data()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("conv_silu/uniform_buffer"),
            size: std::mem::size_of::<ConvSiluUniform>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            bind_group_layout,
            pipeline,
            weights_buffer,
            uniform_buffer,
            kernel_size,
            v_dim,
            qkv_dim,
            v_offset,
        }
    }

    pub fn compute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        qkv_buffer: &wgpu::Buffer,
        dst_buffer: &wgpu::Buffer,
        sequence_length: usize,
    ) {
        let uniform_data = ConvSiluUniform {
            offset_src: self.v_offset as u32,
            offset_weights: 0,
            offset_dst: 0,
            stride_src1: self.qkv_dim as u32,
            stride_weights1: self.kernel_size as u32,
            stride_dst1: self.v_dim as u32,
            ne0: self.v_dim as u32,
            n_rows: sequence_length as u32,
            kernel_size: self.kernel_size as u32,
        };
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[uniform_data]),
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("conv_silu/bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: qkv_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.weights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dst_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("conv_silu/command_encoder"),
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("conv_silu/compute_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroup_size = 256usize;
            let num_workgroups = (sequence_length * self.v_dim + workgroup_size - 1) / workgroup_size;
            compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }
}
