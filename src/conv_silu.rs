use bytemuck;
use safetensors::tensor::TensorView;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ConvSiluParams {
    // Channel group dimensions: layout is [Q, K, V] contiguous per token
    q_dim: u32,
    k_dim: u32,
    v_dim: u32,

    seq_len: u32,
    kernel_size: u32,

    // Elements between consecutive tokens (>= q_dim + k_dim + v_dim when padded)
    stride_src_token: u32,
    stride_dst_token: u32,

    // Per-group mode: 0 = copy, 1 = conv1d + silu
    q_mode: u32,
    k_mode: u32,
    v_mode: u32,
}

/// Per-channel-group processing mode for ConvSilu.
#[derive(Debug, Clone, Copy)]
pub enum ChannelMode {
    /// Copy the channel values from src to dst unchanged.
    Copy = 0,
    /// Apply depthwise causal conv1d followed by SiLU activation.
    ConvSilu = 1,
}

pub struct ConvSiluWebgpu {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    weights_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,

    // Model dimensions
    q_dim: usize,
    k_dim: usize,
    v_dim: usize,
    kernel_size: usize,

    // Per-group modes
    q_mode: ChannelMode,
    k_mode: ChannelMode,
    v_mode: ChannelMode,
}

impl ConvSiluWebgpu {
    pub fn new<'data>(
        device: &wgpu::Device,
        weights: TensorView<'data>,
        q_dim: usize,
        k_dim: usize,
        v_dim: usize,
        kernel_size: usize,
        q_mode: ChannelMode,
        k_mode: ChannelMode,
        v_mode: ChannelMode,
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
            size: std::mem::size_of::<ConvSiluParams>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            bind_group_layout,
            pipeline,
            weights_buffer,
            uniform_buffer,
            q_dim,
            k_dim,
            v_dim,
            kernel_size,
            q_mode,
            k_mode,
            v_mode,
        }
    }

    pub fn compute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src_buffer: &wgpu::Buffer,
        dst_buffer: &wgpu::Buffer,
        seq_len: usize,
    ) {
        let num_channels = self.q_dim + self.k_dim + self.v_dim;
        let params = ConvSiluParams {
            q_dim: self.q_dim as u32,
            k_dim: self.k_dim as u32,
            v_dim: self.v_dim as u32,
            seq_len: seq_len as u32,
            kernel_size: self.kernel_size as u32,
            stride_src_token: num_channels as u32,
            stride_dst_token: num_channels as u32,
            q_mode: self.q_mode as u32,
            k_mode: self.k_mode as u32,
            v_mode: self.v_mode as u32,
        };
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[params]),
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("conv_silu/bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src_buffer.as_entire_binding(),
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
            let num_workgroups = (seq_len * num_channels + workgroup_size - 1) / workgroup_size;
            compute_pass.dispatch_workgroups(num_workgroups as u32, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }
}
