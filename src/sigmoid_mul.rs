/// Mirrors `Params` in `wgsl-shaders/sigmoid_mul.wgsl`.
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct SigmoidMulParams {
    hidden_offset: u32,
    hidden_token_stride: u32,
    hidden_head_stride: u32,

    gate_offset: u32,
    gate_token_stride: u32,
    gate_head_stride: u32,

    num_heads: u32,
    head_dim: u32,
    seq_len: u32,
}

pub struct SigmoidMulInplaceWebgpu {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    uniform_buffer: wgpu::Buffer,

    hidden_size: usize,
}

impl SigmoidMulInplaceWebgpu {
    pub fn new(device: &wgpu::Device, hidden_size: usize) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sigmoid_mul/shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("wgsl-shaders/sigmoid_mul.wgsl").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sigmoid_mul/bind_group_layout"),
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
            label: Some("sigmoid_mul/pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sigmoid_mul/pipeline"),
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
            label: Some("sigmoid_mul/uniform_buffer"),
            size: std::mem::size_of::<SigmoidMulParams>() as wgpu::BufferAddress,
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

    /// Both buffers are tightly packed `[seq_len, hidden_size]`. Equivalent to
    /// `compute_strided` with `num_heads = 1, head_dim = hidden_size`.
    pub fn compute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src_buffer: &wgpu::Buffer,
        gate_buffer: &wgpu::Buffer,
        seq_len: usize,
    ) {
        self.compute_strided(
            device,
            queue,
            src_buffer,
            gate_buffer,
            0,
            self.hidden_size,
            self.hidden_size,
            0,
            self.hidden_size,
            self.hidden_size,
            1,
            self.hidden_size,
            seq_len,
        );
    }

    /// Apply sigmoid(gate) * hidden in place, with explicit per-axis offsets
    /// and strides for both buffers. Use this when the gate is a slice of a
    /// wider interleaved tensor (e.g. the gate half of `q_proj` output, where
    /// gate_head_stride = head_dim * 2 because Q and gate are interleaved per
    /// head).
    pub fn compute_strided(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src_buffer: &wgpu::Buffer,
        gate_buffer: &wgpu::Buffer,
        hidden_offset: usize,
        hidden_token_stride: usize,
        hidden_head_stride: usize,
        gate_offset: usize,
        gate_token_stride: usize,
        gate_head_stride: usize,
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
    ) {
        let uniform = SigmoidMulParams {
            hidden_offset: hidden_offset as u32,
            hidden_token_stride: hidden_token_stride as u32,
            hidden_head_stride: hidden_head_stride as u32,
            gate_offset: gate_offset as u32,
            gate_token_stride: gate_token_stride as u32,
            gate_head_stride: gate_head_stride as u32,
            num_heads: num_heads as u32,
            head_dim: head_dim as u32,
            seq_len: seq_len as u32,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sigmoid_mul/bind_group"),
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
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sigmoid_mul/command_encoder"),
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sigmoid_mul/compute_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(seq_len as u32, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }
}
