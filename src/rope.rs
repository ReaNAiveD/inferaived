#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RopeUniform {
    q_offset: u32,
    k_offset: u32,
    stride_token: u32,
    stride_head: u32,
    head_dim: u32,
    num_heads: u32,
    seq_len: u32,
    n_dims: u32,
    theta_scale: f32,
    pos_offset: u32,
}

pub struct RopeWebgpu {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    uniform_buffer: wgpu::Buffer,
    head_dim: usize,
    num_heads: usize,
    qkv_dim: usize,    // q_dim + k_dim + v_dim (6144)
    n_dims: usize,
    theta_scale: f32,
}

impl RopeWebgpu {
    pub fn new(
        device: &wgpu::Device,
        head_dim: usize,
        num_heads: usize,
        qkv_dim: usize,
        rope_theta: f32,
        partial_rotary_factor: f32,
    ) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rope/shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "wgsl-shaders/rope.wgsl"
            ))),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rope/bind_group_layout"),
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
            label: Some("rope/pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rope/pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("rope"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rope/uniform_buffer"),
            size: std::mem::size_of::<RopeUniform>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let n_dims = (head_dim as f32 * partial_rotary_factor).floor() as usize;
        let theta_scale = rope_theta.powf(-2f32 / n_dims as f32);
        Self {
            bind_group_layout,
            pipeline,
            uniform_buffer,
            head_dim,
            qkv_dim,
            num_heads,
            n_dims,
            theta_scale,
        }
    }

    pub fn compute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src_buffer: &wgpu::Buffer,
        seq_len: usize,
        pos_offset: usize,
    ) {
        let uniform_data = RopeUniform {
            q_offset: 0,
            k_offset: (self.head_dim * self.num_heads) as u32,
            stride_token: self.qkv_dim as u32,
            stride_head: self.head_dim as u32,
            head_dim: self.head_dim as u32,
            num_heads: self.num_heads as u32,
            seq_len: seq_len as u32,
            n_dims: self.n_dims as u32,
            theta_scale: self.theta_scale,
            pos_offset: pos_offset as u32,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniform_data]));
         let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rope/bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rope/command_encoder"),
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rope/compute_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let total_invocations = (self.n_dims / 2) * self.num_heads * seq_len;
            let workgroup_size = 256usize;
            let workgroup_count = (total_invocations / workgroup_size) as u32 + 1;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }
}
