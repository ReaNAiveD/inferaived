/// Mirrors `Params` in `wgsl-shaders/rope.wgsl` exactly. Field order and
/// count must stay in sync with the shader's uniform struct.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RopeParams {
    q_offset: u32,
    q_token_stride: u32,
    q_head_stride: u32,
    k_offset: u32,
    k_token_stride: u32,
    k_head_stride: u32,

    num_q_heads: u32,
    num_k_heads: u32,
    seq_len: u32,
    num_rotated_dims: u32,

    theta_scale: f32,
    position_offset: u32,
}

pub struct RopeInplaceWebgpu {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    uniform_buffer: wgpu::Buffer,
    head_dim: usize,
    num_q_heads: usize,
    num_k_heads: usize,
    num_rotated_dims: usize,
    theta_scale: f32,
}

impl RopeInplaceWebgpu {
    const WORKGROUP_SIZE: u32 = 256;

    pub fn new(
        device: &wgpu::Device,
        num_q_heads: usize,
        num_k_heads: usize,
        head_dim: usize,
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
            size: std::mem::size_of::<RopeParams>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let num_rotated_dims = (head_dim as f32 * partial_rotary_factor).floor() as usize;
        let theta_scale = rope_theta.powf(-2f32 / num_rotated_dims as f32);
        Self {
            bind_group_layout,
            pipeline,
            uniform_buffer,
            head_dim,
            num_q_heads,
            num_k_heads,
            num_rotated_dims,
            theta_scale,
        }
    }

    /// Apply RoPE in-place to Q and K.
    ///
    /// Both buffers are assumed to be tightly packed `[seq, num_*_heads, head_dim]`
    /// in row-major layout starting at offset 0.
    pub fn compute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        q_buffer: &wgpu::Buffer,
        k_buffer: &wgpu::Buffer,
        seq_len: usize,
        position_offset: usize,
    ) {
        let head_dim = self.head_dim as u32;
        let uniform_data = RopeParams {
            q_offset: 0,
            q_token_stride: self.num_q_heads as u32 * head_dim,
            q_head_stride: head_dim,
            k_offset: 0,
            k_token_stride: self.num_k_heads as u32 * head_dim,
            k_head_stride: head_dim,
            num_q_heads: self.num_q_heads as u32,
            num_k_heads: self.num_k_heads as u32,
            seq_len: seq_len as u32,
            num_rotated_dims: self.num_rotated_dims as u32,
            theta_scale: self.theta_scale,
            position_offset: position_offset as u32,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniform_data]));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rope/bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: k_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
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
            let max_heads = self.num_q_heads.max(self.num_k_heads);
            let total_invocations =
                ((self.num_rotated_dims / 2) * max_heads * seq_len) as u32;
            let workgroup_count =
                (total_invocations + Self::WORKGROUP_SIZE - 1) / Self::WORKGROUP_SIZE;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }
}
