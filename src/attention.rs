#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct CausalGqaNaiveAttentionParams {
    q_offset: u32,
    q_token_stride: u32,
    q_head_stride: u32,
    k_offset: u32,
    k_token_stride: u32,
    k_head_stride: u32,
    v_offset: u32,
    v_token_stride: u32,
    v_head_stride: u32,
    output_offset: u32,
    output_token_stride: u32,
    output_head_stride: u32,

    num_q_heads: u32,
    num_kv_heads: u32,
    q_dim: u32,
    v_dim: u32,
    seq_len: u32,
}

pub struct CausalGqaNaiveAttentionWebgpu {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    uniform_buffer: wgpu::Buffer,

    num_q_heads: usize,
    num_kv_heads: usize,
    q_dim: usize,
    v_dim: usize,
}

impl CausalGqaNaiveAttentionWebgpu {
    const WORKGROUP_SIZE: u32 = 128;

    pub fn new(
        device: &wgpu::Device,
        num_q_heads: usize,
        num_kv_heads: usize,
        q_dim: usize,
        v_dim: usize,
    ) -> Self {
        debug_assert!(
            num_q_heads % num_kv_heads == 0,
            "num_q_heads ({}) must be divisible by num_kv_heads ({})",
            num_q_heads,
            num_kv_heads,
        );
        // The shader keeps a private `array<f32, MAX_V_PER_THREAD>` of size 4
        // per thread for the V-weighted accumulator. With workgroup_size=128
        // this caps v_dim at 4 * 128 = 512, which covers all current models.
        let max_v_per_thread = 4u32;
        debug_assert!(
            (v_dim as u32) <= max_v_per_thread * Self::WORKGROUP_SIZE,
            "v_dim ({}) exceeds MAX_V_PER_THREAD ({}) * workgroup_size ({}). \
             Increase MAX_V_PER_THREAD in causal_gqa_naive_attention.wgsl or workgroup_size.",
            v_dim,
            max_v_per_thread,
            Self::WORKGROUP_SIZE,
        );
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("causal_gqa_naive_attention/shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "wgsl-shaders/causal_gqa_naive_attention.wgsl"
            ))),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("causal_gqa_naive_attention/bind_group_layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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
            label: Some("causal_gqa_naive_attention/pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("causal_gqa_naive_attention/pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("workgroup_size".into(), Self::WORKGROUP_SIZE as f64)],
                zero_initialize_workgroup_memory: true,
            },
            cache: None,
        });
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("causal_gqa_naive_attention/uniform_buffer"),
            size: std::mem::size_of::<CausalGqaNaiveAttentionParams>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            bind_group_layout,
            pipeline,
            uniform_buffer,
            num_q_heads,
            num_kv_heads,
            q_dim,
            v_dim,
        }
    }

    /// Run causal GQA attention.
    ///
    /// Buffers are assumed to be tightly packed in row-major layout starting
    /// at offset 0:
    /// - `q_buffer`: `[seq_len, num_q_heads, q_dim]`
    /// - `k_buffer`: `[seq_len, num_kv_heads, q_dim]`
    /// - `v_buffer`: `[seq_len, num_kv_heads, v_dim]`
    /// - `output_buffer`: `[seq_len, num_q_heads, v_dim]`
    pub fn compute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        q_buffer: &wgpu::Buffer,
        k_buffer: &wgpu::Buffer,
        v_buffer: &wgpu::Buffer,
        output_buffer: &wgpu::Buffer,
        seq_len: usize,
    ) {
        let params = CausalGqaNaiveAttentionParams {
            q_offset: 0,
            q_token_stride: (self.num_q_heads * self.q_dim) as u32,
            q_head_stride: self.q_dim as u32,
            k_offset: 0,
            k_token_stride: (self.num_kv_heads * self.q_dim) as u32,
            k_head_stride: self.q_dim as u32,
            v_offset: 0,
            v_token_stride: (self.num_kv_heads * self.v_dim) as u32,
            v_head_stride: self.v_dim as u32,
            output_offset: 0,
            output_token_stride: (self.num_q_heads * self.v_dim) as u32,
            output_head_stride: self.v_dim as u32,
            num_q_heads: self.num_q_heads as u32,
            num_kv_heads: self.num_kv_heads as u32,
            q_dim: self.q_dim as u32,
            v_dim: self.v_dim as u32,
            seq_len: seq_len as u32,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[params]));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("causal_gqa_naive_attention/bind_group"),
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
                    resource: v_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("causal_gqa_naive_attention/command_encoder"),
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("causal_gqa_naive_attention/compute_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            // One workgroup per (q_token, q_head).
            let workgroup_count = (seq_len * self.num_q_heads) as u32;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }
}

