use crate::buffer_view::BufferView;

#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct PrefixGqaNaiveAttentionParams {
    q_token_stride: u32,
    q_head_stride: u32,
    k_token_stride: u32,
    k_head_stride: u32,
    v_token_stride: u32,
    v_head_stride: u32,
    output_token_stride: u32,
    output_head_stride: u32,

    num_q_heads: u32,
    num_kv_heads: u32,
    q_dim: u32,
    v_dim: u32,
    seq_len: u32,
}

/// Naive PrefixLM grouped-query attention. Identical to
/// [`CausalGqaNaiveAttentionWebgpu`](super::attention::CausalGqaNaiveAttentionWebgpu)
/// except a query at absolute position `p` attends to the whole prefix block
/// `[0, prefix_len)` when `p < prefix_len` (bidirectional) and `[0, p]`
/// otherwise (causal). With `prefix_len <= 1` it is numerically identical to the
/// causal kernel, so single-row decode is unchanged.
pub struct PrefixGqaNaiveAttentionWebgpu {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,

    num_q_heads: usize,
    num_kv_heads: usize,
    qk_head_dim: usize,
    v_head_dim: usize,
}

impl PrefixGqaNaiveAttentionWebgpu {
    const WORKGROUP_SIZE: u32 = 128;

    pub fn new(
        device: &wgpu::Device,
        num_q_heads: usize,
        num_kv_heads: usize,
        qk_head_dim: usize,
        v_head_dim: usize,
    ) -> Self {
        debug_assert!(
            num_q_heads % num_kv_heads == 0,
            "num_q_heads ({}) must be divisible by num_kv_heads ({})",
            num_q_heads,
            num_kv_heads,
        );
        let max_v_per_thread = 4u32;
        debug_assert!(
            (v_head_dim as u32) <= max_v_per_thread * Self::WORKGROUP_SIZE,
            "v_head_dim ({}) exceeds MAX_V_PER_THREAD ({}) * workgroup_size ({}). \
             Increase MAX_V_PER_THREAD in prefix_gqa_naive_attention.wgsl or workgroup_size.",
            v_head_dim,
            max_v_per_thread,
            Self::WORKGROUP_SIZE,
        );
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prefix_gqa_naive_attention/shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "wgsl-shaders/prefix_gqa_naive_attention.wgsl"
            ))),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prefix_gqa_naive_attention/bind_group_layout"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
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
            label: Some("prefix_gqa_naive_attention/pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("prefix_gqa_naive_attention/pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("workgroup_size".into(), Self::WORKGROUP_SIZE as f64)],
                zero_initialize_workgroup_memory: true,
            },
            cache: None,
        });
        Self {
            bind_group_layout,
            pipeline,
            num_q_heads,
            num_kv_heads,
            qk_head_dim,
            v_head_dim,
        }
    }

    /// Bake the per-buffer bindings into a [`PrefixGqaNaiveAttentionWebgpuRunner`]
    /// for repeated dispatch into a caller-owned compute pass.
    ///
    /// `position_buffer` is a `1 × u32` uniform holding the absolute position of
    /// `q` row 0; `prefix_buffer` is a `1 × u32` uniform holding the
    /// bidirectional prefix length.
    #[allow(clippy::too_many_arguments)]
    pub fn plan(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        q: BufferView<'_>,
        k: BufferView<'_>,
        v: BufferView<'_>,
        output: BufferView<'_>,
        position_buffer: &wgpu::Buffer,
        prefix_buffer: &wgpu::Buffer,
    ) -> PrefixGqaNaiveAttentionWebgpuRunner {
        debug_assert_eq!(q.rank, 3);
        debug_assert_eq!(k.rank, 3);
        debug_assert_eq!(v.rank, 3);
        debug_assert_eq!(output.rank, 3);
        debug_assert_eq!(q.shape[0], output.shape[0]);
        debug_assert_eq!(k.shape[0], v.shape[0]);
        debug_assert_eq!(q.shape[1] as usize, self.num_q_heads);
        debug_assert_eq!(output.shape[1] as usize, self.num_q_heads);
        debug_assert_eq!(k.shape[1] as usize, self.num_kv_heads);
        debug_assert_eq!(v.shape[1] as usize, self.num_kv_heads);
        debug_assert_eq!(q.shape[2] as usize, self.qk_head_dim);
        debug_assert_eq!(k.shape[2] as usize, self.qk_head_dim);
        debug_assert_eq!(v.shape[2] as usize, self.v_head_dim);
        debug_assert_eq!(output.shape[2] as usize, self.v_head_dim);
        let num_q_rows = q.shape[0];
        let params = PrefixGqaNaiveAttentionParams {
            q_token_stride: q.stride[0],
            q_head_stride: q.stride[1],
            k_token_stride: k.stride[0],
            k_head_stride: k.stride[1],
            v_token_stride: v.stride[0],
            v_head_stride: v.stride[1],
            output_token_stride: output.stride[0],
            output_head_stride: output.stride[1],
            num_q_heads: self.num_q_heads as u32,
            num_kv_heads: self.num_kv_heads as u32,
            q_dim: self.qk_head_dim as u32,
            v_dim: self.v_head_dim as u32,
            seq_len: num_q_rows,
        };
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prefix_gqa_naive_attention_runner/uniform_buffer"),
            size: std::mem::size_of::<PrefixGqaNaiveAttentionParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buffer, 0, bytemuck::cast_slice(&[params]));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prefix_gqa_naive_attention_runner/bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: k.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: v.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: position_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: prefix_buffer.as_entire_binding(),
                },
            ],
        });
        let workgroup_count = num_q_rows * self.num_q_heads as u32;
        let dispatch_grid = crate::dispatch::split_1d_into_2d(workgroup_count);
        PrefixGqaNaiveAttentionWebgpuRunner {
            pipeline: self.pipeline.clone(),
            bind_group,
            dispatch_grid,
        }
    }
}

pub struct PrefixGqaNaiveAttentionWebgpuRunner {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    dispatch_grid: (u32, u32),
}

impl PrefixGqaNaiveAttentionWebgpuRunner {
    pub fn forward(&self, cpass: &mut wgpu::ComputePass<'_>) {
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        let (x, y) = self.dispatch_grid;
        cpass.dispatch_workgroups(x, y, 1);
    }
}
