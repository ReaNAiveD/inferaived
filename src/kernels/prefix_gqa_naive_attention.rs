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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_test_utils::*;

    /// Allocate a 4-byte uniform buffer holding `value` as a `u32`.
    fn make_u32_uniform(device: &wgpu::Device, queue: &wgpu::Queue, value: u32) -> wgpu::Buffer {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prefix_attn_test/u32_uniform"),
            size: std::mem::size_of::<u32>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, bytemuck::bytes_of(&value));
        buffer
    }

    /// CPU reference for naive PrefixLM grouped-query attention. A query at
    /// absolute position `p = q_token + position_offset` attends to:
    ///   * `[0, prefix_len)`  when `p < prefix_len`  (bidirectional prefix)
    ///   * `[0, p]`           when `p >= prefix_len`  (causal)
    #[allow(clippy::too_many_arguments)]
    fn cpu_prefix_gqa_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        num_q_heads: usize,
        num_kv_heads: usize,
        q_dim: usize,
        v_dim: usize,
        seq_len: usize,
        position_offset: usize,
        prefix_len: usize,
    ) -> Vec<f32> {
        let q_token_stride = num_q_heads * q_dim;
        let k_token_stride = num_kv_heads * q_dim;
        let v_token_stride = num_kv_heads * v_dim;
        let output_token_stride = num_q_heads * v_dim;
        let num_q_per_kv = num_q_heads / num_kv_heads;
        let softmax_scale = 1.0 / (q_dim as f32).sqrt();

        let mut output = vec![0.0f32; seq_len * output_token_stride];

        for q_token in 0..seq_len {
            let p = q_token + position_offset;
            let k_token_max = if p < prefix_len { prefix_len - 1 } else { p };
            for q_head in 0..num_q_heads {
                let kv_head = q_head / num_q_per_kv;

                let mut scores = Vec::with_capacity(k_token_max + 1);
                for k_token in 0..=k_token_max {
                    let mut dot = 0.0f32;
                    for d in 0..q_dim {
                        dot += q[q_token * q_token_stride + q_head * q_dim + d]
                            * k[k_token * k_token_stride + kv_head * q_dim + d];
                    }
                    scores.push(dot * softmax_scale);
                }

                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
                let sum_exp: f32 = exp_scores.iter().sum();

                for vd in 0..v_dim {
                    let mut acc = 0.0f32;
                    for (k_token, &w) in exp_scores.iter().enumerate() {
                        acc += (w / sum_exp) * v[k_token * v_token_stride + kv_head * v_dim + vd];
                    }
                    output[q_token * output_token_stride + q_head * v_dim + vd] = acc;
                }
            }
        }
        output
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_gpu(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        num_q_heads: usize,
        num_kv_heads: usize,
        q_dim: usize,
        v_dim: usize,
        seq_len: usize,
        position_offset: u32,
        prefix_len: u32,
    ) -> Vec<f32> {
        let gpu =
            PrefixGqaNaiveAttentionWebgpu::new(device, num_q_heads, num_kv_heads, q_dim, v_dim);
        let q_buf = upload_f32(device, q);
        let k_buf = upload_f32(device, k);
        let v_buf = upload_f32(device, v);
        let out_buf = create_f32_buffer(device, seq_len * num_q_heads * v_dim);
        let sz = std::mem::size_of::<f32>() as u32;
        let q_view =
            BufferView::new_3d_tight(&q_buf, seq_len as u32, num_q_heads as u32, q_dim as u32, sz);
        let k_view = BufferView::new_3d_tight(
            &k_buf,
            seq_len as u32,
            num_kv_heads as u32,
            q_dim as u32,
            sz,
        );
        let v_view = BufferView::new_3d_tight(
            &v_buf,
            seq_len as u32,
            num_kv_heads as u32,
            v_dim as u32,
            sz,
        );
        let out_view = BufferView::new_3d_tight(
            &out_buf,
            seq_len as u32,
            num_q_heads as u32,
            v_dim as u32,
            sz,
        );
        let position_buffer = make_u32_uniform(device, queue, position_offset);
        let prefix_buffer = make_u32_uniform(device, queue, prefix_len);
        let runner = gpu.plan(
            device,
            queue,
            q_view,
            k_view,
            v_view,
            out_view,
            &position_buffer,
            &prefix_buffer,
        );
        run_blocking_compute(device, queue, |cp| runner.forward(cp));
        download_f32(device, queue, &out_buf, seq_len * num_q_heads * v_dim)
    }

    /// Prefill at offset 0 with the whole sequence as one bidirectional prefix:
    /// every query row sees all `[0, seq_len)`, including future tokens.
    #[tokio::test]
    async fn test_prefix_bidirectional_prefill() {
        let (device, queue) = gpu_or_skip!();
        let seq_len = 4;
        let num_q_heads = 4;
        let num_kv_heads = 2;
        let q_dim = 16;
        let v_dim = 16;

        let q: Vec<f32> = (0..seq_len * num_q_heads * q_dim)
            .map(|i| ((i as f32) * 0.05).sin() * 0.3)
            .collect();
        let k: Vec<f32> = (0..seq_len * num_kv_heads * q_dim)
            .map(|i| ((i as f32) * 0.07).cos() * 0.3)
            .collect();
        let v: Vec<f32> = (0..seq_len * num_kv_heads * v_dim)
            .map(|i| ((i as f32) * 0.03).sin() * 0.5)
            .collect();

        // prefix_len == seq_len ⇒ all rows bidirectional over [0, seq_len).
        let expected = cpu_prefix_gqa_attention(
            &q, &k, &v, num_q_heads, num_kv_heads, q_dim, v_dim, seq_len, 0, seq_len,
        );
        let actual = run_gpu(
            &device, &queue, &q, &k, &v, num_q_heads, num_kv_heads, q_dim, v_dim, seq_len, 0,
            seq_len as u32,
        )
        .await;
        assert_approx_eq(&actual, &expected, 1e-3);

        // Row 0 must DIFFER from a causal row 0 (which would see only token 0),
        // proving the bidirectional reach actually takes effect.
        let causal = cpu_prefix_gqa_attention(
            &q, &k, &v, num_q_heads, num_kv_heads, q_dim, v_dim, seq_len, 0, 1,
        );
        let row0_prefix = &actual[0..num_q_heads * v_dim];
        let row0_causal = &causal[0..num_q_heads * v_dim];
        let max_diff = row0_prefix
            .iter()
            .zip(row0_causal)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-3,
            "prefix row 0 should differ from causal row 0 (max_diff={max_diff})",
        );
    }

    /// Mixed batch: a prefix block `[0, prefix_len)` followed by causal rows.
    /// Rows inside the prefix are bidirectional; rows at/after `prefix_len` are
    /// causal `[0, p]`.
    #[tokio::test]
    async fn test_prefix_then_causal() {
        let (device, queue) = gpu_or_skip!();
        let seq_len = 6;
        let prefix_len = 3usize;
        let num_q_heads = 2;
        let num_kv_heads = 1;
        let q_dim = 16;
        let v_dim = 16;

        let q: Vec<f32> = (0..seq_len * num_q_heads * q_dim)
            .map(|i| ((i as f32) * 0.04).sin() * 0.3)
            .collect();
        let k: Vec<f32> = (0..seq_len * num_kv_heads * q_dim)
            .map(|i| ((i as f32) * 0.06).cos() * 0.3)
            .collect();
        let v: Vec<f32> = (0..seq_len * num_kv_heads * v_dim)
            .map(|i| ((i as f32) * 0.02).sin() * 0.5)
            .collect();

        let expected = cpu_prefix_gqa_attention(
            &q, &k, &v, num_q_heads, num_kv_heads, q_dim, v_dim, seq_len, 0, prefix_len,
        );
        let actual = run_gpu(
            &device, &queue, &q, &k, &v, num_q_heads, num_kv_heads, q_dim, v_dim, seq_len, 0,
            prefix_len as u32,
        )
        .await;
        assert_approx_eq(&actual, &expected, 1e-3);
    }

    /// With `prefix_len <= 1` the kernel must reduce to pure causal attention,
    /// so a single decode row at absolute position `position_offset` matches a
    /// causal reference over the full cache.
    #[tokio::test]
    async fn test_prefix_decode_equals_causal() {
        let (device, queue) = gpu_or_skip!();
        let cache_len = 5;
        let num_q_heads = 4;
        let num_kv_heads = 2;
        let q_dim = 16;
        let v_dim = 16;
        let position_offset = cache_len - 1; // decode the last row

        // Single query row.
        let q: Vec<f32> = (0..num_q_heads * q_dim)
            .map(|i| ((i as f32) * 0.05).sin() * 0.3)
            .collect();
        let k: Vec<f32> = (0..cache_len * num_kv_heads * q_dim)
            .map(|i| ((i as f32) * 0.07).cos() * 0.3)
            .collect();
        let v: Vec<f32> = (0..cache_len * num_kv_heads * v_dim)
            .map(|i| ((i as f32) * 0.03).sin() * 0.5)
            .collect();

        // CPU reference: one query row attending causally over [0, position].
        let mut expected = vec![0.0f32; num_q_heads * v_dim];
        let num_q_per_kv = num_q_heads / num_kv_heads;
        let softmax_scale = 1.0 / (q_dim as f32).sqrt();
        for q_head in 0..num_q_heads {
            let kv_head = q_head / num_q_per_kv;
            let mut scores = Vec::with_capacity(position_offset + 1);
            for k_token in 0..=position_offset {
                let mut dot = 0.0f32;
                for d in 0..q_dim {
                    dot += q[q_head * q_dim + d]
                        * k[k_token * num_kv_heads * q_dim + kv_head * q_dim + d];
                }
                scores.push(dot * softmax_scale);
            }
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
            let sum_exp: f32 = exp_scores.iter().sum();
            for vd in 0..v_dim {
                let mut acc = 0.0f32;
                for (k_token, &w) in exp_scores.iter().enumerate() {
                    acc += (w / sum_exp)
                        * v[k_token * num_kv_heads * v_dim + kv_head * v_dim + vd];
                }
                expected[q_head * v_dim + vd] = acc;
            }
        }

        // prefix_len = 1 ⇒ the single row at p=position_offset is causal.
        let actual = run_gpu_decode(
            &device,
            &queue,
            &q,
            &k,
            &v,
            num_q_heads,
            num_kv_heads,
            q_dim,
            v_dim,
            cache_len,
            position_offset as u32,
            1,
        )
        .await;
        assert_approx_eq(&actual, &expected, 1e-3);
    }

    /// GPU helper for the decode shape: a single Q row attends over a K/V cache
    /// of `cache_len` rows (q rows != kv rows, unlike [`run_gpu`]).
    #[allow(clippy::too_many_arguments)]
    async fn run_gpu_decode(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        num_q_heads: usize,
        num_kv_heads: usize,
        q_dim: usize,
        v_dim: usize,
        cache_len: usize,
        position_offset: u32,
        prefix_len: u32,
    ) -> Vec<f32> {
        let gpu =
            PrefixGqaNaiveAttentionWebgpu::new(device, num_q_heads, num_kv_heads, q_dim, v_dim);
        let q_buf = upload_f32(device, q);
        let k_buf = upload_f32(device, k);
        let v_buf = upload_f32(device, v);
        let out_buf = create_f32_buffer(device, num_q_heads * v_dim);
        let sz = std::mem::size_of::<f32>() as u32;
        let q_view = BufferView::new_3d_tight(&q_buf, 1, num_q_heads as u32, q_dim as u32, sz);
        let k_view = BufferView::new_3d_tight(
            &k_buf,
            cache_len as u32,
            num_kv_heads as u32,
            q_dim as u32,
            sz,
        );
        let v_view = BufferView::new_3d_tight(
            &v_buf,
            cache_len as u32,
            num_kv_heads as u32,
            v_dim as u32,
            sz,
        );
        let out_view = BufferView::new_3d_tight(&out_buf, 1, num_q_heads as u32, v_dim as u32, sz);
        let position_buffer = make_u32_uniform(device, queue, position_offset);
        let prefix_buffer = make_u32_uniform(device, queue, prefix_len);
        let runner = gpu.plan(
            device,
            queue,
            q_view,
            k_view,
            v_view,
            out_view,
            &position_buffer,
            &prefix_buffer,
        );
        run_blocking_compute(device, queue, |cp| runner.forward(cp));
        download_f32(device, queue, &out_buf, num_q_heads * v_dim)
    }
}
