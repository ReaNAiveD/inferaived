use crate::buffer_view::BufferView;
use std::ops::Range;

/// Encode the `visibility` storage-buffer contents for
/// [`MaskedBlockAttentionWebgpu::plan`].
pub fn encode_visibility(shared: &[Range<u32>], causal_start: u32) -> Vec<u32> {
    let n = shared.len() + 1;
    let mut buf = Vec::with_capacity(1 + 2 * n);
    buf.push(n as u32);
    for r in shared {
        buf.push(r.start);
        buf.push(r.end);
    }
    buf.push(causal_start);
    buf.push(0);
    buf
}

/// Uniform parameters for [`MaskedBlockAttentionWebgpu`]. All strides are in
/// elements (f32), matching the [`BufferView`] convention.
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct MaskedBlockAttentionParams {
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
    num_q_rows: u32,
}

/// Grouped-query attention over a shared KV row pool, gated by an explicit
/// visible-range list instead of a causal cutoff.
pub struct MaskedBlockAttentionWebgpu {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,

    num_q_heads: usize,
    num_kv_heads: usize,
    qk_head_dim: usize,
    v_head_dim: usize,
}

impl MaskedBlockAttentionWebgpu {
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
            "v_head_dim ({}) exceeds MAX_V_PER_THREAD ({}) * workgroup_size ({}).",
            v_head_dim,
            max_v_per_thread,
            Self::WORKGROUP_SIZE,
        );
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("masked_block_attention/shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "wgsl-shaders/masked_block_attention.wgsl"
            ))),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("masked_block_attention/bind_group_layout"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                uniform_entry(4),
                storage_entry(5, true),
                uniform_entry(6),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("masked_block_attention/pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("masked_block_attention/pipeline"),
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

    /// Bake the per-buffer bindings into a [`MaskedBlockAttentionWebgpuRunner`]
    /// for dispatch into a caller-owned compute pass.
    pub fn plan(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        q: BufferView<'_>,
        k: BufferView<'_>,
        v: BufferView<'_>,
        output: BufferView<'_>,
        visibility: &wgpu::Buffer,
        scatter_position: &wgpu::Buffer,
    ) -> MaskedBlockAttentionWebgpuRunner {
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
        let params = MaskedBlockAttentionParams {
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
            num_q_rows,
        };
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("masked_block_attention_runner/uniform_buffer"),
            size: std::mem::size_of::<MaskedBlockAttentionParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buffer, 0, bytemuck::cast_slice(&[params]));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("masked_block_attention_runner/bind_group"),
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
                    resource: visibility.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: scatter_position.as_entire_binding(),
                },
            ],
        });
        let workgroup_count = num_q_rows * self.num_q_heads as u32;
        let dispatch_grid = crate::dispatch::split_1d_into_2d(workgroup_count);
        MaskedBlockAttentionWebgpuRunner {
            pipeline: self.pipeline.clone(),
            bind_group,
            dispatch_grid,
        }
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub struct MaskedBlockAttentionWebgpuRunner {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    dispatch_grid: (u32, u32),
}

impl MaskedBlockAttentionWebgpuRunner {
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

    /// Upload a `&[u32]` mask to a storage buffer.
    fn upload_u32(device: &wgpu::Device, data: &[u32]) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("masked_block_attention_test/mask"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// Upload a single `u32` to a uniform buffer (for `scatter_position`).
    fn upload_u32_uniform(device: &wgpu::Device, value: u32) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("masked_block_attention_test/scatter_position"),
            contents: bytemuck::bytes_of(&value),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// Expand a `shared` + causal-tail range spec into the dense
    /// `[num_q_rows, kv_pool_len]` row-major bitmask the CPU reference consumes.
    /// Every query row sees the shared ranges; query row `r` also sees the causal
    /// tail `[causal_start, scatter_position + 1 + r)`, mirroring the shader's
    /// `range_end` computation for the last range.
    fn expand_visibility(
        shared: &[Range<u32>],
        causal_start: u32,
        scatter_position: u32,
        num_q_rows: usize,
        kv_pool_len: usize,
    ) -> Vec<u32> {
        let mut mask = vec![0u32; num_q_rows * kv_pool_len];
        for q_row in 0..num_q_rows {
            let base = q_row * kv_pool_len;
            for r in shared {
                for k in r.clone() {
                    mask[base + k as usize] = 1;
                }
            }
            for k in causal_start..scatter_position + 1 + q_row as u32 {
                mask[base + k as usize] = 1;
            }
        }
        mask
    }

    /// CPU reference: masked grouped-query attention over a KV pool. The mask is
    /// per query row (`[num_q_rows, kv_pool_len]` row-major); for query row `r`,
    /// pool rows with `row_visible[r * kv_pool_len + k] == 0` are excluded from
    /// that row's softmax.
    #[allow(clippy::too_many_arguments)]
    fn cpu_masked_block_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        row_visible: &[u32],
        num_q_heads: usize,
        num_kv_heads: usize,
        q_dim: usize,
        v_dim: usize,
        num_q_rows: usize,
        kv_pool_len: usize,
    ) -> Vec<f32> {
        let q_token_stride = num_q_heads * q_dim;
        let k_token_stride = num_kv_heads * q_dim;
        let v_token_stride = num_kv_heads * v_dim;
        let output_token_stride = num_q_heads * v_dim;
        let num_q_per_kv = num_q_heads / num_kv_heads;
        let softmax_scale = 1.0 / (q_dim as f32).sqrt();

        let mut output = vec![0.0f32; num_q_rows * output_token_stride];
        for q_row in 0..num_q_rows {
            for q_head in 0..num_q_heads {
                let kv_head = q_head / num_q_per_kv;
                let mut rows = Vec::new();
                let mut scores = Vec::new();
                for k_token in 0..kv_pool_len {
                    if row_visible[q_row * kv_pool_len + k_token] == 0 {
                        continue;
                    }
                    let mut dot = 0.0f32;
                    for d in 0..q_dim {
                        dot += q[q_row * q_token_stride + q_head * q_dim + d]
                            * k[k_token * k_token_stride + kv_head * q_dim + d];
                    }
                    rows.push(k_token);
                    scores.push(dot * softmax_scale);
                }
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
                let sum_exp: f32 = exp_scores.iter().sum();
                for vd in 0..v_dim {
                    let mut acc = 0.0f32;
                    for (idx, &k_token) in rows.iter().enumerate() {
                        acc += (exp_scores[idx] / sum_exp)
                            * v[k_token * v_token_stride + kv_head * v_dim + vd];
                    }
                    output[q_row * output_token_stride + q_head * v_dim + vd] = acc;
                }
            }
        }
        output
    }

    #[tokio::test]
    async fn masked_attention_matches_cpu_reference() {
        let (device, queue) = gpu_or_skip!();
        let num_q_rows = 3;
        let num_q_heads = 4;
        let num_kv_heads = 2;
        let q_dim = 16;
        let v_dim = 16;
        let kv_pool_len = 7;

        let q: Vec<f32> = (0..num_q_rows * num_q_heads * q_dim)
            .map(|i| ((i as f32) * 0.05).sin() * 0.3)
            .collect();
        let k: Vec<f32> = (0..kv_pool_len * num_kv_heads * q_dim)
            .map(|i| ((i as f32) * 0.07).cos() * 0.3)
            .collect();
        let v: Vec<f32> = (0..kv_pool_len * num_kv_heads * v_dim)
            .map(|i| ((i as f32) * 0.03).sin() * 0.5)
            .collect();
        // Shared + causal range spec: every query row sees pool rows [0, 2); the
        // causal tail starts at row 2 and ends at `scatter_position + 1 + q_row`,
        // so with scatter_position=2 row 0 sees {0,1,2}, row 1 {0,1,2,3}, row 2
        // {0,1,2,3,4} — pool rows 5,6 stay hidden from all.
        let shared = [0u32..2u32];
        let causal_start: u32 = 2;
        let scatter_position_value: u32 = 2;
        let mask = expand_visibility(
            &shared,
            causal_start,
            scatter_position_value,
            num_q_rows,
            kv_pool_len,
        );

        let expected = cpu_masked_block_attention(
            &q,
            &k,
            &v,
            &mask,
            num_q_heads,
            num_kv_heads,
            q_dim,
            v_dim,
            num_q_rows,
            kv_pool_len,
        );

        let gpu = MaskedBlockAttentionWebgpu::new(&device, num_q_heads, num_kv_heads, q_dim, v_dim);
        let q_buf = upload_f32(&device, &q);
        let k_buf = upload_f32(&device, &k);
        let v_buf = upload_f32(&device, &v);
        let visibility = encode_visibility(&shared, causal_start);
        let mask_buf = upload_u32(&device, &visibility);
        let scatter_position_buf = upload_u32_uniform(&device, scatter_position_value);
        let out_buf = create_f32_buffer(&device, num_q_rows * num_q_heads * v_dim);
        let sz = std::mem::size_of::<f32>() as u32;
        let q_view = BufferView::new_3d_tight(
            &q_buf,
            num_q_rows as u32,
            num_q_heads as u32,
            q_dim as u32,
            sz,
        );
        let k_view = BufferView::new_3d_tight(
            &k_buf,
            kv_pool_len as u32,
            num_kv_heads as u32,
            q_dim as u32,
            sz,
        );
        let v_view = BufferView::new_3d_tight(
            &v_buf,
            kv_pool_len as u32,
            num_kv_heads as u32,
            v_dim as u32,
            sz,
        );
        let out_view = BufferView::new_3d_tight(
            &out_buf,
            num_q_rows as u32,
            num_q_heads as u32,
            v_dim as u32,
            sz,
        );
        let runner = gpu.plan(
            &device,
            &queue,
            q_view,
            k_view,
            v_view,
            out_view,
            &mask_buf,
            &scatter_position_buf,
        );
        run_blocking_compute(&device, &queue, |cp| runner.forward(cp));
        let actual = download_f32(&device, &queue, &out_buf, num_q_rows * num_q_heads * v_dim);

        assert_approx_eq(&actual, &expected, 1e-3);
    }

    /// With an all-ones mask covering a contiguous causal prefix, the masked
    /// kernel must reproduce the last-row decode result of plain causal
    /// attention — the correctness anchor for the parallel decode path.
    #[tokio::test]
    async fn all_visible_mask_matches_full_softmax() {
        let (device, queue) = gpu_or_skip!();
        let num_q_heads = 4;
        let num_kv_heads = 2;
        let q_dim = 16;
        let v_dim = 16;
        let kv_pool_len = 5;

        let q: Vec<f32> = (0..num_q_heads * q_dim)
            .map(|i| ((i as f32) * 0.05).sin() * 0.3)
            .collect();
        let k: Vec<f32> = (0..kv_pool_len * num_kv_heads * q_dim)
            .map(|i| ((i as f32) * 0.07).cos() * 0.3)
            .collect();
        let v: Vec<f32> = (0..kv_pool_len * num_kv_heads * v_dim)
            .map(|i| ((i as f32) * 0.03).sin() * 0.5)
            .collect();
        // A single causal range starting at 0 with scatter_position = kv_pool_len
        // - 1 makes query row 0 attend the whole pool — the all-visible last-row
        // decode anchor.
        let causal_start: u32 = 0;
        let scatter_position_value: u32 = (kv_pool_len - 1) as u32;
        let mask = expand_visibility(&[], causal_start, scatter_position_value, 1, kv_pool_len);

        let expected = cpu_masked_block_attention(
            &q,
            &k,
            &v,
            &mask,
            num_q_heads,
            num_kv_heads,
            q_dim,
            v_dim,
            1,
            kv_pool_len,
        );

        let gpu = MaskedBlockAttentionWebgpu::new(&device, num_q_heads, num_kv_heads, q_dim, v_dim);
        let q_buf = upload_f32(&device, &q);
        let k_buf = upload_f32(&device, &k);
        let v_buf = upload_f32(&device, &v);
        let visibility = encode_visibility(&[], causal_start);
        let mask_buf = upload_u32(&device, &visibility);
        let scatter_position_buf = upload_u32_uniform(&device, scatter_position_value);
        let out_buf = create_f32_buffer(&device, num_q_heads * v_dim);
        let sz = std::mem::size_of::<f32>() as u32;
        let q_view = BufferView::new_3d_tight(&q_buf, 1, num_q_heads as u32, q_dim as u32, sz);
        let k_view = BufferView::new_3d_tight(
            &k_buf,
            kv_pool_len as u32,
            num_kv_heads as u32,
            q_dim as u32,
            sz,
        );
        let v_view = BufferView::new_3d_tight(
            &v_buf,
            kv_pool_len as u32,
            num_kv_heads as u32,
            v_dim as u32,
            sz,
        );
        let out_view = BufferView::new_3d_tight(&out_buf, 1, num_q_heads as u32, v_dim as u32, sz);
        let runner = gpu.plan(
            &device,
            &queue,
            q_view,
            k_view,
            v_view,
            out_view,
            &mask_buf,
            &scatter_position_buf,
        );
        run_blocking_compute(&device, &queue, |cp| runner.forward(cp));
        let actual = download_f32(&device, &queue, &out_buf, num_q_heads * v_dim);

        assert_approx_eq(&actual, &expected, 1e-3);
    }
}
