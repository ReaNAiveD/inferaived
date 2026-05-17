use crate::buffer_view::BufferView;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RopeParams {
    q_token_stride: u32,
    q_head_stride: u32,
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
            bind_group_layouts: &[Some(&bind_group_layout)],
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
    pub fn forward(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        q: BufferView<'_>,
        k: BufferView<'_>,
        position_offset: usize,
    ) {
        debug_assert_eq!(
            q.rank, 3,
            "RoPE: q must be rank-3 [seq, num_heads, head_dim]"
        );
        debug_assert_eq!(
            k.rank, 3,
            "RoPE: k must be rank-3 [seq, num_heads, head_dim]"
        );
        debug_assert_eq!(
            q.shape[0], k.shape[0],
            "RoPE: q.shape[0] ({}) must equal k.shape[0] ({})",
            q.shape[0], k.shape[0],
        );
        debug_assert_eq!(
            q.shape[1] as usize, self.num_q_heads,
            "RoPE: q.shape[1] ({}) must equal num_q_heads ({})",
            q.shape[1], self.num_q_heads,
        );
        debug_assert_eq!(
            k.shape[1] as usize, self.num_k_heads,
            "RoPE: k.shape[1] ({}) must equal num_k_heads ({})",
            k.shape[1], self.num_k_heads,
        );
        debug_assert_eq!(
            q.shape[2] as usize, self.head_dim,
            "RoPE: q.shape[2] ({}) must equal head_dim ({})",
            q.shape[2], self.head_dim,
        );
        debug_assert_eq!(
            k.shape[2] as usize, self.head_dim,
            "RoPE: k.shape[2] ({}) must equal head_dim ({})",
            k.shape[2], self.head_dim,
        );

        let num_new_tokens = q.shape[0];
        let uniform_data = RopeParams {
            q_token_stride: q.stride[0],
            q_head_stride: q.stride[1],
            k_token_stride: k.stride[0],
            k_head_stride: k.stride[1],
            num_q_heads: self.num_q_heads as u32,
            num_k_heads: self.num_k_heads as u32,
            seq_len: num_new_tokens,
            num_rotated_dims: self.num_rotated_dims as u32,
            theta_scale: self.theta_scale,
            position_offset: position_offset as u32,
        };
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[uniform_data]),
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rope/bind_group"),
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
                ((self.num_rotated_dims / 2) * max_heads * num_new_tokens as usize) as u32;
            let workgroup_count =
                (total_invocations + Self::WORKGROUP_SIZE - 1) / Self::WORKGROUP_SIZE;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_test_utils::*;

    /// CPU reference for half-split RoPE
    fn cpu_rope(
        q: &mut [f32],
        k: &mut [f32],
        num_q_heads: usize,
        num_k_heads: usize,
        head_dim: usize,
        seq_len: usize,
        theta_scale: f32,
        num_rotated_dims: usize,
        position_offset: usize,
    ) {
        let pair_offset = num_rotated_dims / 2;
        let q_token_stride = num_q_heads * head_dim;
        let k_token_stride = num_k_heads * head_dim;

        for token in 0..seq_len {
            for head in 0..std::cmp::max(num_q_heads, num_k_heads) {
                for pair in 0..pair_offset {
                    let pos = token + position_offset;
                    let theta = pos as f32 * theta_scale.powf(pair as f32);
                    let cos_t = theta.cos();
                    let sin_t = theta.sin();

                    if head < num_q_heads {
                        let a_idx = token * q_token_stride + head * head_dim + pair;
                        let b_idx = token * q_token_stride + head * head_dim + pair + pair_offset;
                        let a = q[a_idx];
                        let b = q[b_idx];
                        q[a_idx] = a * cos_t - b * sin_t;
                        q[b_idx] = a * sin_t + b * cos_t;
                    }
                    if head < num_k_heads {
                        let a_idx = token * k_token_stride + head * head_dim + pair;
                        let b_idx = token * k_token_stride + head * head_dim + pair + pair_offset;
                        let a = k[a_idx];
                        let b = k[b_idx];
                        k[a_idx] = a * cos_t - b * sin_t;
                        k[b_idx] = a * sin_t + b * cos_t;
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn test_rope() {
        let (device, queue) = gpu_or_skip!();
        let seq_len = 3;
        let num_q_heads = 4;
        let num_k_heads = 2;
        let head_dim = 16;
        let rope_theta: f32 = 10000.0;
        let partial_rotary_factor: f32 = 1.0;
        let num_rotated_dims = (head_dim as f32 * partial_rotary_factor).floor() as usize;
        let theta_scale = rope_theta.powf(-2.0 / num_rotated_dims as f32);

        let q: Vec<f32> = (0..seq_len * num_q_heads * head_dim)
            .map(|i| ((i as f32) * 0.1).sin())
            .collect();
        let k: Vec<f32> = (0..seq_len * num_k_heads * head_dim)
            .map(|i| ((i as f32) * 0.07).cos())
            .collect();

        let mut expected_q = q.clone();
        let mut expected_k = k.clone();
        cpu_rope(
            &mut expected_q,
            &mut expected_k,
            num_q_heads,
            num_k_heads,
            head_dim,
            seq_len,
            theta_scale,
            num_rotated_dims,
            0,
        );

        let gpu = RopeInplaceWebgpu::new(
            &device,
            num_q_heads,
            num_k_heads,
            head_dim,
            rope_theta,
            partial_rotary_factor,
        );
        let q_buf = upload_f32(&device, &q);
        let k_buf = upload_f32(&device, &k);
        // Rank-3 views [seq, num_heads, head_dim] tight-packed. The
        // kernel reads per-token and per-head strides from these views;
        // no extra stride parameters needed.
        let q_view = BufferView::new_3d_tight(
            &q_buf,
            seq_len as u32,
            num_q_heads as u32,
            head_dim as u32,
            std::mem::size_of::<f32>() as u32,
        );
        let k_view = BufferView::new_3d_tight(
            &k_buf,
            seq_len as u32,
            num_k_heads as u32,
            head_dim as u32,
            std::mem::size_of::<f32>() as u32,
        );
        gpu.forward(&device, &queue, q_view, k_view, 0);
        let actual_q = download_f32(&device, &queue, &q_buf, seq_len * num_q_heads * head_dim);
        let actual_k = download_f32(&device, &queue, &k_buf, seq_len * num_k_heads * head_dim);

        assert_approx_eq(&actual_q, &expected_q, 1e-4);
        assert_approx_eq(&actual_k, &expected_k, 1e-4);
    }
}
