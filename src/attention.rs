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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_test_utils::*;

    /// CPU reference for causal grouped-query attention
    fn cpu_causal_gqa_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        num_q_heads: usize,
        num_kv_heads: usize,
        q_dim: usize,
        v_dim: usize,
        seq_len: usize,
    ) -> Vec<f32> {
        let q_token_stride = num_q_heads * q_dim;
        let k_token_stride = num_kv_heads * q_dim;
        let v_token_stride = num_kv_heads * v_dim;
        let output_token_stride = num_q_heads * v_dim;
        let num_q_per_kv = num_q_heads / num_kv_heads;
        let softmax_scale = 1.0 / (q_dim as f32).sqrt();

        let mut output = vec![0.0f32; seq_len * output_token_stride];

        for q_token in 0..seq_len {
            for q_head in 0..num_q_heads {
                let kv_head = q_head / num_q_per_kv;

                let mut scores = Vec::with_capacity(q_token + 1);
                for k_token in 0..=q_token {
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

    #[tokio::test]
    async fn test_causal_gqa_attention() {
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

        let expected = cpu_causal_gqa_attention(
            &q, &k, &v,
            num_q_heads, num_kv_heads, q_dim, v_dim, seq_len,
        );

        let gpu = CausalGqaNaiveAttentionWebgpu::new(&device, num_q_heads, num_kv_heads, q_dim, v_dim);
        let q_buf = upload_f32(&device, &q);
        let k_buf = upload_f32(&device, &k);
        let v_buf = upload_f32(&device, &v);
        let out_buf = create_f32_buffer(&device, seq_len * num_q_heads * v_dim);
        gpu.compute(&device, &queue, &q_buf, &k_buf, &v_buf, &out_buf, seq_len);
        let actual = download_f32(&device, &queue, &out_buf, seq_len * num_q_heads * v_dim);

        assert_approx_eq(&actual, &expected, 1e-3);
    }
}

