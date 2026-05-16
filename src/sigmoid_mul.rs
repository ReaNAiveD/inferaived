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

/// In-place fused output gate: `hidden[t, h, d] *= sigmoid(q_gate[t, h, head_dim + d])`.
pub struct SigmoidMulInplaceWebgpu {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    uniform_buffer: wgpu::Buffer,

    num_heads: usize,
    head_dim: usize,
}

impl SigmoidMulInplaceWebgpu {
    pub fn new(device: &wgpu::Device, num_heads: usize, head_dim: usize) -> Self {
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
            num_heads,
            head_dim,
        }
    }

    pub fn forward(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        hidden_buffer: &wgpu::Buffer,
        q_gate_combined_buffer: &wgpu::Buffer,
        num_rows: usize,
    ) {
        let head_dim = self.head_dim as u32;
        let uniform = SigmoidMulParams {
            hidden_offset: 0,
            hidden_token_stride: (self.num_heads * self.head_dim) as u32,
            hidden_head_stride: head_dim,
            // Gate half is at offset `head_dim` within each head; the
            // head stride is `2 * head_dim` because Q and gate are
            // interleaved per head.
            gate_offset: head_dim,
            gate_token_stride: (self.num_heads * self.head_dim * 2) as u32,
            gate_head_stride: (self.head_dim * 2) as u32,
            num_heads: self.num_heads as u32,
            head_dim,
            seq_len: num_rows as u32,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sigmoid_mul/bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: hidden_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: q_gate_combined_buffer.as_entire_binding(),
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
            compute_pass.dispatch_workgroups(num_rows as u32, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_test_utils::*;

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// CPU reference for the Q-gate-interleaved fused output gate:
    ///   `hidden[t, h, d] *= sigmoid(q_gate[t, h, head_dim + d])`
    fn cpu_q_gate_output_gate(
        hidden: &mut [f32],
        q_gate: &[f32],
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
    ) {
        let hidden_token_stride = num_heads * head_dim;
        let q_gate_token_stride = num_heads * head_dim * 2;
        let q_gate_head_stride = head_dim * 2;
        for t in 0..seq_len {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    let h_idx = t * hidden_token_stride + h * head_dim + d;
                    // Gate half lives at offset `head_dim` within each head.
                    let g_idx = t * q_gate_token_stride + h * q_gate_head_stride + head_dim + d;
                    hidden[h_idx] *= sigmoid(q_gate[g_idx]);
                }
            }
        }
    }

    #[tokio::test]
    async fn test_sigmoid_mul_forward() {
        let (device, queue) = gpu_or_skip!();
        let seq_len = 3;
        let num_heads = 4;
        let head_dim = 8;
        let hidden_size = num_heads * head_dim;
        let q_gate_size = num_heads * head_dim * 2;
        let hidden: Vec<f32> = (0..seq_len * hidden_size)
            .map(|i| (i as f32) * 0.1 - 1.0)
            .collect();
        let q_gate: Vec<f32> = (0..seq_len * q_gate_size)
            .map(|i| (i as f32) * 0.05 - 0.7)
            .collect();

        let mut expected = hidden.clone();
        cpu_q_gate_output_gate(&mut expected, &q_gate, num_heads, head_dim, seq_len);

        let gpu = SigmoidMulInplaceWebgpu::new(&device, num_heads, head_dim);
        let h_buf = upload_f32(&device, &hidden);
        let g_buf = upload_f32(&device, &q_gate);
        gpu.forward(&device, &queue, &h_buf, &g_buf, seq_len);
        let actual = download_f32(&device, &queue, &h_buf, seq_len * hidden_size);

        assert_approx_eq(&actual, &expected, 1e-5);
    }
}
