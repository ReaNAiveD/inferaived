#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct SliceCopyParams {
    src_offset: u32,
    src_token_stride: u32,
    src_head_stride: u32,

    dst_offset: u32,
    dst_token_stride: u32,
    dst_head_stride: u32,

    num_heads: u32,
    head_dim: u32,
    seq_len: u32,
}

/// Extract the Q half from a fused `[seq, num_heads, 2 * head_dim]`
/// Q+gate output (the q_proj output in Qwen3-Next / Llama-style self
/// attention) into a tight `[seq, num_heads, head_dim]` Q buffer.
pub struct SliceCopyWebgpu {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    uniform_buffer: wgpu::Buffer,

    num_heads: usize,
    head_dim: usize,
}

impl SliceCopyWebgpu {
    pub fn new(device: &wgpu::Device, num_heads: usize, head_dim: usize) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("slice_copy/shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("wgsl-shaders/slice_copy.wgsl").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("slice_copy/bind_group_layout"),
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
            label: Some("slice_copy/pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("slice_copy/pipeline"),
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
            label: Some("slice_copy/uniform_buffer"),
            size: std::mem::size_of::<SliceCopyParams>() as wgpu::BufferAddress,
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
        q_gate_combined_buffer: &wgpu::Buffer,
        q_dst_buffer: &wgpu::Buffer,
        num_rows: usize,
    ) {
        let uniform = SliceCopyParams {
            src_offset: 0,
            src_token_stride: (self.num_heads * self.head_dim * 2) as u32,
            src_head_stride: (self.head_dim * 2) as u32,
            dst_offset: 0,
            dst_token_stride: (self.num_heads * self.head_dim) as u32,
            dst_head_stride: self.head_dim as u32,
            num_heads: self.num_heads as u32,
            head_dim: self.head_dim as u32,
            seq_len: num_rows as u32,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("slice_copy/bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q_gate_combined_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: q_dst_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("slice_copy/command_encoder"),
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("slice_copy/compute_pass"),
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

    /// CPU reference: copy the Q half out of `[seq, num_heads, 2 *
    /// head_dim]` into tight `[seq, num_heads, head_dim]`.
    fn cpu_q_extract(
        q_gate: &[f32],
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
    ) -> Vec<f32> {
        let q_gate_token_stride = num_heads * head_dim * 2;
        let q_gate_head_stride = head_dim * 2;
        let mut q = vec![0.0f32; seq_len * num_heads * head_dim];
        for t in 0..seq_len {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    let src_idx = t * q_gate_token_stride + h * q_gate_head_stride + d;
                    let dst_idx = t * num_heads * head_dim + h * head_dim + d;
                    q[dst_idx] = q_gate[src_idx];
                }
            }
        }
        q
    }

    #[tokio::test]
    async fn test_slice_copy_forward() {
        let (device, queue) = gpu_or_skip!();
        let seq_len = 3;
        let num_heads = 4;
        let head_dim = 8;
        let q_gate_size = num_heads * head_dim * 2;
        let q_gate: Vec<f32> = (0..seq_len * q_gate_size)
            .map(|i| (i as f32) * 0.07 - 1.5)
            .collect();

        let expected = cpu_q_extract(&q_gate, num_heads, head_dim, seq_len);

        let gpu = SliceCopyWebgpu::new(&device, num_heads, head_dim);
        let g_buf = upload_f32(&device, &q_gate);
        let q_buf = create_f32_buffer(&device, seq_len * num_heads * head_dim);
        gpu.forward(&device, &queue, &g_buf, &q_buf, seq_len);
        let actual = download_f32(&device, &queue, &q_buf, seq_len * num_heads * head_dim);

        assert_approx_eq(&actual, &expected, 0.0);
    }
}
