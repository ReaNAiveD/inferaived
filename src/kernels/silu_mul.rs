use crate::buffer_view::BufferView;

#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct SiluMulParams {
    hidden_token_stride: u32,
    gate_token_stride: u32,

    hidden_size: u32,
    seq_len: u32,
}

pub struct SiluMulInplaceWebgpu {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    uniform_buffer: wgpu::Buffer,

    vec_dim: usize,
}

impl SiluMulInplaceWebgpu {
    pub fn new(device: &wgpu::Device, vec_dim: usize) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("silu_mul/shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("wgsl-shaders/silu_mul.wgsl").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("silu_mul/bind_group_layout"),
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
            label: Some("silu_mul/pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("silu_mul/pipeline"),
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
            label: Some("silu_mul/uniform_buffer"),
            size: std::mem::size_of::<SiluMulParams>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            bind_group_layout,
            pipeline,
            uniform_buffer,
            vec_dim,
        }
    }

    /// In-place: `hidden[t, i] *= silu(gate[t, i])`.
    pub fn forward(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        hidden: BufferView<'_>,
        gate: BufferView<'_>,
    ) {
        debug_assert_eq!(
            hidden.shape[0], gate.shape[0],
            "silu_mul: outer dim mismatch (hidden={}, gate={})",
            hidden.shape[0], gate.shape[0],
        );
        let num_rows = hidden.shape[0];
        let uniform = SiluMulParams {
            hidden_token_stride: hidden.stride[0],
            gate_token_stride: gate.stride[0],
            hidden_size: self.vec_dim as u32,
            seq_len: num_rows,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("silu_mul/bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: hidden.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gate.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("silu_mul/command_encoder"),
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("silu_mul/compute_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(num_rows, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_test_utils::*;

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    /// CPU: hidden[t,i] *= silu(gate[t,i])
    fn cpu_silu_mul(hidden: &mut [f32], gate: &[f32], hidden_size: usize, seq_len: usize) {
        for t in 0..seq_len {
            for i in 0..hidden_size {
                let idx = t * hidden_size + i;
                hidden[idx] *= silu(gate[idx]);
            }
        }
    }

    #[tokio::test]
    async fn test_silu_mul() {
        let (device, queue) = gpu_or_skip!();
        let seq_len = 3;
        let hidden_size = 32;
        let hidden: Vec<f32> = (0..seq_len * hidden_size)
            .map(|i| (i as f32) * 0.05 - 1.0)
            .collect();
        let gate: Vec<f32> = (0..seq_len * hidden_size)
            .map(|i| (i as f32) * -0.03 + 0.5)
            .collect();

        let mut expected = hidden.clone();
        cpu_silu_mul(&mut expected, &gate, hidden_size, seq_len);

        let gpu = SiluMulInplaceWebgpu::new(&device, hidden_size);
        let h_buf = upload_f32(&device, &hidden);
        let g_buf = upload_f32(&device, &gate);
        let elem_size = std::mem::size_of::<f32>() as u32;
        let h_view =
            BufferView::new_2d_tight(&h_buf, seq_len as u32, hidden_size as u32, elem_size);
        let g_view =
            BufferView::new_2d_tight(&g_buf, seq_len as u32, hidden_size as u32, elem_size);
        gpu.forward(&device, &queue, h_view, g_view);
        let actual = download_f32(&device, &queue, &h_buf, seq_len * hidden_size);

        assert_approx_eq(&actual, &expected, 1e-5);
    }
}
