#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ElementwiseAddParams {
    hidden_offset: u32,
    hidden_token_stride: u32,
    addend_offset: u32,
    addend_token_stride: u32,

    hidden_size: u32,
    seq_len: u32,
}

pub struct ElementwiseAddInplaceWebgpu {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    uniform_buffer: wgpu::Buffer,

    vec_dim: usize,
}

impl ElementwiseAddInplaceWebgpu {
    pub fn new(device: &wgpu::Device, vec_dim: usize) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("elementwise_add/shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("wgsl-shaders/elementwise_add.wgsl").into(),
            ),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("elementwise_add/bind_group_layout"),
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
            label: Some("elementwise_add/pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("elementwise_add/pipeline"),
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
            label: Some("elementwise_add/uniform_buffer"),
            size: std::mem::size_of::<ElementwiseAddParams>() as u64,
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

    /// Add `num_rows` rows of `other_buffer` (starting at row
    /// `other_start_row`) onto `src_buffer` (starting at row
    /// `src_start_row`), in place.
    pub fn forward(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src_buffer: &wgpu::Buffer,
        other_buffer: &wgpu::Buffer,
        src_start_row: usize,
        other_start_row: usize,
        num_rows: usize,
    ) {
        let uniform = ElementwiseAddParams {
            hidden_offset: (src_start_row * self.vec_dim) as u32,
            hidden_token_stride: self.vec_dim as u32,
            addend_offset: (other_start_row * self.vec_dim) as u32,
            addend_token_stride: self.vec_dim as u32,
            hidden_size: self.vec_dim as u32,
            seq_len: num_rows as u32,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("elementwise_add/bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: other_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("elementwise_add/command_encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("elementwise_add/compute_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(num_rows as u32, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_test_utils::*;

    /// CPU reference: hidden[t,i] += addend[t,i]
    fn cpu_elementwise_add(hidden: &mut [f32], addend: &[f32], hidden_size: usize, seq_len: usize) {
        for t in 0..seq_len {
            for i in 0..hidden_size {
                hidden[t * hidden_size + i] += addend[t * hidden_size + i];
            }
        }
    }

    #[tokio::test]
    async fn test_elementwise_add() {
        let (device, queue) = gpu_or_skip!();
        let seq_len = 3;
        let hidden_size = 16;
        let hidden: Vec<f32> = (0..seq_len * hidden_size)
            .map(|i| (i as f32) * 0.1)
            .collect();
        let addend: Vec<f32> = (0..seq_len * hidden_size)
            .map(|i| (i as f32) * -0.05 + 1.0)
            .collect();

        let mut expected = hidden.clone();
        cpu_elementwise_add(&mut expected, &addend, hidden_size, seq_len);

        let gpu = ElementwiseAddInplaceWebgpu::new(&device, hidden_size);
        let h_buf = upload_f32(&device, &hidden);
        let a_buf = upload_f32(&device, &addend);
        gpu.forward(&device, &queue, &h_buf, &a_buf, 0, 0, seq_len);
        let actual = download_f32(&device, &queue, &h_buf, seq_len * hidden_size);

        assert_approx_eq(&actual, &expected, 1e-5);
    }
}
