/// Mirrors `Params` in `wgsl-shaders/slice_copy.wgsl`.
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

/// Copy a `[seq_len, num_heads, head_dim]` view from one buffer into another,
/// honoring per-axis offsets and strides on both sides. Useful for pulling the
/// Q half out of q_proj output (interleaved [Q | gate] per head) into a tight
/// buffer.
pub struct SliceCopyWebgpu {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    uniform_buffer: wgpu::Buffer,
}

impl SliceCopyWebgpu {
    pub fn new(device: &wgpu::Device) -> Self {
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
        }
    }

    pub fn compute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src_buffer: &wgpu::Buffer,
        dst_buffer: &wgpu::Buffer,
        src_offset: usize,
        src_token_stride: usize,
        src_head_stride: usize,
        dst_offset: usize,
        dst_token_stride: usize,
        dst_head_stride: usize,
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
    ) {
        let uniform = SliceCopyParams {
            src_offset: src_offset as u32,
            src_token_stride: src_token_stride as u32,
            src_head_stride: src_head_stride as u32,
            dst_offset: dst_offset as u32,
            dst_token_stride: dst_token_stride as u32,
            dst_head_stride: dst_head_stride as u32,
            num_heads: num_heads as u32,
            head_dim: head_dim as u32,
            seq_len: seq_len as u32,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("slice_copy/bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dst_buffer.as_entire_binding(),
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
            compute_pass.dispatch_workgroups(seq_len as u32, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_test_utils::*;

    /// CPU reference: copy [seq_len, num_heads, head_dim] with arbitrary strides
    fn cpu_slice_copy(
        src: &[f32],
        dst: &mut [f32],
        src_offset: usize,
        src_token_stride: usize,
        src_head_stride: usize,
        dst_offset: usize,
        dst_token_stride: usize,
        dst_head_stride: usize,
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
    ) {
        for t in 0..seq_len {
            for h in 0..num_heads {
                for i in 0..head_dim {
                    let s = src_offset + t * src_token_stride + h * src_head_stride + i;
                    let d = dst_offset + t * dst_token_stride + h * dst_head_stride + i;
                    dst[d] = src[s];
                }
            }
        }
    }

    #[tokio::test]
    async fn test_slice_copy() {
        let (device, queue) = gpu_or_skip!();
        let seq_len = 2;
        let num_heads = 2;
        let head_dim = 8;
        let src_token_stride = 48;
        let src_head_stride = 16;
        let dst_token_stride = num_heads * head_dim;
        let dst_head_stride = head_dim;
        let src_offset = 4;
        let dst_offset = 0;

        let src_size = src_offset + seq_len * src_token_stride;
        let dst_size = seq_len * dst_token_stride;

        let src: Vec<f32> = (0..src_size).map(|i| (i as f32) * 0.1).collect();
        let mut expected_dst = vec![0.0f32; dst_size];
        cpu_slice_copy(
            &src, &mut expected_dst,
            src_offset, src_token_stride, src_head_stride,
            dst_offset, dst_token_stride, dst_head_stride,
            num_heads, head_dim, seq_len,
        );

        let gpu = SliceCopyWebgpu::new(&device);
        let s_buf = upload_f32(&device, &src);
        let d_buf = create_f32_buffer(&device, dst_size);
        gpu.compute(
            &device, &queue, &s_buf, &d_buf,
            src_offset, src_token_stride, src_head_stride,
            dst_offset, dst_token_stride, dst_head_stride,
            num_heads, head_dim, seq_len,
        );
        let actual = download_f32(&device, &queue, &d_buf, dst_size);

        assert_approx_eq(&actual, &expected_dst, 1e-6);
    }
}
