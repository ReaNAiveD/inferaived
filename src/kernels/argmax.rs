use bytemuck;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BufferDescriptor, ComputePipeline, ComputePipelineDescriptor, Device,
    PipelineCompilationOptions, PipelineLayoutDescriptor, Queue,
};

use crate::buffer_view::BufferView;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ArgmaxParams {
    vocab_size: u32,
}

pub struct GpuArgmax {
    bind_group_layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

impl GpuArgmax {
    pub fn new(device: &Device) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("argmax/shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "wgsl-shaders/argmax.wgsl"
            ))),
        });
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("argmax/bind_group_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
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
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("argmax/pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("argmax/pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            bind_group_layout,
            pipeline,
        }
    }

    /// Bake bindings for one argmax dispatch.
    ///
    /// `logits` is a 1-D f32 view of length `vocab_size`. `token_out`
    /// is a 1-D u32 view of length 1 that will receive the winning
    /// token id.
    pub fn plan(
        &self,
        device: &Device,
        queue: &Queue,
        logits: BufferView<'_>,
        token_out: BufferView<'_>,
    ) -> GpuArgmaxRunner {
        debug_assert_eq!(logits.rank, 1, "argmax: logits must be 1-D");
        debug_assert_eq!(
            logits.elem_size, 4,
            "argmax: logits must be 4-byte (f32) elements, got elem_size={}",
            logits.elem_size,
        );
        debug_assert_eq!(token_out.rank, 1, "argmax: token_out must be 1-D");
        debug_assert_eq!(
            token_out.shape[0], 1,
            "argmax: token_out must be length 1, got shape={:?}",
            token_out.shape,
        );
        debug_assert_eq!(
            token_out.elem_size, 4,
            "argmax: token_out must be 4-byte (u32) elements, got elem_size={}",
            token_out.elem_size,
        );

        let uniform = ArgmaxParams {
            vocab_size: logits.shape[0],
        };
        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("argmax_runner/uniform_buffer"),
            size: std::mem::size_of::<ArgmaxParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&uniform));

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("argmax_runner/bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: logits.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: token_out.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        GpuArgmaxRunner {
            pipeline: self.pipeline.clone(),
            bind_group,
        }
    }
}

pub struct GpuArgmaxRunner {
    pipeline: ComputePipeline,
    bind_group: BindGroup,
}

impl GpuArgmaxRunner {
    pub fn forward(&self, cpass: &mut wgpu::ComputePass<'_>) {
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        // Single workgroup; the shader internally strides through the
        // whole vocab.
        cpass.dispatch_workgroups(1, 1, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_test_utils::*;
    use wgpu::util::{BufferInitDescriptor, DeviceExt};

    fn run_argmax(device: &wgpu::Device, queue: &wgpu::Queue, logits: &[f32]) -> u32 {
        let kernel = GpuArgmax::new(device);

        let logits_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("test/logits"),
            contents: bytemuck::cast_slice(logits),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let token_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test/token_out"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test/readback"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let runner = kernel.plan(
            device,
            queue,
            BufferView::new_1d(&logits_buf, 4, logits.len() as u32),
            BufferView::new_1d(&token_buf, 4, 1),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            runner.forward(&mut cpass);
        }
        encoder.copy_buffer_to_buffer(&token_buf, 0, &readback, 0, 4);
        let submission = queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        });
        rx.recv().unwrap().unwrap();
        let bytes = slice.get_mapped_range();
        let id = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        drop(bytes);
        readback.unmap();
        id
    }

    #[tokio::test]
    async fn argmax_picks_strict_max() {
        let (device, queue) = gpu_or_skip!();
        let logits = vec![1.0, 5.0, 3.0, 4.0, 2.0];
        let id = run_argmax(&device, &queue, &logits);
        assert_eq!(id, 1);
    }

    #[tokio::test]
    async fn argmax_handles_negative_logits() {
        let (device, queue) = gpu_or_skip!();
        let logits = vec![-10.0, -5.0, -3.0, -7.0];
        let id = run_argmax(&device, &queue, &logits);
        assert_eq!(id, 2);
    }

    #[tokio::test]
    async fn argmax_smallest_idx_wins_on_tie() {
        let (device, queue) = gpu_or_skip!();
        let logits = vec![1.0, 5.0, 5.0, 5.0, 2.0];
        let id = run_argmax(&device, &queue, &logits);
        assert_eq!(id, 1, "tie-break should favor smallest index");
    }

    #[tokio::test]
    async fn argmax_large_vocab_single_workgroup() {
        let (device, queue) = gpu_or_skip!();
        // Larger-than-workgroup vocab forces the stride loop and the
        // tree reduction to interact correctly.
        let n = 4096usize;
        let target_idx = 1337usize;
        let mut logits = vec![0.0f32; n];
        for (i, l) in logits.iter_mut().enumerate() {
            *l = (i as f32) * 0.001;
        }
        logits[target_idx] = 999.0;
        let id = run_argmax(&device, &queue, &logits);
        assert_eq!(id as usize, target_idx);
    }
}
