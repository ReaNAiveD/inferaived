use bytemuck;
use safetensors::tensor::TensorView;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BufferDescriptor, ComputePipeline, ComputePipelineDescriptor, Device,
    PipelineCompilationOptions, PipelineLayoutDescriptor, Queue,
};

use crate::buffer_view::BufferView;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GetRowsParams {
    source_offset: u32, // in u32 elements (always 0; BufferBinding handles offsets)
    source_row_stride: u32, // in u32 elements (= hidden_size / 2 for tight bf16)
    indices_offset: u32, // in i32 elements (always 0; BufferBinding handles offsets)
    output_offset: u32, // in f32 elements (always 0; BufferBinding handles offsets)
    output_row_stride: u32, // in f32 elements

    hidden_size: u32,
    num_tokens: u32,
}

/// GPU row-gather over a packed-bf16 `[vocab_size, hidden_size]` source table.
pub struct GetRows {
    hidden_size: usize,
    bind_group_layout: BindGroupLayout,
    pipeline: ComputePipeline,
    source_buffer: wgpu::Buffer,
}

impl GetRows {
    /// Upload the source table (bf16, `[vocab_size, hidden_size]`,
    /// little-endian) and compile the pipeline.
    pub fn new<'data>(device: &Device, queue: &Queue, source: TensorView<'data>) -> Self {
        debug_assert_eq!(
            source.shape().len(),
            2,
            "GetRows source must be 2-D [vocab_size, hidden_size], got shape {:?}",
            source.shape(),
        );
        let hidden_size = source.shape()[1];
        debug_assert_eq!(
            hidden_size % 2,
            0,
            "GetRows requires hidden_size to be even (bf16 packing into u32), got {}",
            hidden_size,
        );

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("get_rows/shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "wgsl-shaders/get_rows.wgsl"
            ))),
        });
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("get_rows/bind_group_layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
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
            label: Some("get_rows/pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("get_rows/pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        let source_bytes = source.data().len() as wgpu::BufferAddress;
        let align_mask = wgpu::COPY_BUFFER_ALIGNMENT - 1;
        let padded_size =
            ((source_bytes + align_mask) & !align_mask).max(wgpu::COPY_BUFFER_ALIGNMENT);
        let source_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("get_rows/source_buffer"),
            size: padded_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&source_buffer, 0, source.data());
        Self {
            hidden_size,
            bind_group_layout,
            pipeline,
            source_buffer,
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Bake the per-call bindings into a [`GetRowsRunner`] for repeated
    /// dispatch into a caller-owned compute pass.
    pub fn plan(
        &self,
        device: &Device,
        queue: &Queue,
        tokens: BufferView<'_>,
        dst: BufferView<'_>,
    ) -> GetRowsRunner {
        debug_assert_eq!(tokens.rank, 1, "get_rows: tokens must be 1-D");
        debug_assert_eq!(dst.rank, 2, "get_rows: dst must be 2-D");
        debug_assert_eq!(
            tokens.shape[0], dst.shape[0],
            "get_rows: tokens len ({}) must match dst rows ({})",
            tokens.shape[0], dst.shape[0],
        );
        debug_assert_eq!(
            dst.shape[1] as usize, self.hidden_size,
            "get_rows: dst inner dim {} != hidden_size {}",
            dst.shape[1], self.hidden_size,
        );
        debug_assert_eq!(
            tokens.elem_size, 4,
            "get_rows: tokens must be 4-byte (u32/i32) elements, got elem_size={}",
            tokens.elem_size,
        );
        debug_assert_eq!(
            dst.elem_size, 4,
            "get_rows: dst must be 4-byte (f32) elements, got elem_size={}",
            dst.elem_size,
        );

        let num_tokens = tokens.shape[0];
        let uniform = GetRowsParams {
            source_offset: 0,
            source_row_stride: (self.hidden_size / 2) as u32,
            indices_offset: 0,
            output_offset: 0,
            output_row_stride: dst.stride[0],
            hidden_size: self.hidden_size as u32,
            num_tokens,
        };
        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("get_rows_runner/uniform_buffer"),
            size: std::mem::size_of::<GetRowsParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&uniform));

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("get_rows_runner/bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.source_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tokens.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dst.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        GetRowsRunner {
            pipeline: self.pipeline.clone(),
            bind_group,
            num_tokens,
        }
    }
}

pub struct GetRowsRunner {
    pipeline: ComputePipeline,
    bind_group: BindGroup,
    num_tokens: u32,
}

impl GetRowsRunner {
    pub fn forward(&self, cpass: &mut wgpu::ComputePass<'_>) {
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        let workgroup_count = (self.num_tokens + 255) / 256;
        cpass.dispatch_workgroups(workgroup_count, 1, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_test_utils::*;
    use wgpu::util::{BufferInitDescriptor, DeviceExt};

    /// CPU reference: for each token index, gather the row from the bf16
    /// source table and convert to f32.
    fn cpu_get_rows(source_packed: &[u32], indices: &[u32], hidden_size: usize) -> Vec<f32> {
        let row_stride_u32 = hidden_size / 2;
        let mut out = Vec::with_capacity(indices.len() * hidden_size);
        for &idx in indices {
            let base = (idx as usize) * row_stride_u32;
            for i in 0..hidden_size {
                out.push(unpack_bf16(&source_packed, base * 2 + i));
            }
        }
        out
    }

    #[tokio::test]
    async fn test_get_rows_runner() {
        let (device, queue) = gpu_or_skip!();
        let vocab_size = 8;
        let hidden_size = 16;
        let source_f32: Vec<f32> = (0..vocab_size * hidden_size)
            .map(|i| (i as f32) * 0.1 - 3.0)
            .collect();
        let source_bf16_bytes: Vec<u8> = source_f32
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect();

        let indices: Vec<u32> = vec![0, 3, 7, 1];

        let source_packed = pack_f32_to_bf16_u32(&source_f32);
        let expected = cpu_get_rows(&source_packed, &indices, hidden_size);

        let tv = TensorView::new(
            safetensors::Dtype::BF16,
            vec![vocab_size, hidden_size],
            &source_bf16_bytes,
        )
        .unwrap();
        let gpu = GetRows::new(&device, &queue, tv);

        // Caller owns the persistent indices and output buffers; the runner
        // bakes views over them.
        let indices_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("test/indices_buf"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buf = create_f32_buffer(&device, indices.len() * hidden_size);

        let tokens_view = BufferView::new_1d(&indices_buf, 4, indices.len() as u32);
        let dst_view =
            BufferView::new_2d_tight(&out_buf, indices.len() as u32, hidden_size as u32, 4);
        let runner = gpu.plan(&device, &queue, tokens_view, dst_view);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test/encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("test/cpass"),
                timestamp_writes: None,
            });
            runner.forward(&mut cpass);
        }
        queue.submit(Some(encoder.finish()));

        let actual = download_f32(&device, &queue, &out_buf, indices.len() * hidden_size);
        assert_approx_eq(&actual, &expected, 1e-5);
    }

    #[tokio::test]
    async fn test_get_rows_runner_reusable_across_dispatches() {
        // Same runner dispatched twice in one compute pass should produce
        // the same output (indices and source don't change between
        // dispatches).
        let (device, queue) = gpu_or_skip!();
        let vocab_size = 4;
        let hidden_size = 8;
        let source_f32: Vec<f32> = (0..vocab_size * hidden_size).map(|i| i as f32).collect();
        let source_bf16_bytes: Vec<u8> = source_f32
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect();
        let indices: Vec<u32> = vec![2, 0];

        let tv = TensorView::new(
            safetensors::Dtype::BF16,
            vec![vocab_size, hidden_size],
            &source_bf16_bytes,
        )
        .unwrap();
        let gpu = GetRows::new(&device, &queue, tv);

        let indices_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("test/indices_buf"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_buf = create_f32_buffer(&device, indices.len() * hidden_size);

        let runner = gpu.plan(
            &device,
            &queue,
            BufferView::new_1d(&indices_buf, 4, indices.len() as u32),
            BufferView::new_2d_tight(&out_buf, indices.len() as u32, hidden_size as u32, 4),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            runner.forward(&mut cpass);
            runner.forward(&mut cpass);
        }
        queue.submit(Some(encoder.finish()));

        let actual = download_f32(&device, &queue, &out_buf, indices.len() * hidden_size);
        // Row 2 of source is [16, 17, 18, ..., 23]; row 0 is [0..8].
        let expected = vec![
            16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
        ];
        assert_approx_eq(&actual, &expected, 1e-3);
    }
}
