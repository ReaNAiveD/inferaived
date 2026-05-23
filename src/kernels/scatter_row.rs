use crate::buffer_view::BufferView;

#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct ScatterRowParams {
    row_width: u32,
    num_rows: u32,
}

pub struct ScatterRowWebgpu {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    row_width: usize,
}

impl ScatterRowWebgpu {
    const WORKGROUP_SIZE: u32 = 128;

    pub fn new(device: &wgpu::Device, row_width: usize) -> Self {
        debug_assert!(row_width >= 1, "row_width must be >= 1");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scatter_row/shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "wgsl-shaders/scatter_row.wgsl"
            ))),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scatter_row/bind_group_layout"),
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
                wgpu::BindGroupLayoutEntry {
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
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scatter_row/pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("scatter_row/pipeline"),
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
            row_width,
        }
    }

    /// Bake the per-buffer bindings into a [`ScatterRowWebgpuRunner`]
    /// for repeated dispatch into a caller-owned compute pass.
    pub fn plan(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src: BufferView<'_>,
        dst: BufferView<'_>,
        position_buffer: &wgpu::Buffer,
    ) -> ScatterRowWebgpuRunner {
        let sz = std::mem::size_of::<f32>() as u32;
        debug_assert_eq!(src.elem_size, sz, "scatter_row: src must be f32");
        debug_assert_eq!(dst.elem_size, sz, "scatter_row: dst must be f32");
        let num_rows = src.shape[0];
        debug_assert!(num_rows >= 1, "scatter_row: src.shape[0] must be >= 1");
        let src_elems = num_rows as usize * self.row_width;
        debug_assert_eq!(
            src.total_byte_size() as usize,
            src_elems * std::mem::size_of::<f32>(),
            "scatter_row: src byte size ({}) must equal num_rows*row_width*4 ({})",
            src.total_byte_size(),
            src_elems * std::mem::size_of::<f32>(),
        );
        debug_assert!(
            dst.total_byte_size() as usize >= src_elems * std::mem::size_of::<f32>(),
            "scatter_row: dst byte size ({}) must hold at least num_rows*row_width*4 ({})",
            dst.total_byte_size(),
            src_elems * std::mem::size_of::<f32>(),
        );
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scatter_row_runner/uniform_buffer"),
            size: std::mem::size_of::<ScatterRowParams>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform = ScatterRowParams {
            row_width: self.row_width as u32,
            num_rows,
        };
        queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&uniform));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scatter_row_runner/bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dst.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: position_buffer.as_entire_binding(),
                },
            ],
        });
        let workgroup_count = (src_elems as u32 + Self::WORKGROUP_SIZE - 1) / Self::WORKGROUP_SIZE;
        ScatterRowWebgpuRunner {
            pipeline: self.pipeline.clone(),
            bind_group,
            workgroup_count,
        }
    }
}

pub struct ScatterRowWebgpuRunner {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    workgroup_count: u32,
}

impl ScatterRowWebgpuRunner {
    pub fn forward(&self, cpass: &mut wgpu::ComputePass<'_>) {
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.dispatch_workgroups(self.workgroup_count, 1, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_test_utils::*;

    /// 4-byte position uniform buffer initialized to `pos`.
    fn make_position_buffer(device: &wgpu::Device, queue: &wgpu::Queue, pos: u32) -> wgpu::Buffer {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scatter_row_test/position_buffer"),
            size: std::mem::size_of::<u32>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, bytemuck::bytes_of(&pos));
        buffer
    }

    #[tokio::test]
    async fn test_scatter_row_writes_at_position() {
        let (device, queue) = gpu_or_skip!();
        let row_width = 64usize;
        let num_rows = 8usize;
        let pos = 3u32;

        // src = [1, 2, 3, ..., row_width]
        let src: Vec<f32> = (0..row_width).map(|i| (i + 1) as f32).collect();
        // dst initialized to all 0.5 (distinguishes "untouched" from "zeroed
        // by scatter" — scatter writes from src, so target row gets src).
        let dst: Vec<f32> = vec![0.5; num_rows * row_width];

        let gpu = ScatterRowWebgpu::new(&device, row_width);
        let src_buf = upload_f32(&device, &src);
        let dst_buf = upload_f32(&device, &dst);
        let sz = std::mem::size_of::<f32>() as u32;
        // src is a 1-row source; shape[0] = 1 tells scatter to copy a
        // single row.
        let src_view = BufferView::new_2d_tight(&src_buf, 1, row_width as u32, sz);
        let dst_view = BufferView::new_2d_tight(&dst_buf, num_rows as u32, row_width as u32, sz);
        let position_buffer = make_position_buffer(&device, &queue, pos);

        let runner = gpu.plan(&device, &queue, src_view, dst_view, &position_buffer);
        run_blocking_compute(&device, &queue, |cp| runner.forward(cp));

        let actual = download_f32(&device, &queue, &dst_buf, num_rows * row_width);

        // Row `pos` should equal src; all other rows should still be 0.5.
        for r in 0..num_rows {
            for c in 0..row_width {
                let got = actual[r * row_width + c];
                let want = if r == pos as usize {
                    (c + 1) as f32
                } else {
                    0.5
                };
                assert!(
                    (got - want).abs() < 1e-6,
                    "row {} col {}: got {}, want {}",
                    r,
                    c,
                    got,
                    want,
                );
            }
        }
    }

    #[tokio::test]
    async fn test_scatter_row_runner_two_steps() {
        // Verify the runner is reusable across steps: build once, scatter
        // twice with different position_buffer contents.
        let (device, queue) = gpu_or_skip!();
        let row_width = 16usize;
        let num_rows = 4usize;

        let src1: Vec<f32> = (0..row_width).map(|i| (i + 1) as f32).collect();
        let src2: Vec<f32> = (0..row_width).map(|i| -((i + 100) as f32)).collect();
        let dst_init: Vec<f32> = vec![0.0; num_rows * row_width];

        let gpu = ScatterRowWebgpu::new(&device, row_width);
        let dst_buf = upload_f32(&device, &dst_init);
        let position_buffer = make_position_buffer(&device, &queue, 0);

        let sz = std::mem::size_of::<f32>() as u32;
        let dst_view = BufferView::new_2d_tight(&dst_buf, num_rows as u32, row_width as u32, sz);

        // Step 1: scatter src1 to row 1.
        let src1_buf = upload_f32(&device, &src1);
        let src1_view = BufferView::new_2d_tight(&src1_buf, 1, row_width as u32, sz);
        let runner1 = gpu.plan(&device, &queue, src1_view, dst_view, &position_buffer);
        queue.write_buffer(&position_buffer, 0, bytemuck::bytes_of(&1u32));
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut cpass = encoder.begin_compute_pass(&Default::default());
            runner1.forward(&mut cpass);
        }
        queue.submit(Some(encoder.finish()));

        // Step 2: with src1_view / runner1 dropped, build runner2 around
        // src2 and scatter at row 2 (note: typical real-world usage builds
        // ONE runner bound to a fixed scratch and just rewrites that
        // scratch's contents per step — see Phase 2 of the refactor).
        let src2_buf = upload_f32(&device, &src2);
        let src2_view = BufferView::new_2d_tight(&src2_buf, 1, row_width as u32, sz);
        let runner2 = gpu.plan(&device, &queue, src2_view, dst_view, &position_buffer);
        queue.write_buffer(&position_buffer, 0, bytemuck::bytes_of(&2u32));
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut cpass = encoder.begin_compute_pass(&Default::default());
            runner2.forward(&mut cpass);
        }
        queue.submit(Some(encoder.finish()));

        let actual = download_f32(&device, &queue, &dst_buf, num_rows * row_width);
        for r in 0..num_rows {
            for c in 0..row_width {
                let got = actual[r * row_width + c];
                let want = match r {
                    1 => (c + 1) as f32,
                    2 => -((c + 100) as f32),
                    _ => 0.0,
                };
                assert!(
                    (got - want).abs() < 1e-6,
                    "row {} col {}: got {}, want {}",
                    r,
                    c,
                    got,
                    want,
                );
            }
        }
    }

    /// Multi-row scatter: copy a 3-row source into rows [pos, pos+3).
    #[tokio::test]
    async fn test_scatter_row_multi_row() {
        let (device, queue) = gpu_or_skip!();
        let row_width = 32usize;
        let num_rows_src = 3usize;
        let num_rows_dst = 8usize;
        let pos = 2u32;

        let src: Vec<f32> = (0..num_rows_src * row_width)
            .map(|i| (i as f32) * 0.1 + 1.0)
            .collect();
        let dst: Vec<f32> = vec![0.0; num_rows_dst * row_width];

        let gpu = ScatterRowWebgpu::new(&device, row_width);
        let src_buf = upload_f32(&device, &src);
        let dst_buf = upload_f32(&device, &dst);
        let sz = std::mem::size_of::<f32>() as u32;
        let src_view =
            BufferView::new_2d_tight(&src_buf, num_rows_src as u32, row_width as u32, sz);
        let dst_view =
            BufferView::new_2d_tight(&dst_buf, num_rows_dst as u32, row_width as u32, sz);
        let position_buffer = make_position_buffer(&device, &queue, pos);

        let runner = gpu.plan(&device, &queue, src_view, dst_view, &position_buffer);
        run_blocking_compute(&device, &queue, |cp| runner.forward(cp));

        let actual = download_f32(&device, &queue, &dst_buf, num_rows_dst * row_width);
        for r in 0..num_rows_dst {
            for c in 0..row_width {
                let got = actual[r * row_width + c];
                let want = if r >= pos as usize && r < pos as usize + num_rows_src {
                    let src_r = r - pos as usize;
                    src[src_r * row_width + c]
                } else {
                    0.0
                };
                assert!(
                    (got - want).abs() < 1e-6,
                    "row {} col {}: got {}, want {}",
                    r,
                    c,
                    got,
                    want,
                );
            }
        }
    }
}
