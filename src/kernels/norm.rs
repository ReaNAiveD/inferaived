use safetensors::tensor::TensorView;
use wgpu::{BindGroupLayout, Buffer, ComputePipeline};

use crate::buffer_view::BufferView;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RmsNormParams {
    input_row_stride: u32,
    output_row_stride: u32,
    hidden_size: u32,
    seq_len: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RmsNormInplaceParams {
    hidden_row_stride: u32,
    hidden_size: u32,
    seq_len: u32,
    eps: f32,
}

pub struct RmsNormWebgpu {
    bind_group_layout: BindGroupLayout,
    pipeline: ComputePipeline,
    weight_buffer: Buffer,
    norm_dim: usize,
}

impl RmsNormWebgpu {
    pub fn new<'data>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        weight: TensorView<'data>,
    ) -> Self {
        debug_assert_eq!(
            weight.shape().len(),
            1,
            "RmsNormWebgpu weight must be 1-D, got shape {:?}",
            weight.shape(),
        );
        let norm_dim = weight.shape()[0] as usize;
        let weight_f32: Vec<f32> = weight
            .data()
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                half::bf16::from_bits(bits).to_f32()
            })
            .collect();
        debug_assert_eq!(
            weight_f32.len(),
            norm_dim,
            "RmsNormWebgpu weight data length ({} bf16 elements) does not match shape {:?}",
            weight_f32.len(),
            weight.shape(),
        );
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rms_norm/shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "wgsl-shaders/rms_norm.wgsl"
            ))),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rms_norm/bind_group_layout"),
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
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rms_norm/pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rms_norm/pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("workgroup_size", 256f64)],
                zero_initialize_workgroup_memory: true,
            },
            cache: None,
        });
        let weight_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rms_norm/weight_buffer"),
            size: (weight_f32.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&weight_buffer, 0, bytemuck::cast_slice(&weight_f32));
        Self {
            bind_group_layout,
            pipeline,
            weight_buffer,
            norm_dim,
        }
    }

    /// Bake the per-buffer bindings into a [`RmsNormWebgpuRunner`]
    /// for repeated dispatch into a caller-owned compute pass.
    pub fn plan(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: BufferView<'_>,
        dst: BufferView<'_>,
    ) -> RmsNormWebgpuRunner {
        debug_assert_eq!(
            input.shape[0], dst.shape[0],
            "rms_norm: outer dim mismatch (input={}, dst={})",
            input.shape[0], dst.shape[0],
        );
        let seq_len = input.shape[0];
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rms_norm_runner/uniform_buffer"),
            size: std::mem::size_of::<RmsNormParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform = RmsNormParams {
            input_row_stride: input.stride[0],
            output_row_stride: dst.stride[0],
            hidden_size: self.norm_dim as u32,
            seq_len,
            eps: 1e-6,
        };
        queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&uniform));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rms_norm_runner/bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dst.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.weight_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });
        RmsNormWebgpuRunner {
            pipeline: self.pipeline.clone(),
            bind_group,
            seq_len,
        }
    }
}

pub struct RmsNormWebgpuRunner {
    pipeline: ComputePipeline,
    bind_group: wgpu::BindGroup,
    seq_len: u32,
}

impl RmsNormWebgpuRunner {
    pub fn forward(&self, cpass: &mut wgpu::ComputePass<'_>) {
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.dispatch_workgroups(self.seq_len, 1, 1);
    }
}

pub struct RmsNormInplaceWebgpu {
    bind_group_layout: BindGroupLayout,
    pipeline: ComputePipeline,
    weight_buffer: Buffer,
    norm_dim: usize,
}

impl RmsNormInplaceWebgpu {
    pub fn new<'data>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        weight: TensorView<'data>,
    ) -> Self {
        debug_assert_eq!(
            weight.shape().len(),
            1,
            "RmsNormInplaceWebgpu weight must be 1-D, got shape {:?}",
            weight.shape(),
        );
        let norm_dim = weight.shape()[0] as usize;
        let weight_f32: Vec<f32> = weight
            .data()
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                half::bf16::from_bits(bits).to_f32()
            })
            .collect();
        debug_assert_eq!(
            weight_f32.len(),
            norm_dim,
            "RmsNormInplaceWebgpu weight data length ({} bf16 elements) does not match shape {:?}",
            weight_f32.len(),
            weight.shape(),
        );
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rms_norm_inplace/shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "wgsl-shaders/rms_norm_inplace.wgsl"
            ))),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rms_norm_inplace/bind_group_layout"),
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
            label: Some("rms_norm_inplace/pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rms_norm_inplace/pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("workgroup_size", 256f64)],
                zero_initialize_workgroup_memory: true,
            },
            cache: None,
        });
        let weight_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rms_norm_inplace/weight_buffer"),
            size: (weight_f32.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&weight_buffer, 0, bytemuck::cast_slice(&weight_f32));
        Self {
            bind_group_layout,
            pipeline,
            weight_buffer,
            norm_dim,
        }
    }

    /// Bake the per-buffer bindings into a [`RmsNormInplaceWebgpuRunner`]
    /// for repeated dispatch into a caller-owned compute pass.
    pub fn plan(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        hidden: BufferView<'_>,
    ) -> RmsNormInplaceWebgpuRunner {
        let seq_len = hidden.shape[0];
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rms_norm_inplace_runner/uniform_buffer"),
            size: std::mem::size_of::<RmsNormInplaceParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform = RmsNormInplaceParams {
            hidden_row_stride: hidden.stride[0],
            hidden_size: self.norm_dim as u32,
            seq_len,
            eps: 1e-6,
        };
        queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&uniform));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rms_norm_inplace_runner/bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: hidden.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.weight_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });
        RmsNormInplaceWebgpuRunner {
            pipeline: self.pipeline.clone(),
            bind_group,
            seq_len,
        }
    }
}

pub struct RmsNormInplaceWebgpuRunner {
    pipeline: ComputePipeline,
    bind_group: wgpu::BindGroup,
    seq_len: u32,
}

impl RmsNormInplaceWebgpuRunner {
    pub fn forward(&self, cpass: &mut wgpu::ComputePass<'_>) {
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.dispatch_workgroups(self.seq_len, 1, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_test_utils::*;

    /// CPU reference: out[t,i] = (input[t,i] / rms) * (1 + weight[i])
    fn cpu_rms_norm(
        input: &[f32],
        weight: &[f32],
        hidden_size: usize,
        seq_len: usize,
        eps: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; seq_len * hidden_size];
        for t in 0..seq_len {
            let row = &input[t * hidden_size..(t + 1) * hidden_size];
            let ss: f32 = row.iter().map(|x| x * x).sum();
            let scale = 1.0 / (ss / hidden_size as f32 + eps).sqrt();
            for i in 0..hidden_size {
                out[t * hidden_size + i] = row[i] * scale * (1.0 + weight[i]);
            }
        }
        out
    }

    #[tokio::test]
    async fn test_rms_norm() {
        let (device, queue) = gpu_or_skip!();
        let seq_len = 2;
        let hidden_size = 32;
        let input: Vec<f32> = (0..seq_len * hidden_size)
            .map(|i| ((i as f32) * 0.03).sin())
            .collect();
        let weight_f32: Vec<f32> = (0..hidden_size).map(|i| (i as f32) * 0.01).collect();

        let weight_bf16_bytes: Vec<u8> = weight_f32
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect();
        let tv = safetensors::tensor::TensorView::new(
            safetensors::Dtype::BF16,
            vec![hidden_size],
            &weight_bf16_bytes,
        )
        .unwrap();

        let weight_roundtrip: Vec<f32> = weight_bf16_bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect();
        let expected = cpu_rms_norm(&input, &weight_roundtrip, hidden_size, seq_len, 1e-6);

        let gpu = RmsNormWebgpu::new(&device, &queue, tv);
        let in_buf = upload_f32(&device, &input);
        let out_buf = create_f32_buffer(&device, seq_len * hidden_size);
        let elem_size = std::mem::size_of::<f32>() as u32;
        let in_view =
            BufferView::new_2d_tight(&in_buf, seq_len as u32, hidden_size as u32, elem_size);
        let out_view =
            BufferView::new_2d_tight(&out_buf, seq_len as u32, hidden_size as u32, elem_size);
        let runner = gpu.plan(&device, &queue, in_view, out_view);
        run_blocking_compute(&device, &queue, |cp| runner.forward(cp));
        let actual = download_f32(&device, &queue, &out_buf, seq_len * hidden_size);

        assert_approx_eq(&actual, &expected, 1e-4);
    }

    /// Mimics the decode-step use case: read a single row from the middle
    /// of a large input buffer (rows [input_start_row..input_start_row +
    /// num_rows)), write the normalized rows tight-packed starting at row
    /// 0 of a small output buffer sized for exactly num_rows.
    ///
    /// `hidden_size` is chosen so that `row_bytes = hidden_size * 4` is a
    /// multiple of `min_storage_buffer_offset_alignment` (256 on most
    /// adapters), which the binding-side offset encoding requires. Real
    /// model dims (e.g. `hidden_size = 1024`, `head_dim = 256`) satisfy
    /// this naturally; tests just have to pick aligned dims explicitly.
    #[tokio::test]
    async fn test_rms_norm_input_offset_decode_style() {
        let (device, queue) = gpu_or_skip!();
        let total_rows = 8;
        let input_start_row = 5;
        let num_rows = 1;
        let hidden_size = 64; // row_bytes = 256, aligned to all known adapters

        let input_full: Vec<f32> = (0..total_rows * hidden_size)
            .map(|i| ((i as f32) * 0.05).sin())
            .collect();
        let weight_f32: Vec<f32> = (0..hidden_size).map(|i| (i as f32) * 0.02 - 0.1).collect();

        let weight_bf16_bytes: Vec<u8> = weight_f32
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect();
        let weight_roundtrip: Vec<f32> = weight_bf16_bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect();
        let tv = safetensors::tensor::TensorView::new(
            safetensors::Dtype::BF16,
            vec![hidden_size],
            &weight_bf16_bytes,
        )
        .unwrap();

        let sliced_input =
            &input_full[input_start_row * hidden_size..(input_start_row + num_rows) * hidden_size];
        let expected = cpu_rms_norm(sliced_input, &weight_roundtrip, hidden_size, num_rows, 1e-6);

        let gpu = RmsNormWebgpu::new(&device, &queue, tv);
        let in_buf = upload_f32(&device, &input_full);
        // Output buffer sized only for `num_rows` — the kernel must write
        // tight-packed from row 0 (not at the input offset, which would
        // be out of bounds here).
        let out_buf = create_f32_buffer(&device, num_rows * hidden_size);
        let elem_size = std::mem::size_of::<f32>() as u32;
        let in_view =
            BufferView::new_2d_tight(&in_buf, total_rows as u32, hidden_size as u32, elem_size)
                .narrow(0, input_start_row as u32, num_rows as u32);
        let out_view =
            BufferView::new_2d_tight(&out_buf, num_rows as u32, hidden_size as u32, elem_size);
        let runner = gpu.plan(&device, &queue, in_view, out_view);
        run_blocking_compute(&device, &queue, |cp| runner.forward(cp));
        let actual = download_f32(&device, &queue, &out_buf, num_rows * hidden_size);

        assert_approx_eq(&actual, &expected, 1e-4);
    }

    /// CPU reference: hidden[t,i] = (hidden[t,i] / rms) * (1 + weight[i])
    fn cpu_rms_norm_inplace(
        hidden: &mut [f32],
        weight: &[f32],
        hidden_size: usize,
        offset: usize,
        n_rows: usize,
        row_stride: usize,
        eps: f32,
    ) {
        for t in 0..n_rows {
            let base = offset + t * row_stride;
            let ss: f32 = (0..hidden_size)
                .map(|i| hidden[base + i] * hidden[base + i])
                .sum();
            let scale = 1.0 / (ss / hidden_size as f32 + eps).sqrt();
            for i in 0..hidden_size {
                hidden[base + i] = hidden[base + i] * scale * (1.0 + weight[i]);
            }
        }
    }

    #[tokio::test]
    async fn test_rms_norm_inplace_basic() {
        let (device, queue) = gpu_or_skip!();
        let seq_len = 3;
        let hidden_size = 32;
        let data: Vec<f32> = (0..seq_len * hidden_size)
            .map(|i| ((i as f32) * 0.07).cos())
            .collect();
        let weight_f32: Vec<f32> = (0..hidden_size).map(|i| (i as f32) * -0.005).collect();

        let weight_bf16_bytes: Vec<u8> = weight_f32
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect();
        let weight_roundtrip: Vec<f32> = weight_bf16_bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect();
        let tv = safetensors::tensor::TensorView::new(
            safetensors::Dtype::BF16,
            vec![hidden_size],
            &weight_bf16_bytes,
        )
        .unwrap();

        let mut expected = data.clone();
        cpu_rms_norm_inplace(
            &mut expected,
            &weight_roundtrip,
            hidden_size,
            0,
            seq_len,
            hidden_size,
            1e-6,
        );

        let gpu = RmsNormInplaceWebgpu::new(&device, &queue, tv);
        let buf = upload_f32(&device, &data);
        let elem_size = std::mem::size_of::<f32>() as u32;
        let view = BufferView::new_2d_tight(&buf, seq_len as u32, hidden_size as u32, elem_size);
        let runner = gpu.plan(&device, &queue, view);
        run_blocking_compute(&device, &queue, |cp| runner.forward(cp));
        let actual = download_f32(&device, &queue, &buf, seq_len * hidden_size);

        assert_approx_eq(&actual, &expected, 1e-4);
    }
}
