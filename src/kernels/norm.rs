use safetensors::tensor::TensorView;
use wgpu::{BindGroupLayout, Buffer, ComputePipeline};

use crate::buffer_view::BufferView;

// RMSNorm comes in two gain conventions that differ only in the per-channel
// scale applied after normalization:
//
//   * Llama / MiniCPM (`LlamaRMSNorm`): `out = x_normed * weight`
//   * Gemma-style centered weights (also this repo's Qwen3.5 checkpoint):
//     `out = x_normed * (1 + weight)`
//
// Following the vLLM / HF Transformers pattern, each convention is its own
// public type backed by its own shader, so the convention is visible at the
// call site and there is no runtime branch. The shared dispatch boilerplate
// lives in the private `*Impl` structs below.

/// Decode a 1-D bf16 RMSNorm weight tensor into f32, validating its shape.
fn load_norm_weight(weight: &TensorView<'_>, what: &str) -> Vec<f32> {
    debug_assert_eq!(
        weight.shape().len(),
        1,
        "{what} weight must be 1-D, got shape {:?}",
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
        "{what} weight data length ({} bf16 elements) does not match shape {:?}",
        weight_f32.len(),
        weight.shape(),
    );
    weight_f32
}

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

/// Shared out-of-place RMSNorm implementation. Both gain conventions use the
/// identical 4-binding layout and dispatch; only the shader source differs.
struct RmsNormOutOfPlaceImpl {
    bind_group_layout: BindGroupLayout,
    pipeline: ComputePipeline,
    weight_buffer: Buffer,
    norm_dim: usize,
}

impl RmsNormOutOfPlaceImpl {
    fn new<'data>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        weight: TensorView<'data>,
        shader_source: &str,
        label: &str,
    ) -> Self {
        let weight_f32 = load_norm_weight(&weight, label);
        Self::from_weight_f32(device, queue, weight_f32, shader_source, label)
    }

    fn from_weight_f32(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        weight_f32: Vec<f32>,
        shader_source: &str,
        label: &str,
    ) -> Self {
        let norm_dim = weight_f32.len();
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{label}/shader")),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader_source)),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{label}/bind_group_layout")),
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
            label: Some(&format!("{label}/pipeline_layout")),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{label}/pipeline")),
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
            label: Some(&format!("{label}/weight_buffer")),
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

    fn plan(
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

/// Out-of-place RMSNorm with plain `weight` gain (Llama / MiniCPM).
pub struct LlamaRmsNormWebgpu(RmsNormOutOfPlaceImpl);

impl LlamaRmsNormWebgpu {
    pub fn new<'data>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        weight: TensorView<'data>,
    ) -> Self {
        Self(RmsNormOutOfPlaceImpl::new(
            device,
            queue,
            weight,
            include_str!("wgsl-shaders/llama_rms_norm.wgsl"),
            "llama_rms_norm",
        ))
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
        self.0.plan(device, queue, input, dst)
    }
}

pub struct ParameterlessRmsNormWebgpu(RmsNormOutOfPlaceImpl);

impl ParameterlessRmsNormWebgpu {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, norm_dim: usize) -> Self {
        debug_assert!(norm_dim >= 1, "parameterless_rms_norm: norm_dim must be >= 1");
        Self(RmsNormOutOfPlaceImpl::from_weight_f32(
            device,
            queue,
            vec![1.0f32; norm_dim],
            include_str!("wgsl-shaders/llama_rms_norm.wgsl"),
            "parameterless_rms_norm",
        ))
    }

    pub fn plan(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: BufferView<'_>,
        dst: BufferView<'_>,
    ) -> RmsNormWebgpuRunner {
        self.0.plan(device, queue, input, dst)
    }
}

/// Out-of-place RMSNorm with `1 + weight` gain (Gemma-style; Qwen3.5).
pub struct GemmaRmsNormWebgpu(RmsNormOutOfPlaceImpl);

impl GemmaRmsNormWebgpu {
    pub fn new<'data>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        weight: TensorView<'data>,
    ) -> Self {
        Self(RmsNormOutOfPlaceImpl::new(
            device,
            queue,
            weight,
            include_str!("wgsl-shaders/gemma_rms_norm.wgsl"),
            "gemma_rms_norm",
        ))
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
        self.0.plan(device, queue, input, dst)
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

/// Shared in-place RMSNorm implementation. Both gain conventions use the
/// identical 3-binding layout and dispatch; only the shader source differs.
struct RmsNormInplaceImpl {
    bind_group_layout: BindGroupLayout,
    pipeline: ComputePipeline,
    weight_buffer: Buffer,
    norm_dim: usize,
}

impl RmsNormInplaceImpl {
    fn new<'data>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        weight: TensorView<'data>,
        shader_source: &str,
        label: &str,
    ) -> Self {
        let weight_f32 = load_norm_weight(&weight, label);
        let norm_dim = weight_f32.len();
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{label}/shader")),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader_source)),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{label}/bind_group_layout")),
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
            label: Some(&format!("{label}/pipeline_layout")),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{label}/pipeline")),
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
            label: Some(&format!("{label}/weight_buffer")),
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

    fn plan(
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

/// In-place RMSNorm with plain `weight` gain (Llama / MiniCPM).
pub struct LlamaRmsNormInplaceWebgpu(RmsNormInplaceImpl);

impl LlamaRmsNormInplaceWebgpu {
    pub fn new<'data>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        weight: TensorView<'data>,
    ) -> Self {
        Self(RmsNormInplaceImpl::new(
            device,
            queue,
            weight,
            include_str!("wgsl-shaders/llama_rms_norm_inplace.wgsl"),
            "llama_rms_norm_inplace",
        ))
    }

    /// Bake the per-buffer bindings into a [`RmsNormInplaceWebgpuRunner`]
    /// for repeated dispatch into a caller-owned compute pass.
    pub fn plan(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        hidden: BufferView<'_>,
    ) -> RmsNormInplaceWebgpuRunner {
        self.0.plan(device, queue, hidden)
    }
}

/// In-place RMSNorm with `1 + weight` gain (Gemma-style; Qwen3.5).
pub struct GemmaRmsNormInplaceWebgpu(RmsNormInplaceImpl);

impl GemmaRmsNormInplaceWebgpu {
    pub fn new<'data>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        weight: TensorView<'data>,
    ) -> Self {
        Self(RmsNormInplaceImpl::new(
            device,
            queue,
            weight,
            include_str!("wgsl-shaders/gemma_rms_norm_inplace.wgsl"),
            "gemma_rms_norm_inplace",
        ))
    }

    /// Bake the per-buffer bindings into a [`RmsNormInplaceWebgpuRunner`]
    /// for repeated dispatch into a caller-owned compute pass.
    pub fn plan(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        hidden: BufferView<'_>,
    ) -> RmsNormInplaceWebgpuRunner {
        self.0.plan(device, queue, hidden)
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
        unit_offset: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; seq_len * hidden_size];
        for t in 0..seq_len {
            let row = &input[t * hidden_size..(t + 1) * hidden_size];
            let ss: f32 = row.iter().map(|x| x * x).sum();
            let scale = 1.0 / (ss / hidden_size as f32 + eps).sqrt();
            for i in 0..hidden_size {
                out[t * hidden_size + i] = row[i] * scale * (unit_offset + weight[i]);
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
        let expected = cpu_rms_norm(&input, &weight_roundtrip, hidden_size, seq_len, 1e-6, 1.0);

        let gpu = GemmaRmsNormWebgpu::new(&device, &queue, tv);
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
        let expected = cpu_rms_norm(
            sliced_input,
            &weight_roundtrip,
            hidden_size,
            num_rows,
            1e-6,
            1.0,
        );

        let gpu = GemmaRmsNormWebgpu::new(&device, &queue, tv);
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

    /// CPU reference: hidden[t,i] = (hidden[t,i] / rms) * (unit_offset + weight[i])
    fn cpu_rms_norm_inplace(
        hidden: &mut [f32],
        weight: &[f32],
        hidden_size: usize,
        offset: usize,
        n_rows: usize,
        row_stride: usize,
        eps: f32,
        unit_offset: f32,
    ) {
        for t in 0..n_rows {
            let base = offset + t * row_stride;
            let ss: f32 = (0..hidden_size)
                .map(|i| hidden[base + i] * hidden[base + i])
                .sum();
            let scale = 1.0 / (ss / hidden_size as f32 + eps).sqrt();
            for i in 0..hidden_size {
                hidden[base + i] = hidden[base + i] * scale * (unit_offset + weight[i]);
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
            1.0,
        );

        let gpu = GemmaRmsNormInplaceWebgpu::new(&device, &queue, tv);
        let buf = upload_f32(&device, &data);
        let elem_size = std::mem::size_of::<f32>() as u32;
        let view = BufferView::new_2d_tight(&buf, seq_len as u32, hidden_size as u32, elem_size);
        let runner = gpu.plan(&device, &queue, view);
        run_blocking_compute(&device, &queue, |cp| runner.forward(cp));
        let actual = download_f32(&device, &queue, &buf, seq_len * hidden_size);

        assert_approx_eq(&actual, &expected, 1e-4);
    }

    #[tokio::test]
    async fn test_llama_rms_norm_plain_gain() {
        let (device, queue) = gpu_or_skip!();
        let seq_len = 2;
        let hidden_size = 32;
        let input: Vec<f32> = (0..seq_len * hidden_size)
            .map(|i| ((i as f32) * 0.03).sin())
            .collect();
        let weight_f32: Vec<f32> = (0..hidden_size).map(|i| (i as f32) * 0.01 + 0.5).collect();

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
        // Plain gain: unit_offset = 0.0.
        let expected = cpu_rms_norm(&input, &weight_roundtrip, hidden_size, seq_len, 1e-6, 0.0);

        let gpu = LlamaRmsNormWebgpu::new(&device, &queue, tv);
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

    #[tokio::test]
    async fn test_llama_rms_norm_inplace_plain_gain() {
        let (device, queue) = gpu_or_skip!();
        let seq_len = 3;
        let hidden_size = 32;
        let data: Vec<f32> = (0..seq_len * hidden_size)
            .map(|i| ((i as f32) * 0.07).cos())
            .collect();
        let weight_f32: Vec<f32> = (0..hidden_size)
            .map(|i| (i as f32) * -0.005 + 1.0)
            .collect();

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
        // Plain gain: unit_offset = 0.0.
        cpu_rms_norm_inplace(
            &mut expected,
            &weight_roundtrip,
            hidden_size,
            0,
            seq_len,
            hidden_size,
            1e-6,
            0.0,
        );

        let gpu = LlamaRmsNormInplaceWebgpu::new(&device, &queue, tv);
        let buf = upload_f32(&device, &data);
        let elem_size = std::mem::size_of::<f32>() as u32;
        let view = BufferView::new_2d_tight(&buf, seq_len as u32, hidden_size as u32, elem_size);
        let runner = gpu.plan(&device, &queue, view);
        run_blocking_compute(&device, &queue, |cp| runner.forward(cp));
        let actual = download_f32(&device, &queue, &buf, seq_len * hidden_size);

        assert_approx_eq(&actual, &expected, 1e-4);
    }
}
