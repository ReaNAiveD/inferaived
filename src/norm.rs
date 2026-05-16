use safetensors::tensor::TensorView;
use wgpu::{BindGroupLayout, Buffer, ComputePipeline};

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RmsNormParams {
    input_offset: u32,
    input_row_stride: u32,
    hidden_size: u32,
    seq_len: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RmsNormInplaceParams {
    hidden_offset: u32,
    hidden_row_stride: u32,
    hidden_size: u32,
    seq_len: u32,
    eps: f32,
}

pub struct RmsNormWebgpu {
    bind_group_layout: BindGroupLayout,
    pipeline: ComputePipeline,
    uniform_buffer: Buffer,
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
            bind_group_layouts: &[&bind_group_layout],
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
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rms_norm/uniform_buffer"),
            size: std::mem::size_of::<RmsNormParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
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
            uniform_buffer,
            weight_buffer,
            norm_dim,
        }
    }

    /// Normalize `num_rows` rows of `input_buffer` starting at row
    /// `input_start_row` and write the normalized rows tight-packed from
    /// row 0 of `dst_buffer`.
    pub fn forward(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_buffer: &Buffer,
        dst_buffer: &Buffer,
        input_start_row: usize,
        num_rows: usize,
    ) {
        let uniform = RmsNormParams {
            input_offset: (input_start_row * self.norm_dim) as u32,
            input_row_stride: self.norm_dim as u32,
            hidden_size: self.norm_dim as u32,
            seq_len: num_rows as u32,
            eps: 1e-6,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rms_norm/bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: dst_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.weight_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rms_norm/command_encoder"),
        });
        let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rms_norm/compute_pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(num_rows as u32, 1, 1);
        drop(compute_pass);
        queue.submit(Some(command_encoder.finish()));
    }
}

pub struct RmsNormInplaceWebgpu {
    bind_group_layout: BindGroupLayout,
    pipeline: ComputePipeline,
    uniform_buffer: Buffer,
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
            bind_group_layouts: &[&bind_group_layout],
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
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rms_norm_inplace/uniform_buffer"),
            size: std::mem::size_of::<RmsNormInplaceParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
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
            uniform_buffer,
            weight_buffer,
            norm_dim,
        }
    }

    /// In-place normalize `num_rows` rows of `src_buffer` starting at row
    /// `start_row`. Each row is `self.norm_dim` f32 elements.
    pub fn forward(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src_buffer: &Buffer,
        start_row: usize,
        num_rows: usize,
    ) {
        let uniform = RmsNormInplaceParams {
            hidden_offset: (start_row * self.norm_dim) as u32,
            hidden_row_stride: self.norm_dim as u32,
            hidden_size: self.norm_dim as u32,
            seq_len: num_rows as u32,
            eps: 1e-6,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniform));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rms_norm_inplace/bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.weight_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rms_norm_inplace/command_encoder"),
        });
        let mut compute_pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rms_norm_inplace/compute_pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(num_rows as u32, 1, 1);
        drop(compute_pass);
        queue.submit(Some(command_encoder.finish()));
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
        gpu.forward(&device, &queue, &in_buf, &out_buf, 0, seq_len);
        let actual = download_f32(&device, &queue, &out_buf, seq_len * hidden_size);

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
        gpu.forward(&device, &queue, &buf, 0, seq_len);
        let actual = download_f32(&device, &queue, &buf, seq_len * hidden_size);

        assert_approx_eq(&actual, &expected, 1e-4);
    }
}
