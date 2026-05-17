use bytemuck;
use safetensors::tensor::TensorView;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ConvSiluParams {
    // Channel group dimensions: layout is [Q, K, V] contiguous per token
    q_dim: u32,
    k_dim: u32,
    v_dim: u32,

    seq_len: u32,
    kernel_size: u32,

    // Elements between consecutive tokens (>= q_dim + k_dim + v_dim when padded)
    input_token_stride: u32,
    output_token_stride: u32,
    state_token_stride: u32,

    // Per-group flag: 0 = passthrough copy, 1 = conv1d + silu
    q_apply_conv: u32,
    k_apply_conv: u32,
    v_apply_conv: u32,
}

/// Per-channel-group processing mode for ConvSilu.
#[derive(Debug, Clone, Copy)]
pub enum ChannelMode {
    /// Copy the channel values from src to dst unchanged.
    Copy = 0,
    /// Apply depthwise causal conv1d followed by SiLU activation.
    ConvSilu = 1,
}

pub struct ConvSiluWebgpu {
    bind_group_layout: wgpu::BindGroupLayout,
    conv_pipeline: wgpu::ComputePipeline,
    state_update_pipeline: wgpu::ComputePipeline,
    weights_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,

    // Model dimensions
    q_dim: usize,
    k_dim: usize,
    v_dim: usize,
    kernel_size: usize,

    // Per-group modes
    q_mode: ChannelMode,
    k_mode: ChannelMode,
    v_mode: ChannelMode,
}

impl ConvSiluWebgpu {
    pub fn new<'data>(
        device: &wgpu::Device,
        weights: TensorView<'data>,
        q_dim: usize,
        k_dim: usize,
        v_dim: usize,
        kernel_size: usize,
        q_mode: ChannelMode,
        k_mode: ChannelMode,
        v_mode: ChannelMode,
    ) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("conv_silu/shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "wgsl-shaders/depthwise_causal_conv_silu.wgsl"
            ))),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("conv_silu/bind_group_layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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
            label: None,
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let conv_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("conv_silu/conv_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("conv1d_silu"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let state_update_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("conv_silu/state_update_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("conv_state_update"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let weights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("conv_silu/weights_buffer"),
            contents: bytemuck::cast_slice(weights.data()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("conv_silu/uniform_buffer"),
            size: std::mem::size_of::<ConvSiluParams>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            bind_group_layout,
            conv_pipeline,
            state_update_pipeline,
            weights_buffer,
            uniform_buffer,
            q_dim,
            k_dim,
            v_dim,
            kernel_size,
            q_mode,
            k_mode,
            v_mode,
        }
    }

    /// Run depthwise causal conv1d + SiLU over `num_rows` token rows of
    /// `src_buffer` (read tight from row 0), writing the activated
    /// outputs to `dst_buffer` (also tight from row 0).
    pub fn forward(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src_buffer: &wgpu::Buffer,
        dst_buffer: &wgpu::Buffer,
        state_buffer: &wgpu::Buffer,
        seq_len: usize,
    ) {
        let num_channels = self.q_dim + self.k_dim + self.v_dim;
        let params = ConvSiluParams {
            q_dim: self.q_dim as u32,
            k_dim: self.k_dim as u32,
            v_dim: self.v_dim as u32,
            seq_len: seq_len as u32,
            kernel_size: self.kernel_size as u32,
            input_token_stride: num_channels as u32,
            output_token_stride: num_channels as u32,
            state_token_stride: num_channels as u32,
            q_apply_conv: self.q_mode as u32,
            k_apply_conv: self.k_mode as u32,
            v_apply_conv: self.v_mode as u32,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[params]));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("conv_silu/bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: src_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.weights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dst_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("conv_silu/command_encoder"),
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("conv_silu/compute_pass"),
                timestamp_writes: None,
            });
            let workgroup_size = 256usize;
            // Conv reads from state; safe because state writes happen in
            // the next dispatch (consecutive dispatches in a compute pass
            // are ordered by wgpu).
            compute_pass.set_pipeline(&self.conv_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let conv_workgroups =
                (seq_len * num_channels + workgroup_size - 1) / workgroup_size;
            compute_pass.dispatch_workgroups(conv_workgroups as u32, 1, 1);
            // Refresh the rolling-window state for next call.
            if self.kernel_size >= 2 {
                compute_pass.set_pipeline(&self.state_update_pipeline);
                let update_workgroups = (num_channels + workgroup_size - 1) / workgroup_size;
                compute_pass.dispatch_workgroups(update_workgroups as u32, 1, 1);
            }
        }
        queue.submit(Some(encoder.finish()));
    }

    /// f32 element count of the conv state buffer this kernel reads from
    /// and writes back to: `(K - 1) * num_channels`. Returns 0 if
    /// `kernel_size <= 1` (the shader skips state work in that case).
    pub fn conv_state_size(&self) -> usize {
        let num_channels = self.q_dim + self.k_dim + self.v_dim;
        if self.kernel_size <= 1 {
            0
        } else {
            (self.kernel_size - 1) * num_channels
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_test_utils::*;

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    /// CPU reference: per-channel causal 1D conv + SiLU
    fn cpu_conv_silu(
        input: &[f32],
        weight_packed: &[u32],
        q_dim: usize,
        k_dim: usize,
        v_dim: usize,
        seq_len: usize,
        kernel_size: usize,
        q_mode: ChannelMode,
        k_mode: ChannelMode,
        v_mode: ChannelMode,
        input_token_stride: usize,
        output_token_stride: usize,
    ) -> Vec<f32> {
        let nc = q_dim + k_dim + v_dim;
        let mut output = vec![0.0f32; seq_len * output_token_stride];
        for token in 0..seq_len {
            for ch in 0..nc {
                let apply_conv = if ch < q_dim {
                    matches!(q_mode, ChannelMode::ConvSilu)
                } else if ch < q_dim + k_dim {
                    matches!(k_mode, ChannelMode::ConvSilu)
                } else {
                    matches!(v_mode, ChannelMode::ConvSilu)
                };

                let out_idx = token * output_token_stride + ch;
                if !apply_conv {
                    output[out_idx] = input[token * input_token_stride + ch];
                } else {
                    let mut sum = 0.0f32;
                    for ki in 0..kernel_size {
                        let lag = kernel_size - 1 - ki;
                        let inp = if lag > token {
                            0.0
                        } else {
                            input[(token - lag) * input_token_stride + ch]
                        };
                        let w_idx = ch * kernel_size + ki;
                        let w = unpack_bf16(weight_packed, w_idx);
                        sum += inp * w;
                    }
                    output[out_idx] = silu(sum);
                }
            }
        }
        output
    }

    #[tokio::test]
    async fn test_conv_silu() {
        let (device, queue) = gpu_or_skip!();
        let q_dim = 8;
        let k_dim = 4;
        let v_dim = 4;
        let nc = q_dim + k_dim + v_dim;
        let seq_len = 4;
        let kernel_size = 4;

        let weight_f32: Vec<f32> = (0..nc * kernel_size)
            .map(|i| ((i as f32) * 0.13).sin() * 0.5)
            .collect();
        let padded_weight = if weight_f32.len() % 2 != 0 {
            let mut w = weight_f32.clone();
            w.push(0.0);
            w
        } else {
            weight_f32.clone()
        };
        let weight_packed = pack_f32_to_bf16_u32(&padded_weight);
        let weight_bf16_bytes: Vec<u8> = padded_weight
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect();

        let input: Vec<f32> = (0..seq_len * nc)
            .map(|i| ((i as f32) * 0.1).sin())
            .collect();

        let expected = cpu_conv_silu(
            &input,
            &weight_packed,
            q_dim,
            k_dim,
            v_dim,
            seq_len,
            kernel_size,
            ChannelMode::ConvSilu,
            ChannelMode::ConvSilu,
            ChannelMode::ConvSilu,
            nc,
            nc,
        );

        let tv = safetensors::tensor::TensorView::new(
            safetensors::Dtype::BF16,
            vec![nc, kernel_size],
            &weight_bf16_bytes,
        )
        .unwrap();
        let gpu = ConvSiluWebgpu::new(
            &device,
            tv,
            q_dim,
            k_dim,
            v_dim,
            kernel_size,
            ChannelMode::ConvSilu,
            ChannelMode::ConvSilu,
            ChannelMode::ConvSilu,
        );
        let in_buf = upload_f32(&device, &input);
        let out_buf = create_f32_buffer(&device, seq_len * nc);
        let state_buf = create_f32_buffer(&device, gpu.conv_state_size());
        gpu.forward(&device, &queue, &in_buf, &out_buf, &state_buf, seq_len);
        let actual = download_f32(&device, &queue, &out_buf, seq_len * nc);

        assert_approx_eq(&actual, &expected, 1e-2);
    }
}
