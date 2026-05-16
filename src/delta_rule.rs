use half::bf16;
use safetensors::tensor::TensorView;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DeltaRuleParams {
    // Model dimensions
    num_key_heads: u32,
    key_head_dim: u32,
    value_head_dim: u32,
    seq_len: u32,

    // QKV source buffer layout
    q_offset: u32,
    k_offset: u32,
    v_offset: u32,
    qk_head_stride: u32,   // = key_head_dim
    v_head_stride: u32,    // = value_head_dim (may differ from key_head_dim)
    qkv_token_stride: u32, // = num_key_heads * key_head_dim * 2 + num_value_heads * value_head_dim

    // Projection buffers (per-head scalars)
    proj_a_offset: u32,
    proj_a_token_stride: u32, // = num_key_heads
    proj_b_offset: u32,
    proj_b_token_stride: u32, // = num_key_heads

    // Gate params buffer (dt_bias and A_log packed together)
    dt_bias_offset: u32,
    a_log_offset: u32,

    // Output buffer
    output_token_stride: u32, // = num_key_heads * value_head_dim
    output_head_stride: u32,  // = value_head_dim

    // State buffer
    state_head_stride: u32, // = key_head_dim * value_head_dim

    eps: f32,
}

pub struct DeltaRuleWebgpu {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    gate_params_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,

    num_key_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
}

impl DeltaRuleWebgpu {
    const WORKGROUP_SIZE: usize = 128;

    pub fn new<'data>(
        device: &wgpu::Device,
        dt_bias_tensor: TensorView<'data>,
        a_log_tensor: TensorView<'data>,
        num_key_heads: usize,
        key_head_dim: usize,
        value_head_dim: usize,
    ) -> Self {
        let dt_bias: Vec<f32> = dt_bias_tensor
            .data()
            .chunks_exact(2)
            .map(|pair| bf16::from_le_bytes([pair[0], pair[1]]).to_f32())
            .collect();
        let a_log: Vec<f32> = a_log_tensor
            .data()
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect();
        debug_assert_eq!(dt_bias.len(), num_key_heads);
        debug_assert_eq!(a_log.len(), num_key_heads);
        let cols_per_thread = (value_head_dim + Self::WORKGROUP_SIZE - 1) / Self::WORKGROUP_SIZE;
        debug_assert!(
            cols_per_thread * key_head_dim <= 256,
            "Private state overflow: need {} floats per thread but MAX_KEY_HEAD_DIM=256. \
             Increase MAX_KEY_HEAD_DIM in delta_rule.wgsl or increase WORKGROUP_SIZE.",
            cols_per_thread * key_head_dim as usize
        );
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("delta_rule/shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "wgsl-shaders/delta_rule.wgsl"
            ))),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("delta_rule/bind_group_layout"),
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
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
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("delta_rule/pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("delta_rule"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("workgroup_size".into(), Self::WORKGROUP_SIZE as f64)],
                zero_initialize_workgroup_memory: true,
            },
            cache: None,
        });
        let gate_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("delta_rule/gate_params_buffer"),
            contents: bytemuck::cast_slice(&[dt_bias, a_log].concat()),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("delta_rule/uniform_buffer"),
            size: std::mem::size_of::<DeltaRuleParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            bind_group_layout,
            pipeline,
            gate_params_buffer,
            uniform_buffer,
            num_key_heads,
            key_head_dim,
            value_head_dim,
        }
    }

    /// Run the gated-delta-rule SSM over `num_rows` token rows of the
    /// QKV / projection buffers (all read tight from row 0).
    pub fn forward(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        qkv_buffer: &wgpu::Buffer,
        proj_a_buffer: &wgpu::Buffer,
        proj_b_buffer: &wgpu::Buffer,
        state_buffer: &wgpu::Buffer,
        dst_buffer: &wgpu::Buffer,
        seq_len: usize,
    ) {
        let params = DeltaRuleParams {
            num_key_heads: self.num_key_heads as u32,
            key_head_dim: self.key_head_dim as u32,
            value_head_dim: self.value_head_dim as u32,
            seq_len: seq_len as u32,
            q_offset: 0,
            k_offset: (self.key_head_dim * self.num_key_heads) as u32,
            v_offset: (self.key_head_dim * self.num_key_heads * 2) as u32,
            qk_head_stride: self.key_head_dim as u32,
            v_head_stride: self.value_head_dim as u32,
            qkv_token_stride: (self.num_key_heads * self.key_head_dim * 2
                + self.num_key_heads * self.value_head_dim) as u32,
            proj_a_offset: 0,
            proj_a_token_stride: self.num_key_heads as u32,
            proj_b_offset: 0,
            proj_b_token_stride: self.num_key_heads as u32,
            dt_bias_offset: 0,
            a_log_offset: self.num_key_heads as u32,
            output_token_stride: (self.num_key_heads * self.value_head_dim) as u32,
            output_head_stride: self.value_head_dim as u32,
            state_head_stride: (self.key_head_dim * self.value_head_dim) as u32,
            eps: 1e-6,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[params]));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("delta_rule/bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: qkv_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: proj_a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: proj_b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.gate_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: dst_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("delta_rule/command_encoder"),
        });
        {
            let mut cpass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("delta_rule/compute_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(self.num_key_heads as u32, 1, 1);
        }
        queue.submit(Some(command_encoder.finish()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_test_utils::*;

    fn softplus(x: f32) -> f32 {
        if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
    }
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// CPU reference for delta_rule SSM
    fn cpu_delta_rule(
        qkv: &mut [f32],
        proj_a: &[f32],
        proj_b: &[f32],
        dt_bias: &[f32],
        a_log: &[f32],
        state: &mut [f32],
        num_key_heads: usize,
        key_head_dim: usize,
        value_head_dim: usize,
        seq_len: usize,
        eps: f32,
    ) -> Vec<f32> {
        let qk_head_stride = key_head_dim;
        let v_head_stride = value_head_dim;
        let qkv_token_stride = num_key_heads * key_head_dim * 2 + num_key_heads * value_head_dim;
        let q_offset = 0;
        let k_offset = num_key_heads * key_head_dim;
        let v_offset = num_key_heads * key_head_dim * 2;
        let state_head_stride = key_head_dim * value_head_dim;
        let output_head_stride = value_head_dim;
        let output_token_stride = num_key_heads * value_head_dim;

        let mut output = vec![0.0f32; seq_len * output_token_stride];

        for head in 0..num_key_heads {
            let al = a_log[head];
            let dtb = dt_bias[head];

            for token in 0..seq_len {
                let beta = sigmoid(proj_b[token * num_key_heads + head]);
                let g = -al.exp() * softplus(proj_a[token * num_key_heads + head] + dtb);
                let gamma = g.exp();

                let mut q_sq = 0.0f32;
                for i in 0..key_head_dim {
                    let v = qkv[q_offset + i + head * qk_head_stride + token * qkv_token_stride];
                    q_sq += v * v;
                }
                let q_scale = 1.0 / ((q_sq + eps) * key_head_dim as f32).sqrt();
                for i in 0..key_head_dim {
                    let idx = q_offset + i + head * qk_head_stride + token * qkv_token_stride;
                    qkv[idx] *= q_scale;
                }

                let mut k_sq = 0.0f32;
                for i in 0..key_head_dim {
                    let v = qkv[k_offset + i + head * qk_head_stride + token * qkv_token_stride];
                    k_sq += v * v;
                }
                let k_scale = 1.0 / (k_sq + eps).sqrt();
                for i in 0..key_head_dim {
                    let idx = k_offset + i + head * qk_head_stride + token * qkv_token_stride;
                    qkv[idx] *= k_scale;
                }

                for vi in 0..value_head_dim {
                    let mut kv_mem = 0.0f32;
                    for ki in 0..key_head_dim {
                        let s_idx = head * state_head_stride + ki * value_head_dim + vi;
                        state[s_idx] *= gamma;
                        let k_val =
                            qkv[k_offset + ki + head * qk_head_stride + token * qkv_token_stride];
                        kv_mem += k_val * state[s_idx];
                    }

                    let v_val =
                        qkv[v_offset + vi + head * v_head_stride + token * qkv_token_stride];
                    let delta_v = (v_val - kv_mem) * beta;

                    let mut out_acc = 0.0f32;
                    for ki in 0..key_head_dim {
                        let s_idx = head * state_head_stride + ki * value_head_dim + vi;
                        let k_val =
                            qkv[k_offset + ki + head * qk_head_stride + token * qkv_token_stride];
                        state[s_idx] += k_val * delta_v;
                        let q_val =
                            qkv[q_offset + ki + head * qk_head_stride + token * qkv_token_stride];
                        out_acc += q_val * state[s_idx];
                    }
                    output[token * output_token_stride + head * output_head_stride + vi] = out_acc;
                }
            }
        }
        output
    }

    #[tokio::test]
    async fn test_delta_rule() {
        let (device, queue) = gpu_or_skip!();
        let num_key_heads = 2;
        let key_head_dim = 8;
        let value_head_dim = 8;
        let seq_len = 3;
        let eps = 1e-6f32;

        let qkv_token_stride = num_key_heads * key_head_dim * 2 + num_key_heads * value_head_dim;
        let qkv: Vec<f32> = (0..seq_len * qkv_token_stride)
            .map(|i| ((i as f32) * 0.1).sin() * 0.5)
            .collect();
        let proj_a: Vec<f32> = (0..seq_len * num_key_heads)
            .map(|i| (i as f32) * 0.1 - 0.5)
            .collect();
        let proj_b: Vec<f32> = (0..seq_len * num_key_heads)
            .map(|i| (i as f32) * 0.15)
            .collect();
        let dt_bias: Vec<f32> = (0..num_key_heads).map(|i| (i as f32) * 0.1).collect();
        let a_log: Vec<f32> = (0..num_key_heads)
            .map(|i| -1.0 + (i as f32) * 0.2)
            .collect();

        let state_size = num_key_heads * key_head_dim * value_head_dim;

        let dt_bias_bf16: Vec<u8> = dt_bias
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect();
        let a_log_f32_bytes: Vec<u8> = a_log.iter().flat_map(|&v| v.to_le_bytes()).collect();
        let dt_tv = safetensors::tensor::TensorView::new(
            safetensors::Dtype::BF16,
            vec![num_key_heads],
            &dt_bias_bf16,
        )
        .unwrap();
        let al_tv = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            vec![num_key_heads],
            &a_log_f32_bytes,
        )
        .unwrap();

        let dt_bias_rt: Vec<f32> = dt_bias
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_f32())
            .collect();
        let mut cpu_state = vec![0.0f32; state_size];
        let mut cpu_qkv = qkv.clone();
        let expected = cpu_delta_rule(
            &mut cpu_qkv,
            &proj_a,
            &proj_b,
            &dt_bias_rt,
            &a_log,
            &mut cpu_state,
            num_key_heads,
            key_head_dim,
            value_head_dim,
            seq_len,
            eps,
        );

        let gpu = DeltaRuleWebgpu::new(
            &device,
            dt_tv,
            al_tv,
            num_key_heads,
            key_head_dim,
            value_head_dim,
        );
        let qkv_buf = upload_f32(&device, &qkv);
        let pa_buf = upload_f32(&device, &proj_a);
        let pb_buf = upload_f32(&device, &proj_b);
        let state_buf = upload_f32(&device, &vec![0.0f32; state_size]);
        let out_buf = create_f32_buffer(&device, seq_len * num_key_heads * value_head_dim);
        gpu.forward(
            &device, &queue, &qkv_buf, &pa_buf, &pb_buf, &state_buf, &out_buf, seq_len,
        );
        let actual = download_f32(
            &device,
            &queue,
            &out_buf,
            seq_len * num_key_heads * value_head_dim,
        );

        assert_approx_eq(&actual, &expected, 1e-3);
    }
}
