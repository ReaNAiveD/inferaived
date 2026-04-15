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
    stride_qk_head: u32,   // = key_head_dim
    stride_v_head: u32,    // = value_head_dim (may differ from key_head_dim)
    stride_qkv_token: u32, // = num_key_heads * key_head_dim * 2 + num_value_heads * value_head_dim

    // Projection buffers (per-head scalars)
    proj_a_offset: u32,
    stride_proj_a_token: u32, // = num_key_heads
    proj_b_offset: u32,
    stride_proj_b_token: u32, // = num_key_heads

    // Gate params buffer (dt_bias and A_log packed together)
    dt_bias_offset: u32,
    a_log_offset: u32,

    // Output buffer
    stride_dst_token: u32, // = num_key_heads * value_head_dim
    stride_dst_head: u32,  // = value_head_dim

    // State buffer
    stride_state_head: u32, // = key_head_dim * value_head_dim

    eps: f32,
}

pub struct DeltaRuleWebgpu {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    gate_params_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,

    num_key_heads: u32,
    key_head_dim: u32,
    value_head_dim: u32,
}

impl DeltaRuleWebgpu {
    const WORKGROUP_SIZE: usize = 128;

    pub fn new<'data>(
        device: &wgpu::Device,
        dt_bias_tensor: TensorView<'data>,
        a_log_tensor: TensorView<'data>,
        num_key_heads: u32,
        key_head_dim: u32,
        value_head_dim: u32,
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
        assert_eq!(dt_bias.len(), num_key_heads as usize);
        assert_eq!(a_log.len(), num_key_heads as usize);
        let cols_per_thread =
            (value_head_dim as usize + Self::WORKGROUP_SIZE - 1) / Self::WORKGROUP_SIZE;
        assert!(
            cols_per_thread * key_head_dim as usize <= 256,
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

    pub fn compute(
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
            num_key_heads: self.num_key_heads,
            key_head_dim: self.key_head_dim,
            value_head_dim: self.value_head_dim,
            seq_len: seq_len as u32,
            q_offset: 0,
            k_offset: self.key_head_dim * self.num_key_heads,
            v_offset: self.key_head_dim * self.num_key_heads * 2,
            stride_qk_head: self.key_head_dim,
            stride_v_head: self.value_head_dim,
            stride_qkv_token: self.num_key_heads * self.key_head_dim * 2
                + self.num_key_heads * self.value_head_dim,
            proj_a_offset: 0,
            stride_proj_a_token: self.num_key_heads,
            proj_b_offset: 0,
            stride_proj_b_token: self.num_key_heads,
            dt_bias_offset: 0,
            a_log_offset: self.num_key_heads as u32,
            stride_dst_token: (self.num_key_heads * self.value_head_dim) as u32,
            stride_dst_head: self.value_head_dim as u32,
            stride_state_head: (self.key_head_dim * self.value_head_dim) as u32,
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
            cpass.dispatch_workgroups(self.num_key_heads, 1, 1);
        }
        queue.submit(Some(command_encoder.finish()));
    }
}
