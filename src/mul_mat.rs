use safetensors::tensor::TensorView;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MulMatUniform {
    pub offset_src0: u32,
    pub offset_src1: u32,
    pub offset_dst: u32,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub stride_src0_1: u32,
    pub stride_src0_2: u32,
    pub stride_src0_3: u32,
    pub stride_src1_1: u32,
    pub stride_src1_2: u32,
    pub stride_src1_3: u32,
}

pub struct MulMatWebgpu {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    uniform_buffer: wgpu::Buffer,
    mat_src0_buffer: wgpu::Buffer,
    m_size: usize,
    hidden_size: usize,
}

impl MulMatWebgpu {
    // TILE_M and TILE_N are hardcoded as `const` in the WGSL shader (not `override`)
    // because WGSL only allows override-sized arrays in `var<workgroup>` scope,
    // and the per-thread accumulator/register arrays require constructible types.
    // Changing these values here will NOT affect the shader — update the shader consts too.
    pub const TILE_M: usize = 4;
    pub const TILE_N: usize = 4;
    pub const TILE_K: usize = 16;
    pub const WORKGROUP_SIZE_M: usize = 8;
    pub const WORKGROUP_SIZE_N: usize = 4;

    pub fn new<'data>(device: &wgpu::Device, queue: &wgpu::Queue, mat_src0: TensorView<'data>, hidden_size: usize) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mul_mat/shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "wgsl-shaders/mul_mat_reg_tile.wgsl"
            ))),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mul_mat/bind_group_layout"),
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
            label: Some("mul_mat/pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[
                    ("tile_k", Self::TILE_K as f64),
                    ("workgroup_size_m", Self::WORKGROUP_SIZE_M as f64),
                    ("workgroup_size_n", Self::WORKGROUP_SIZE_N as f64),
                ],
                zero_initialize_workgroup_memory: true
            },
            cache: None,
        });
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mul_mat/uniform_buffer"),
            size: std::mem::size_of::<MulMatUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mat_src_u32 = mat_src0.data().chunks_exact(4).map(|chunk| {
            u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
        }).collect::<Vec<u32>>();
        if mat_src_u32.len() * 2 % hidden_size != 0 {
            panic!("The size of matrix src0 must be a multiple of hidden_size");
        }
        let m = mat_src_u32.len() * 2 / hidden_size;
        let mat_src0_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mul_mat/mat_src0_buffer"),
            size: (mat_src_u32.len() * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&mat_src0_buffer, 0, bytemuck::cast_slice(&mat_src_u32));
        Self {
            bind_group_layout,
            pipeline,
            uniform_buffer,
            mat_src0_buffer,
            m_size: m,
            hidden_size,
        }
    }

    pub fn execute(&self, device: &wgpu::Device, queue: &wgpu::Queue, mat_src1_buffer: &wgpu::Buffer, mat_dst_buffer: &wgpu::Buffer, n_rows: usize) {
        let uniform = MulMatUniform {
            offset_src0: 0,
            offset_src1: 0,
            offset_dst: 0,
            m: self.m_size as u32,
            n: n_rows as u32,
            k: self.hidden_size as u32,
            stride_src0_1: self.hidden_size as u32,
            stride_src0_2: self.hidden_size as u32 * self.m_size as u32,
            stride_src0_3: 0,
            stride_src1_1: self.hidden_size as u32,
            stride_src1_2: self.hidden_size as u32 * n_rows as u32,
            stride_src1_3: 0,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniform]));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mul_mat/bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.mat_src0_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: mat_src1_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: mat_dst_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mul_mat/command_encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("mul_mat/compute_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let wg_num_m = (self.m_size + Self::WORKGROUP_SIZE_M * Self::TILE_M - 1) / (Self::WORKGROUP_SIZE_M * Self::TILE_M);
            let wg_num_n = (n_rows + Self::WORKGROUP_SIZE_N * Self::TILE_N - 1) / (Self::WORKGROUP_SIZE_N * Self::TILE_N);
            cpass.dispatch_workgroups((wg_num_m * wg_num_n) as u32, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }
}
