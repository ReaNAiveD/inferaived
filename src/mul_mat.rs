use safetensors::tensor::TensorView;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MulMatParams {
    pub weight_offset: u32,
    pub input_offset: u32,
    pub output_offset: u32,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub weight_row_stride: u32,
    pub input_row_stride: u32,
}

pub struct MulMatWebgpu {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    uniform_buffer: wgpu::Buffer,
    mat_src0_buffer: wgpu::Buffer,
    m_dim: usize,
    k_dim: usize,
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

    pub fn new<'data>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mat_src0: TensorView<'data>,
    ) -> Self {
        debug_assert_eq!(
            mat_src0.shape().len(),
            2,
            "MulMatWebgpu weight must be 2-D, got shape {:?}",
            mat_src0.shape(),
        );
        let m_dim = mat_src0.shape()[0] as usize;
        let k_dim = mat_src0.shape()[1] as usize;
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
                zero_initialize_workgroup_memory: true,
            },
            cache: None,
        });
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mul_mat/uniform_buffer"),
            size: std::mem::size_of::<MulMatParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mat_src_u32 = mat_src0
            .data()
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<u32>>();
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
            m_dim,
            k_dim,
        }
    }

    pub fn compute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mat_src1_buffer: &wgpu::Buffer,
        mat_dst_buffer: &wgpu::Buffer,
        n_dim: usize,
    ) {
        let uniform = MulMatParams {
            weight_offset: 0,
            input_offset: 0,
            output_offset: 0,
            m: self.m_dim as u32,
            n: n_dim as u32,
            k: self.k_dim as u32,
            weight_row_stride: self.k_dim as u32,
            input_row_stride: self.k_dim as u32,
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
            let wg_num_m = (self.m_dim + Self::WORKGROUP_SIZE_M * Self::TILE_M - 1)
                / (Self::WORKGROUP_SIZE_M * Self::TILE_M);
            let wg_num_n = (n_dim + Self::WORKGROUP_SIZE_N * Self::TILE_N - 1)
                / (Self::WORKGROUP_SIZE_N * Self::TILE_N);
            cpass.dispatch_workgroups((wg_num_m * wg_num_n) as u32, 1, 1);
        }
        queue.submit(Some(encoder.finish()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_test_utils::*;

    /// CPU reference: output[n,m] = sum_k weight[m,k] * input[n,k]
    fn cpu_mul_mat(weight_packed: &[u32], input: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n * m];
        for row_n in 0..n {
            for col_m in 0..m {
                let mut acc = 0.0f32;
                for ki in 0..k {
                    let w_idx = col_m * k + ki;
                    let w_val = unpack_bf16(weight_packed, w_idx);
                    let i_val = input[row_n * k + ki];
                    acc += w_val * i_val;
                }
                out[row_n * m + col_m] = acc;
            }
        }
        out
    }

    #[tokio::test]
    async fn test_mul_mat() {
        let (device, queue) = gpu_or_skip!();
        let m = 8;
        let n = 3;
        let k = 16;

        let weight_f32: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.05).sin()).collect();
        let weight_packed = pack_f32_to_bf16_u32(&weight_f32);
        let weight_bf16_bytes: Vec<u8> = weight_f32
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect();

        let input: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.07).cos()).collect();

        let expected = cpu_mul_mat(&weight_packed, &input, m, n, k);

        let tv = safetensors::tensor::TensorView::new(
            safetensors::Dtype::BF16,
            vec![m, k],
            &weight_bf16_bytes,
        )
        .unwrap();
        let gpu = MulMatWebgpu::new(&device, &queue, tv);
        let in_buf = upload_f32(&device, &input);
        let out_buf = create_f32_buffer(&device, n * m);
        gpu.compute(&device, &queue, &in_buf, &out_buf, n);
        let actual = download_f32(&device, &queue, &out_buf, n * m);

        assert_approx_eq(&actual, &expected, 1e-2);
    }
}
