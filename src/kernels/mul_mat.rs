use safetensors::tensor::TensorView;

use crate::buffer_view::BufferView;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MulMatParams {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub weight_row_stride: u32, // bf16 elements
    pub input_row_stride: u32,  // f32 elements (from view)
}

pub struct MulMatWebgpu {
    bind_group_layout: wgpu::BindGroupLayout,
    // Selected by `compute` on N: reg_tile for N > 1, vec for N == 1.
    pipeline_reg_tile: wgpu::ComputePipeline,
    pipeline_vec: wgpu::ComputePipeline,
    // Decode-optimised GEMV pipeline (N == 1 only).  Present when the device
    // supports wgpu::Features::SUBGROUP.  Supersedes pipeline_vec when set.
    pipeline_decode: Option<wgpu::ComputePipeline>,
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
    pub const WORKGROUP_SIZE_VEC: usize = 256;
    // Must match ROWS_PER_WG in mul_mat_vec_decode.wgsl.
    pub const ROWS_PER_WG_DECODE: usize = 4;
    pub const WORKGROUP_SIZE_DECODE: usize = 128;

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
        let reg_tile_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mul_mat/reg_tile_shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "wgsl-shaders/mul_mat_reg_tile.wgsl"
            ))),
        });
        let vec_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("mul_mat/vec_shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "wgsl-shaders/mul_mat_vec.wgsl"
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
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline_reg_tile = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mul_mat/pipeline_reg_tile"),
            layout: Some(&pipeline_layout),
            module: &reg_tile_shader,
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
        let pipeline_vec = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mul_mat/pipeline_vec"),
            layout: Some(&pipeline_layout),
            module: &vec_shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("workgroup_size", Self::WORKGROUP_SIZE_VEC as f64)],
                zero_initialize_workgroup_memory: true,
            },
            cache: None,
        });
        // Created only when the device exposes Features::SUBGROUP.
        let pipeline_decode = if device
            .features()
            .contains(wgpu::Features::SUBGROUP)
        {
            let decode_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mul_mat/decode_shader"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                    "wgsl-shaders/mul_mat_vec_decode.wgsl"
                ))),
            });
            Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("mul_mat/pipeline_decode"),
                layout: Some(&pipeline_layout),
                module: &decode_shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[(
                        "workgroup_size",
                        Self::WORKGROUP_SIZE_DECODE as f64,
                    )],
                    zero_initialize_workgroup_memory: true,
                },
                cache: None,
            }))
        } else {
            None
        };
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mul_mat/uniform_buffer"),
            size: std::mem::size_of::<MulMatParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mat_src_bytes = mat_src0.data();
        // wgpu validates `write_buffer` lengths against COPY_BUFFER_ALIGNMENT
        // (= 4) at runtime, even though the Rust API doc only mentions
        // in-bounds. bf16 is 2 bytes, so an odd element count leaves a 2-byte
        // tail that has to be padded to a u32 word in a second write.
        let aligned_len = mat_src_bytes.len() & !3;
        let tail_len = mat_src_bytes.len() - aligned_len; // 0 or 2
        let padded_size = aligned_len + if tail_len == 0 { 0 } else { 4 };
        let mat_src0_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mul_mat/mat_src0_buffer"),
            size: padded_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if aligned_len > 0 {
            queue.write_buffer(&mat_src0_buffer, 0, &mat_src_bytes[..aligned_len]);
        }
        if tail_len > 0 {
            let mut tail = [0u8; 4];
            tail[..tail_len].copy_from_slice(&mat_src_bytes[aligned_len..]);
            queue.write_buffer(&mat_src0_buffer, aligned_len as u64, &tail);
        }
        Self {
            bind_group_layout,
            pipeline_reg_tile,
            pipeline_vec,
            pipeline_decode,
            uniform_buffer,
            mat_src0_buffer,
            m_dim,
            k_dim,
        }
    }

    pub fn forward(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: BufferView<'_>,
        dst: BufferView<'_>,
    ) {
        debug_assert_eq!(
            input.shape[0], dst.shape[0],
            "mul_mat: outer dim mismatch (input={}, dst={})",
            input.shape[0], dst.shape[0],
        );
        debug_assert_eq!(
            dst.stride[0] as usize, self.m_dim,
            "mul_mat: dst must be tight-packed at m={} (stride[0]={} elements)",
            self.m_dim, dst.stride[0],
        );
        let num_rows = input.shape[0];
        let uniform = MulMatParams {
            m: self.m_dim as u32,
            n: num_rows,
            k: self.k_dim as u32,
            weight_row_stride: self.k_dim as u32,
            input_row_stride: input.stride[0],
        };
        let (pipeline, workgroup_count, variant) = if num_rows == 1 {
            if let Some(ref p) = self.pipeline_decode {
                let wg_count = (self.m_dim + Self::ROWS_PER_WG_DECODE - 1)
                    / Self::ROWS_PER_WG_DECODE;
                (p, wg_count as u32, "decode")
            } else {
                (&self.pipeline_vec, self.m_dim as u32, "vec")
            }
        } else {
            let wg_num_m = (self.m_dim + Self::WORKGROUP_SIZE_M * Self::TILE_M - 1)
                / (Self::WORKGROUP_SIZE_M * Self::TILE_M);
            let wg_num_n = (num_rows as usize + Self::WORKGROUP_SIZE_N * Self::TILE_N - 1)
                / (Self::WORKGROUP_SIZE_N * Self::TILE_N);
            (
                &self.pipeline_reg_tile,
                (wg_num_m * wg_num_n) as u32,
                "reg_tile",
            )
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
                    resource: input.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dst.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("mul_mat/command_encoder_{}", variant)),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("mul_mat/compute_pass_{}", variant)),
                timestamp_writes: None,
            });
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(workgroup_count, 1, 1);
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
        let sz = std::mem::size_of::<f32>() as u32;
        gpu.forward(
            &device,
            &queue,
            BufferView::new_2d_tight(&in_buf, n as u32, k as u32, sz),
            BufferView::new_2d_tight(&out_buf, n as u32, m as u32, sz),
        );
        let actual = download_f32(&device, &queue, &out_buf, n * m);

        assert_approx_eq(&actual, &expected, 1e-2);
    }

    /// `compute` with n_dim == 1 must take the matvec path and still match
    /// the CPU reference.
    #[tokio::test]
    async fn test_mul_mat_vec_n1() {
        let (device, queue) = gpu_or_skip!();
        // Use a larger M and K than the default test to actually exercise
        // the workgroup-wide K-split + tree reduction.
        let m = 320;
        let n = 1;
        let k = 512;

        let weight_f32: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.013).sin()).collect();
        let weight_packed = pack_f32_to_bf16_u32(&weight_f32);
        let weight_bf16_bytes: Vec<u8> = weight_f32
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect();
        let input: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.021).cos()).collect();

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
        let sz = std::mem::size_of::<f32>() as u32;
        gpu.forward(
            &device,
            &queue,
            BufferView::new_2d_tight(&in_buf, n as u32, k as u32, sz),
            BufferView::new_2d_tight(&out_buf, n as u32, m as u32, sz),
        );
        let actual = download_f32(&device, &queue, &out_buf, n * m);

        assert_approx_eq(&actual, &expected, 1e-2);
    }

    /// Matvec must handle odd K (and the resulting odd row alignment) — the
    /// shader resolves bf16 lo/hi per element rather than iterating in
    /// u32-aligned pairs, so no caller-side padding is required.
    #[tokio::test]
    async fn test_mul_mat_vec_odd_k() {
        let (device, queue) = gpu_or_skip!();
        let m = 7;
        let n = 1;
        let k = 13;

        let weight_f32: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.041).sin()).collect();
        let weight_packed = pack_f32_to_bf16_u32(&weight_f32);
        let weight_bf16_bytes: Vec<u8> = weight_f32
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect();
        let input: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.037).cos()).collect();

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
        let sz = std::mem::size_of::<f32>() as u32;
        gpu.forward(
            &device,
            &queue,
            BufferView::new_2d_tight(&in_buf, n as u32, k as u32, sz),
            BufferView::new_2d_tight(&out_buf, n as u32, m as u32, sz),
        );
        let actual = download_f32(&device, &queue, &out_buf, n * m);

        assert_approx_eq(&actual, &expected, 1e-2);
    }

    /// Same as `forward` but called with `src_start_row = dst_start_row =
    /// n - 1, num_rows = 1`: read only the last input row and write only
    /// the last output slot. Other output rows must be left untouched.
    ///
    /// `m` and `k` are chosen so that `m * 4` and `k * 4` are both multiples
    /// of `min_storage_buffer_offset_alignment` (256), which the binding-side
    /// offset encoding requires for non-zero start rows.
    #[tokio::test]
    async fn test_mul_mat_forward_last_row() {
        let (device, queue) = gpu_or_skip!();
        let m = 128;
        let n = 4;
        let k = 128;

        let weight_f32: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.017).sin()).collect();
        let weight_packed = pack_f32_to_bf16_u32(&weight_f32);
        let weight_bf16_bytes: Vec<u8> = weight_f32
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect();
        let input: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.029).cos()).collect();

        // Reference: full matmul. We only compare the last row of it.
        let expected_full = cpu_mul_mat(&weight_packed, &input, m, n, k);
        let expected_last = &expected_full[(n - 1) * m..n * m];

        let tv = safetensors::tensor::TensorView::new(
            safetensors::Dtype::BF16,
            vec![m, k],
            &weight_bf16_bytes,
        )
        .unwrap();
        let gpu = MulMatWebgpu::new(&device, &queue, tv);
        let in_buf = upload_f32(&device, &input);

        // Pre-fill output with a sentinel so we can verify the earlier rows
        // were not overwritten.
        let sentinel: Vec<f32> = vec![-12345.0; n * m];
        let out_buf = upload_f32(&device, &sentinel);

        gpu.forward(
            &device,
            &queue,
            BufferView::new_2d_tight(
                &in_buf,
                n as u32,
                k as u32,
                std::mem::size_of::<f32>() as u32,
            )
            .narrow(0, (n - 1) as u32, 1),
            BufferView::new_2d_tight(
                &out_buf,
                n as u32,
                m as u32,
                std::mem::size_of::<f32>() as u32,
            )
            .narrow(0, (n - 1) as u32, 1),
        );
        let actual = download_f32(&device, &queue, &out_buf, n * m);

        // Last row matches the reference.
        assert_approx_eq(&actual[(n - 1) * m..n * m], expected_last, 1e-2);
        // All earlier rows are untouched.
        for row in 0..n - 1 {
            for col in 0..m {
                assert_eq!(
                    actual[row * m + col],
                    sentinel[row * m + col],
                    "row {} col {} was modified",
                    row,
                    col,
                );
            }
        }
    }

    // ── Decode-tuned pipeline tests ────────────────────────────────────────
    // These tests require a device with Features::SUBGROUP enabled; they are
    // skipped if the adapter does not expose that feature.

    /// Create a (device, queue) pair with SUBGROUP support requested, or return
    /// None if the adapter does not expose the feature.
    async fn create_subgroup_device_queue() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok()?;
        if !adapter.features().contains(wgpu::Features::SUBGROUP) {
            eprintln!("Adapter does not support SUBGROUP — skipping decode test");
            return None;
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("test_device_subgroup"),
                required_features: wgpu::Features::SUBGROUP,
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .ok()?;
        Some((device, queue))
    }

    macro_rules! subgroup_or_skip {
        () => {
            match create_subgroup_device_queue().await {
                Some(dq) => dq,
                None => return,
            }
        };
    }

    /// Verify the decode pipeline matches the CPU reference for a typical
    /// decode workload: M = 320, K = 512, N = 1.
    #[tokio::test]
    async fn test_mul_mat_vec_decode_n1() {
        let (device, queue) = subgroup_or_skip!();
        let m = 320;
        let n = 1;
        let k = 512;

        let weight_f32: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.013).sin()).collect();
        let weight_packed = pack_f32_to_bf16_u32(&weight_f32);
        let weight_bf16_bytes: Vec<u8> = weight_f32
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect();
        let input: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.021).cos()).collect();

        let expected = cpu_mul_mat(&weight_packed, &input, m, n, k);

        let tv = safetensors::tensor::TensorView::new(
            safetensors::Dtype::BF16,
            vec![m, k],
            &weight_bf16_bytes,
        )
        .unwrap();
        let gpu = MulMatWebgpu::new(&device, &queue, tv);

        // The decode pipeline should be present when SUBGROUP is enabled.
        assert!(
            gpu.pipeline_decode.is_some(),
            "pipeline_decode should be Some when SUBGROUP feature is available"
        );

        let in_buf = upload_f32(&device, &input);
        let out_buf = create_f32_buffer(&device, n * m);
        let sz = std::mem::size_of::<f32>() as u32;
        gpu.forward(
            &device,
            &queue,
            BufferView::new_2d_tight(&in_buf, n as u32, k as u32, sz),
            BufferView::new_2d_tight(&out_buf, n as u32, m as u32, sz),
        );
        let actual = download_f32(&device, &queue, &out_buf, n * m);

        assert_approx_eq(&actual, &expected, 1e-2);
    }

    /// Decode pipeline with M not a multiple of ROWS_PER_WG (boundary handling).
    #[tokio::test]
    async fn test_mul_mat_vec_decode_non_multiple_m() {
        let (device, queue) = subgroup_or_skip!();
        // m = 13: not a multiple of ROWS_PER_WG (4), exercises the partial
        // last workgroup guard (global_row < params.m).
        let m = 13;
        let n = 1;
        let k = 64;

        let weight_f32: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.031).sin()).collect();
        let weight_packed = pack_f32_to_bf16_u32(&weight_f32);
        let weight_bf16_bytes: Vec<u8> = weight_f32
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect();
        let input: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.047).cos()).collect();

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
        let sz = std::mem::size_of::<f32>() as u32;
        gpu.forward(
            &device,
            &queue,
            BufferView::new_2d_tight(&in_buf, n as u32, k as u32, sz),
            BufferView::new_2d_tight(&out_buf, n as u32, m as u32, sz),
        );
        let actual = download_f32(&device, &queue, &out_buf, n * m);

        assert_approx_eq(&actual, &expected, 1e-2);
    }

    /// Decode pipeline with odd K (the last u32 word's hi bf16 lane should be
    /// zero-masked by the `k1 < params.k` guard on the input side and the
    /// zero-pad on the weight side).
    #[tokio::test]
    async fn test_mul_mat_vec_decode_odd_k() {
        let (device, queue) = subgroup_or_skip!();
        let m = 8;
        let n = 1;
        let k = 13; // odd — exercises the k%2 edge case

        let weight_f32: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.041).sin()).collect();
        let weight_packed = pack_f32_to_bf16_u32(&weight_f32);
        let weight_bf16_bytes: Vec<u8> = weight_f32
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect();
        let input: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.037).cos()).collect();

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
        let sz = std::mem::size_of::<f32>() as u32;
        gpu.forward(
            &device,
            &queue,
            BufferView::new_2d_tight(&in_buf, n as u32, k as u32, sz),
            BufferView::new_2d_tight(&out_buf, n as u32, m as u32, sz),
        );
        let actual = download_f32(&device, &queue, &out_buf, n * m);

        assert_approx_eq(&actual, &expected, 1e-2);
    }

    /// Decode pipeline with large K (exercises multiple tile iterations and
    /// the cross-subgroup reduction).  Uses model-realistic dimensions:
    /// M = 1024, K = 3584 (MLP down-projection in the 0.8B Qwen3.5 variant).
    #[tokio::test]
    async fn test_mul_mat_vec_decode_large() {
        let (device, queue) = subgroup_or_skip!();
        let m = 128; // keep small so the test runs quickly; K exercises the tiles
        let n = 1;
        let k = 3584;

        let weight_f32: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.003).sin()).collect();
        let weight_packed = pack_f32_to_bf16_u32(&weight_f32);
        let weight_bf16_bytes: Vec<u8> = weight_f32
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect();
        let input: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.007).cos()).collect();

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
        let sz = std::mem::size_of::<f32>() as u32;
        gpu.forward(
            &device,
            &queue,
            BufferView::new_2d_tight(&in_buf, n as u32, k as u32, sz),
            BufferView::new_2d_tight(&out_buf, n as u32, m as u32, sz),
        );
        let actual = download_f32(&device, &queue, &out_buf, n * m);

        assert_approx_eq(&actual, &expected, 1e-2);
    }
}
