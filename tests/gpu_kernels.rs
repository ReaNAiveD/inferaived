/// Integration tests that compare GPU (WebGPU / WGSL) shader results against
/// CPU reference implementations for every compute kernel in the project.
///
/// Each sub-module covers one WGSL shader / Rust host pair.
///
/// GPU buffer I/O goes through the helpers in `gpu_test_utils` so that the
/// test code never touches raw `map_async` / `get_mapped_range` directly.

mod gpu_test_utils;

use gpu_test_utils::*;

// ---------------------------------------------------------------------------
// 1. elementwise_add
// ---------------------------------------------------------------------------
mod test_elementwise_add {
    use super::*;
    use inferaived::binary::ElementwiseAddInplaceWebgpu;

    /// CPU reference: hidden[t,i] += addend[t,i]
    fn cpu_elementwise_add(hidden: &mut [f32], addend: &[f32], hidden_size: usize, seq_len: usize) {
        for t in 0..seq_len {
            for i in 0..hidden_size {
                hidden[t * hidden_size + i] += addend[t * hidden_size + i];
            }
        }
    }

    #[tokio::test]
    async fn test_elementwise_add() {
        let (device, queue) = gpu_or_skip!();
        let seq_len = 3;
        let hidden_size = 16;
        let hidden: Vec<f32> = (0..seq_len * hidden_size)
            .map(|i| (i as f32) * 0.1)
            .collect();
        let addend: Vec<f32> = (0..seq_len * hidden_size)
            .map(|i| (i as f32) * -0.05 + 1.0)
            .collect();

        // CPU reference
        let mut expected = hidden.clone();
        cpu_elementwise_add(&mut expected, &addend, hidden_size, seq_len);

        // GPU
        let gpu = ElementwiseAddInplaceWebgpu::new(&device, hidden_size);
        let h_buf = upload_f32(&device, &queue, &hidden);
        let a_buf = upload_f32(&device, &queue, &addend);
        gpu.compute(&device, &queue, &h_buf, &a_buf, seq_len);
        let actual = download_f32(&device, &queue, &h_buf, seq_len * hidden_size);

        assert_approx_eq(&actual, &expected, 1e-5);
    }
}

// ---------------------------------------------------------------------------
// 2. rms_norm (out-of-place)
// ---------------------------------------------------------------------------
mod test_rms_norm {
    use super::*;
    use inferaived::norm::RmsNormWebgpu;

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

        // Build bf16 bytes for the TensorView the GPU constructor expects
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

        // Use bf16-round-tripped weight for CPU expected values
        let weight_roundtrip: Vec<f32> = weight_bf16_bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect();
        let expected = cpu_rms_norm(&input, &weight_roundtrip, hidden_size, seq_len, 1e-6);

        let gpu = RmsNormWebgpu::new(&device, &queue, tv, hidden_size);
        let in_buf = upload_f32(&device, &queue, &input);
        let out_buf = create_f32_buffer(&device, seq_len * hidden_size);
        gpu.compute(&device, &queue, &in_buf, &out_buf, seq_len);
        let actual = download_f32(&device, &queue, &out_buf, seq_len * hidden_size);

        assert_approx_eq(&actual, &expected, 1e-4);
    }
}

// ---------------------------------------------------------------------------
// 3. rms_norm_inplace
// ---------------------------------------------------------------------------
mod test_rms_norm_inplace {
    use super::*;
    use inferaived::norm::RmsNormInplaceWebgpu;

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
            let ss: f32 = (0..hidden_size).map(|i| hidden[base + i] * hidden[base + i]).sum();
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
        cpu_rms_norm_inplace(&mut expected, &weight_roundtrip, hidden_size, 0, seq_len, hidden_size, 1e-6);

        let gpu = RmsNormInplaceWebgpu::new(&device, &queue, tv, hidden_size);
        let buf = upload_f32(&device, &queue, &data);
        gpu.compute(&device, &queue, &buf, seq_len);
        let actual = download_f32(&device, &queue, &buf, seq_len * hidden_size);

        assert_approx_eq(&actual, &expected, 1e-4);
    }

    #[tokio::test]
    async fn test_rms_norm_inplace_strided() {
        let (device, queue) = gpu_or_skip!();
        let hidden_size = 16;
        let row_stride = 32; // larger stride (padding between rows)
        let n_rows = 2;
        let offset = 4;
        let total = offset + n_rows * row_stride;
        let data: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.11).sin()).collect();
        let weight_f32: Vec<f32> = (0..hidden_size).map(|i| (i as f32) * 0.02).collect();

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
        cpu_rms_norm_inplace(&mut expected, &weight_roundtrip, hidden_size, offset, n_rows, row_stride, 1e-6);

        let gpu = RmsNormInplaceWebgpu::new(&device, &queue, tv, hidden_size);
        let buf = upload_f32(&device, &queue, &data);
        gpu.compute_strided(&device, &queue, &buf, offset, n_rows, row_stride);
        let actual = download_f32(&device, &queue, &buf, total);

        assert_approx_eq(&actual, &expected, 1e-4);
    }
}

// ---------------------------------------------------------------------------
// 4. silu_mul
// ---------------------------------------------------------------------------
mod test_silu_mul {
    use super::*;
    use inferaived::silu_mul::SiluMulInplaceWebgpu;

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    /// CPU: hidden[t,i] *= silu(gate[t,i])
    fn cpu_silu_mul(hidden: &mut [f32], gate: &[f32], hidden_size: usize, seq_len: usize) {
        for t in 0..seq_len {
            for i in 0..hidden_size {
                let idx = t * hidden_size + i;
                hidden[idx] *= silu(gate[idx]);
            }
        }
    }

    #[tokio::test]
    async fn test_silu_mul() {
        let (device, queue) = gpu_or_skip!();
        let seq_len = 3;
        let hidden_size = 32;
        let hidden: Vec<f32> = (0..seq_len * hidden_size)
            .map(|i| (i as f32) * 0.05 - 1.0)
            .collect();
        let gate: Vec<f32> = (0..seq_len * hidden_size)
            .map(|i| (i as f32) * -0.03 + 0.5)
            .collect();

        let mut expected = hidden.clone();
        cpu_silu_mul(&mut expected, &gate, hidden_size, seq_len);

        let gpu = SiluMulInplaceWebgpu::new(&device, hidden_size);
        let h_buf = upload_f32(&device, &queue, &hidden);
        let g_buf = upload_f32(&device, &queue, &gate);
        gpu.compute(&device, &queue, &h_buf, &g_buf, seq_len);
        let actual = download_f32(&device, &queue, &h_buf, seq_len * hidden_size);

        assert_approx_eq(&actual, &expected, 1e-5);
    }
}

// ---------------------------------------------------------------------------
// 5. sigmoid_mul
// ---------------------------------------------------------------------------
mod test_sigmoid_mul {
    use super::*;
    use inferaived::sigmoid_mul::SigmoidMulInplaceWebgpu;

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// CPU: hidden[t,h,i] *= sigmoid(gate[t,h,i])
    fn cpu_sigmoid_mul(
        hidden: &mut [f32],
        gate: &[f32],
        hidden_size: usize,
        seq_len: usize,
    ) {
        for idx in 0..seq_len * hidden_size {
            hidden[idx] *= sigmoid(gate[idx]);
        }
    }

    #[tokio::test]
    async fn test_sigmoid_mul() {
        let (device, queue) = gpu_or_skip!();
        let seq_len = 2;
        let hidden_size = 32;
        let hidden: Vec<f32> = (0..seq_len * hidden_size)
            .map(|i| (i as f32) * 0.1 - 2.0)
            .collect();
        let gate: Vec<f32> = (0..seq_len * hidden_size)
            .map(|i| (i as f32) * 0.04 - 0.5)
            .collect();

        let mut expected = hidden.clone();
        cpu_sigmoid_mul(&mut expected, &gate, hidden_size, seq_len);

        let gpu = SigmoidMulInplaceWebgpu::new(&device, hidden_size);
        let h_buf = upload_f32(&device, &queue, &hidden);
        let g_buf = upload_f32(&device, &queue, &gate);
        gpu.compute(&device, &queue, &h_buf, &g_buf, seq_len);
        let actual = download_f32(&device, &queue, &h_buf, seq_len * hidden_size);

        assert_approx_eq(&actual, &expected, 1e-5);
    }

    /// CPU strided variant: hidden and gate can have different layouts
    fn cpu_sigmoid_mul_strided(
        hidden: &mut [f32],
        gate: &[f32],
        hidden_offset: usize,
        hidden_token_stride: usize,
        hidden_head_stride: usize,
        gate_offset: usize,
        gate_token_stride: usize,
        gate_head_stride: usize,
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
    ) {
        for t in 0..seq_len {
            for h in 0..num_heads {
                for i in 0..head_dim {
                    let h_idx = hidden_offset + t * hidden_token_stride + h * hidden_head_stride + i;
                    let g_idx = gate_offset + t * gate_token_stride + h * gate_head_stride + i;
                    hidden[h_idx] *= sigmoid(gate[g_idx]);
                }
            }
        }
    }

    #[tokio::test]
    async fn test_sigmoid_mul_strided() {
        let (device, queue) = gpu_or_skip!();
        let seq_len = 2;
        let num_heads = 2;
        let head_dim = 8;
        let hidden_head_stride = head_dim;
        let hidden_token_stride = num_heads * head_dim;
        let total = seq_len * hidden_token_stride;
        let hidden: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let gate: Vec<f32> = (0..total).map(|i| (i as f32) * 0.05).collect();

        let mut expected = hidden.clone();
        cpu_sigmoid_mul_strided(
            &mut expected, &gate,
            0, hidden_token_stride, hidden_head_stride,
            0, hidden_token_stride, hidden_head_stride,
            num_heads, head_dim, seq_len,
        );

        let gpu = SigmoidMulInplaceWebgpu::new(&device, num_heads * head_dim);
        let h_buf = upload_f32(&device, &queue, &hidden);
        let g_buf = upload_f32(&device, &queue, &gate);
        gpu.compute_strided(
            &device, &queue, &h_buf, &g_buf,
            0, hidden_token_stride, hidden_head_stride,
            0, hidden_token_stride, hidden_head_stride,
            num_heads, head_dim, seq_len,
        );
        let actual = download_f32(&device, &queue, &h_buf, total);

        assert_approx_eq(&actual, &expected, 1e-5);
    }
}

// ---------------------------------------------------------------------------
// 6. rope (rotary positional embedding)
// ---------------------------------------------------------------------------
mod test_rope {
    use super::*;
    use inferaived::rope::RopeInplaceWebgpu;

    /// CPU reference for half-split RoPE
    fn cpu_rope(
        q: &mut [f32],
        k: &mut [f32],
        num_q_heads: usize,
        num_k_heads: usize,
        head_dim: usize,
        seq_len: usize,
        theta_scale: f32,
        num_rotated_dims: usize,
        position_offset: usize,
    ) {
        let pair_offset = num_rotated_dims / 2;
        let q_token_stride = num_q_heads * head_dim;
        let k_token_stride = num_k_heads * head_dim;

        for token in 0..seq_len {
            for head in 0..std::cmp::max(num_q_heads, num_k_heads) {
                for pair in 0..pair_offset {
                    let pos = token + position_offset;
                    let theta = pos as f32 * theta_scale.powf(pair as f32);
                    let cos_t = theta.cos();
                    let sin_t = theta.sin();

                    if head < num_q_heads {
                        let a_idx = token * q_token_stride + head * head_dim + pair;
                        let b_idx = token * q_token_stride + head * head_dim + pair + pair_offset;
                        let a = q[a_idx];
                        let b = q[b_idx];
                        q[a_idx] = a * cos_t - b * sin_t;
                        q[b_idx] = a * sin_t + b * cos_t;
                    }
                    if head < num_k_heads {
                        let a_idx = token * k_token_stride + head * head_dim + pair;
                        let b_idx = token * k_token_stride + head * head_dim + pair + pair_offset;
                        let a = k[a_idx];
                        let b = k[b_idx];
                        k[a_idx] = a * cos_t - b * sin_t;
                        k[b_idx] = a * sin_t + b * cos_t;
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn test_rope() {
        let (device, queue) = gpu_or_skip!();
        let seq_len = 3;
        let num_q_heads = 4;
        let num_k_heads = 2;
        let head_dim = 16;
        let rope_theta: f32 = 10000.0;
        let partial_rotary_factor: f32 = 1.0;
        let num_rotated_dims = (head_dim as f32 * partial_rotary_factor).floor() as usize;
        let theta_scale = rope_theta.powf(-2.0 / num_rotated_dims as f32);

        let q: Vec<f32> = (0..seq_len * num_q_heads * head_dim)
            .map(|i| ((i as f32) * 0.1).sin())
            .collect();
        let k: Vec<f32> = (0..seq_len * num_k_heads * head_dim)
            .map(|i| ((i as f32) * 0.07).cos())
            .collect();

        let mut expected_q = q.clone();
        let mut expected_k = k.clone();
        cpu_rope(
            &mut expected_q,
            &mut expected_k,
            num_q_heads,
            num_k_heads,
            head_dim,
            seq_len,
            theta_scale,
            num_rotated_dims,
            0,
        );

        let gpu = RopeInplaceWebgpu::new(
            &device,
            num_q_heads,
            num_k_heads,
            head_dim,
            rope_theta,
            partial_rotary_factor,
        );
        let q_buf = upload_f32(&device, &queue, &q);
        let k_buf = upload_f32(&device, &queue, &k);
        gpu.compute(&device, &queue, &q_buf, &k_buf, seq_len, 0);
        let actual_q = download_f32(&device, &queue, &q_buf, seq_len * num_q_heads * head_dim);
        let actual_k = download_f32(&device, &queue, &k_buf, seq_len * num_k_heads * head_dim);

        assert_approx_eq(&actual_q, &expected_q, 1e-4);
        assert_approx_eq(&actual_k, &expected_k, 1e-4);
    }
}

// ---------------------------------------------------------------------------
// 7. embedding_lookup (get_rows)
// ---------------------------------------------------------------------------
mod test_embedding_lookup {
    use super::*;
    use inferaived::embedding_lookup::EmbeddingLookupWebgpu;

    /// CPU reference: for each token index, gather the row from the bf16
    /// embedding table and convert to f32.
    fn cpu_embedding_lookup(
        embeddings_packed: &[u32],
        indices: &[u32],
        hidden_size: usize,
    ) -> Vec<f32> {
        let row_stride_u32 = hidden_size / 2;
        let mut out = Vec::with_capacity(indices.len() * hidden_size);
        for &idx in indices {
            let base = (idx as usize) * row_stride_u32;
            for i in 0..hidden_size {
                out.push(unpack_bf16(&embeddings_packed, base * 2 + i));
            }
        }
        out
    }

    #[tokio::test]
    async fn test_embedding_lookup() {
        let (device, queue) = gpu_or_skip!();
        let vocab_size = 8;
        let hidden_size = 16;
        // Create fake embeddings in f32, then pack to bf16 bytes
        let embed_f32: Vec<f32> = (0..vocab_size * hidden_size)
            .map(|i| (i as f32) * 0.1 - 3.0)
            .collect();
        let embed_bf16_bytes: Vec<u8> = embed_f32
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect();

        let indices: Vec<u32> = vec![0, 3, 7, 1];

        // CPU reference using bf16 round-trip
        let embed_packed = pack_f32_to_bf16_u32(&embed_f32);
        let expected = cpu_embedding_lookup(&embed_packed, &indices, hidden_size);

        let tv = safetensors::tensor::TensorView::new(
            safetensors::Dtype::BF16,
            vec![vocab_size, hidden_size],
            &embed_bf16_bytes,
        )
        .unwrap();
        let gpu = EmbeddingLookupWebgpu::new(&device, &queue, tv, hidden_size);
        let out_buf = create_f32_buffer(&device, indices.len() * hidden_size);
        gpu.compute(&device, &queue, &indices, &out_buf);
        let actual = download_f32(&device, &queue, &out_buf, indices.len() * hidden_size);

        assert_approx_eq(&actual, &expected, 1e-5);
    }
}

// ---------------------------------------------------------------------------
// 8. mul_mat (matrix multiply with bf16 weights)
// ---------------------------------------------------------------------------
mod test_mul_mat {
    use super::*;
    use inferaived::mul_mat::MulMatWebgpu;

    /// CPU reference: output[n,m] = sum_k weight[m,k] * input[n,k]
    /// weight is stored as packed bf16, input and output are f32.
    /// Output layout: N rows × M columns, M contiguous (column-major from
    /// the math perspective: output[n*M + m]).
    fn cpu_mul_mat(
        weight_packed: &[u32],
        input: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Vec<f32> {
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
        let m = 8;  // output neurons
        let n = 3;  // tokens
        let k = 16; // hidden dim (must be even for bf16 packing)

        // Create weight in f32 then pack to bf16
        let weight_f32: Vec<f32> = (0..m * k)
            .map(|i| ((i as f32) * 0.05).sin())
            .collect();
        let weight_packed = pack_f32_to_bf16_u32(&weight_f32);
        let weight_bf16_bytes: Vec<u8> = weight_f32
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect();

        let input: Vec<f32> = (0..n * k)
            .map(|i| ((i as f32) * 0.07).cos())
            .collect();

        let expected = cpu_mul_mat(&weight_packed, &input, m, n, k);

        let tv = safetensors::tensor::TensorView::new(
            safetensors::Dtype::BF16,
            vec![m, k],
            &weight_bf16_bytes,
        )
        .unwrap();
        let gpu = MulMatWebgpu::new(&device, &queue, tv, k);
        let in_buf = upload_f32(&device, &queue, &input);
        let out_buf = create_f32_buffer(&device, n * m);
        gpu.compute(&device, &queue, &in_buf, &out_buf, n);
        let actual = download_f32(&device, &queue, &out_buf, n * m);

        assert_approx_eq(&actual, &expected, 1e-2);
    }
}

// ---------------------------------------------------------------------------
// 9. slice_copy
// ---------------------------------------------------------------------------
mod test_slice_copy {
    use super::*;
    use inferaived::slice_copy::SliceCopyWebgpu;

    /// CPU reference: copy [seq_len, num_heads, head_dim] with arbitrary strides
    fn cpu_slice_copy(
        src: &[f32],
        dst: &mut [f32],
        src_offset: usize,
        src_token_stride: usize,
        src_head_stride: usize,
        dst_offset: usize,
        dst_token_stride: usize,
        dst_head_stride: usize,
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
    ) {
        for t in 0..seq_len {
            for h in 0..num_heads {
                for i in 0..head_dim {
                    let s = src_offset + t * src_token_stride + h * src_head_stride + i;
                    let d = dst_offset + t * dst_token_stride + h * dst_head_stride + i;
                    dst[d] = src[s];
                }
            }
        }
    }

    #[tokio::test]
    async fn test_slice_copy() {
        let (device, queue) = gpu_or_skip!();
        let seq_len = 2;
        let num_heads = 2;
        let head_dim = 8;
        let src_token_stride = 48; // larger than needed
        let src_head_stride = 16;
        let dst_token_stride = num_heads * head_dim; // packed
        let dst_head_stride = head_dim;
        let src_offset = 4;
        let dst_offset = 0;

        let src_size = src_offset + seq_len * src_token_stride;
        let dst_size = seq_len * dst_token_stride;

        let src: Vec<f32> = (0..src_size).map(|i| (i as f32) * 0.1).collect();
        let mut expected_dst = vec![0.0f32; dst_size];
        cpu_slice_copy(
            &src, &mut expected_dst,
            src_offset, src_token_stride, src_head_stride,
            dst_offset, dst_token_stride, dst_head_stride,
            num_heads, head_dim, seq_len,
        );

        let gpu = SliceCopyWebgpu::new(&device);
        let s_buf = upload_f32(&device, &queue, &src);
        let d_buf = create_f32_buffer(&device, dst_size);
        gpu.compute(
            &device, &queue, &s_buf, &d_buf,
            src_offset, src_token_stride, src_head_stride,
            dst_offset, dst_token_stride, dst_head_stride,
            num_heads, head_dim, seq_len,
        );
        let actual = download_f32(&device, &queue, &d_buf, dst_size);

        assert_approx_eq(&actual, &expected_dst, 1e-6);
    }
}

// ---------------------------------------------------------------------------
// 10. gated_rms_norm
// ---------------------------------------------------------------------------
mod test_gated_rms_norm {
    use super::*;
    use inferaived::gated_rms_norm::GatedRmsNormInplaceWebgpu;

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    /// CPU: hidden[t,h,i] = (hidden[t,h,i] / rms) * weight[i] * silu(gate[t,h,i])
    fn cpu_gated_rms_norm(
        hidden: &mut [f32],
        gate: &[f32],
        weight: &[f32],
        num_heads: usize,
        head_dim: usize,
        seq_len: usize,
        eps: f32,
    ) {
        let token_stride = num_heads * head_dim;
        for t in 0..seq_len {
            for h in 0..num_heads {
                let base = t * token_stride + h * head_dim;
                let ss: f32 = (0..head_dim).map(|i| hidden[base + i] * hidden[base + i]).sum();
                let scale = 1.0 / (ss / head_dim as f32 + eps).sqrt();
                for i in 0..head_dim {
                    let g = gate[base + i];
                    hidden[base + i] = hidden[base + i] * scale * weight[i] * silu(g);
                }
            }
        }
    }

    #[tokio::test]
    async fn test_gated_rms_norm() {
        let (device, queue) = gpu_or_skip!();
        let seq_len = 2;
        let num_heads = 2;
        let head_dim = 16;
        let eps = 1e-6f32;
        let total = seq_len * num_heads * head_dim;

        let hidden: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.09).sin()).collect();
        let gate: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.07).cos()).collect();
        let weight_f32: Vec<f32> = (0..head_dim).map(|i| 1.0 + (i as f32) * 0.01).collect();

        // GatedRmsNormInplaceWebgpu expects F32 TensorView (reads 4-byte chunks)
        let weight_f32_bytes: Vec<u8> = weight_f32
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        let tv = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            vec![head_dim],
            &weight_f32_bytes,
        )
        .unwrap();

        let mut expected = hidden.clone();
        cpu_gated_rms_norm(&mut expected, &gate, &weight_f32, num_heads, head_dim, seq_len, eps);

        let gpu = GatedRmsNormInplaceWebgpu::new(&device, tv, num_heads, head_dim, eps);
        let h_buf = upload_f32(&device, &queue, &hidden);
        let g_buf = upload_f32(&device, &queue, &gate);
        gpu.compute(&device, &queue, &h_buf, &g_buf, seq_len);
        let actual = download_f32(&device, &queue, &h_buf, total);

        assert_approx_eq(&actual, &expected, 1e-3);
    }
}

// ---------------------------------------------------------------------------
// 11. conv_silu (depthwise causal conv + SiLU)
// ---------------------------------------------------------------------------
mod test_conv_silu {
    use super::*;
    use inferaived::conv_silu::{ChannelMode, ConvSiluWebgpu};

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
                        // unpack weight
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

        // Create weight as f32, then pack to bf16 bytes
        let weight_f32: Vec<f32> = (0..nc * kernel_size)
            .map(|i| ((i as f32) * 0.13).sin() * 0.5)
            .collect();
        // Pad to even length if needed
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
            q_dim, k_dim, v_dim,
            seq_len, kernel_size,
            ChannelMode::ConvSilu,
            ChannelMode::ConvSilu,
            ChannelMode::ConvSilu,
            nc, nc,
        );

        let tv = safetensors::tensor::TensorView::new(
            safetensors::Dtype::BF16,
            vec![nc, kernel_size],
            &weight_bf16_bytes,
        )
        .unwrap();
        let gpu = ConvSiluWebgpu::new(
            &device, tv, q_dim, k_dim, v_dim, kernel_size,
            ChannelMode::ConvSilu, ChannelMode::ConvSilu, ChannelMode::ConvSilu,
        );
        let in_buf = upload_f32(&device, &queue, &input);
        let out_buf = create_f32_buffer(&device, seq_len * nc);
        gpu.compute(&device, &queue, &in_buf, &out_buf, seq_len);
        let actual = download_f32(&device, &queue, &out_buf, seq_len * nc);

        assert_approx_eq(&actual, &expected, 1e-2);
    }
}

// ---------------------------------------------------------------------------
// 12. delta_rule
// ---------------------------------------------------------------------------
mod test_delta_rule {
    use super::*;
    use inferaived::delta_rule::DeltaRuleWebgpu;

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

                // L2 norm Q
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

                // L2 norm K
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

                // Gated delta rule per value column
                for vi in 0..value_head_dim {
                    // Decay + retrieve
                    let mut kv_mem = 0.0f32;
                    for ki in 0..key_head_dim {
                        let s_idx = head * state_head_stride + ki * value_head_dim + vi;
                        state[s_idx] *= gamma;
                        let k_val = qkv[k_offset + ki + head * qk_head_stride + token * qkv_token_stride];
                        kv_mem += k_val * state[s_idx];
                    }

                    // Error correction
                    let v_val = qkv[v_offset + vi + head * v_head_stride + token * qkv_token_stride];
                    let delta_v = (v_val - kv_mem) * beta;

                    // State update + output
                    let mut out_acc = 0.0f32;
                    for ki in 0..key_head_dim {
                        let s_idx = head * state_head_stride + ki * value_head_dim + vi;
                        let k_val = qkv[k_offset + ki + head * qk_head_stride + token * qkv_token_stride];
                        state[s_idx] += k_val * delta_v;
                        let q_val = qkv[q_offset + ki + head * qk_head_stride + token * qkv_token_stride];
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
        let a_log: Vec<f32> = (0..num_key_heads).map(|i| -1.0 + (i as f32) * 0.2).collect();

        let state_size = num_key_heads * key_head_dim * value_head_dim;

        // dt_bias is BF16, a_log is F32 in the TensorView
        let dt_bias_bf16: Vec<u8> = dt_bias.iter().flat_map(|&v| half::bf16::from_f32(v).to_le_bytes()).collect();
        let a_log_f32_bytes: Vec<u8> = a_log.iter().flat_map(|&v| v.to_le_bytes()).collect();
        let dt_tv = safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, vec![num_key_heads], &dt_bias_bf16).unwrap();
        let al_tv = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![num_key_heads], &a_log_f32_bytes).unwrap();

        // CPU reference should use bf16-roundtripped dt_bias for accuracy
        let dt_bias_rt: Vec<f32> = dt_bias.iter().map(|&v| half::bf16::from_f32(v).to_f32()).collect();
        let mut cpu_state = vec![0.0f32; state_size];
        let mut cpu_qkv = qkv.clone();
        let expected = cpu_delta_rule(
            &mut cpu_qkv, &proj_a, &proj_b, &dt_bias_rt, &a_log,
            &mut cpu_state, num_key_heads, key_head_dim, value_head_dim, seq_len, eps,
        );

        let gpu = DeltaRuleWebgpu::new(&device, dt_tv, al_tv, num_key_heads, key_head_dim, value_head_dim);
        let qkv_buf = upload_f32(&device, &queue, &qkv);
        let pa_buf = upload_f32(&device, &queue, &proj_a);
        let pb_buf = upload_f32(&device, &queue, &proj_b);
        let state_buf = upload_f32(&device, &queue, &vec![0.0f32; state_size]);
        let out_buf = create_f32_buffer(&device, seq_len * num_key_heads * value_head_dim);
        gpu.compute(&device, &queue, &qkv_buf, &pa_buf, &pb_buf, &state_buf, &out_buf, seq_len);
        let actual = download_f32(&device, &queue, &out_buf, seq_len * num_key_heads * value_head_dim);

        assert_approx_eq(&actual, &expected, 1e-3);
    }
}

// ---------------------------------------------------------------------------
// 13. mamba_scan
// ---------------------------------------------------------------------------
mod test_mamba_scan {
    use super::*;
    use inferaived::mamba_scan::MambaScanWebgpu;

    fn softplus(x: f32) -> f32 {
        if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
    }

    /// CPU reference for mamba_scan SSM
    fn cpu_mamba_scan(
        qkv: &[f32],
        proj_a: &[f32],
        proj_b: &[f32],
        dt_bias: &[f32],
        a_log: &[f32],
        state: &mut [f32],
        num_key_heads: usize,
        key_head_dim: usize,
        value_head_dim: usize,
        seq_len: usize,
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
            let a = -a_log[head].exp();
            let dtb = dt_bias[head];

            for token in 0..seq_len {
                let pa = proj_a[token * num_key_heads + head];
                let dt = softplus(pa + dtb);
                let da = (a * dt).exp();
                let pb = proj_b[token * num_key_heads + head];
                let scale = dt * pb;

                for vi in 0..value_head_dim {
                    let v_val = qkv[v_offset + vi + head * v_head_stride + token * qkv_token_stride];
                    for ki in 0..key_head_dim {
                        let s_idx = head * state_head_stride + ki * value_head_dim + vi;
                        let k_val = qkv[k_offset + ki + head * qk_head_stride + token * qkv_token_stride];
                        state[s_idx] = da * state[s_idx] + scale * k_val * v_val;
                    }
                    let mut acc = 0.0f32;
                    for ki in 0..key_head_dim {
                        let s_idx = head * state_head_stride + ki * value_head_dim + vi;
                        let q_val = qkv[q_offset + ki + head * qk_head_stride + token * qkv_token_stride];
                        acc += q_val * state[s_idx];
                    }
                    output[token * output_token_stride + head * output_head_stride + vi] = acc;
                }
            }
        }
        output
    }

    #[tokio::test]
    async fn test_mamba_scan() {
        let (device, queue) = gpu_or_skip!();
        let num_key_heads = 2;
        let key_head_dim = 8;
        let value_head_dim = 8;
        let seq_len = 3;

        let qkv_token_stride = num_key_heads * key_head_dim * 2 + num_key_heads * value_head_dim;
        let qkv: Vec<f32> = (0..seq_len * qkv_token_stride)
            .map(|i| ((i as f32) * 0.08).sin() * 0.3)
            .collect();
        let proj_a: Vec<f32> = (0..seq_len * num_key_heads)
            .map(|i| (i as f32) * 0.1)
            .collect();
        let proj_b: Vec<f32> = (0..seq_len * num_key_heads)
            .map(|i| (i as f32) * 0.05 + 0.1)
            .collect();
        let dt_bias: Vec<f32> = (0..num_key_heads).map(|i| (i as f32) * 0.1 + 0.5).collect();
        let a_log: Vec<f32> = (0..num_key_heads).map(|i| -1.0 + (i as f32) * 0.3).collect();

        let state_size = num_key_heads * key_head_dim * value_head_dim;

        // dt_bias is BF16, a_log is F32 in the TensorView
        let dt_bias_bf16: Vec<u8> = dt_bias.iter().flat_map(|&v| half::bf16::from_f32(v).to_le_bytes()).collect();
        let a_log_f32_bytes: Vec<u8> = a_log.iter().flat_map(|&v| v.to_le_bytes()).collect();
        let dt_tv = safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, vec![num_key_heads], &dt_bias_bf16).unwrap();
        let al_tv = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![num_key_heads], &a_log_f32_bytes).unwrap();

        // Use bf16-roundtripped dt_bias for CPU reference
        let dt_bias_rt: Vec<f32> = dt_bias.iter().map(|&v| half::bf16::from_f32(v).to_f32()).collect();
        let mut cpu_state = vec![0.0f32; state_size];
        let expected = cpu_mamba_scan(
            &qkv, &proj_a, &proj_b, &dt_bias_rt, &a_log,
            &mut cpu_state, num_key_heads, key_head_dim, value_head_dim, seq_len,
        );

        let gpu = MambaScanWebgpu::new(
            &device, dt_tv, al_tv,
            num_key_heads as u32, key_head_dim as u32, value_head_dim as u32,
        );
        let qkv_buf = upload_f32(&device, &queue, &qkv);
        let pa_buf = upload_f32(&device, &queue, &proj_a);
        let pb_buf = upload_f32(&device, &queue, &proj_b);
        let state_buf = upload_f32(&device, &queue, &vec![0.0f32; state_size]);
        let out_buf = create_f32_buffer(&device, seq_len * num_key_heads * value_head_dim);
        gpu.compute(&device, &queue, &qkv_buf, &pa_buf, &pb_buf, &state_buf, &out_buf, seq_len);
        let actual = download_f32(&device, &queue, &out_buf, seq_len * num_key_heads * value_head_dim);

        assert_approx_eq(&actual, &expected, 1e-3);
    }
}

// ---------------------------------------------------------------------------
// 14. causal_gqa_naive_attention
// ---------------------------------------------------------------------------
mod test_attention {
    use super::*;
    use inferaived::attention::CausalGqaNaiveAttentionWebgpu;

    /// CPU reference for causal grouped-query attention
    fn cpu_causal_gqa_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        num_q_heads: usize,
        num_kv_heads: usize,
        q_dim: usize,
        v_dim: usize,
        seq_len: usize,
    ) -> Vec<f32> {
        let q_token_stride = num_q_heads * q_dim;
        let k_token_stride = num_kv_heads * q_dim;
        let v_token_stride = num_kv_heads * v_dim;
        let output_token_stride = num_q_heads * v_dim;
        let num_q_per_kv = num_q_heads / num_kv_heads;
        let softmax_scale = 1.0 / (q_dim as f32).sqrt();

        let mut output = vec![0.0f32; seq_len * output_token_stride];

        for q_token in 0..seq_len {
            for q_head in 0..num_q_heads {
                let kv_head = q_head / num_q_per_kv;

                // Compute scores for all k_tokens <= q_token
                let mut scores = Vec::with_capacity(q_token + 1);
                for k_token in 0..=q_token {
                    let mut dot = 0.0f32;
                    for d in 0..q_dim {
                        dot += q[q_token * q_token_stride + q_head * q_dim + d]
                            * k[k_token * k_token_stride + kv_head * q_dim + d];
                    }
                    scores.push(dot * softmax_scale);
                }

                // Softmax
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
                let sum_exp: f32 = exp_scores.iter().sum();

                // Weighted sum of V
                for vd in 0..v_dim {
                    let mut acc = 0.0f32;
                    for (k_token, &w) in exp_scores.iter().enumerate() {
                        acc += (w / sum_exp) * v[k_token * v_token_stride + kv_head * v_dim + vd];
                    }
                    output[q_token * output_token_stride + q_head * v_dim + vd] = acc;
                }
            }
        }
        output
    }

    #[tokio::test]
    async fn test_causal_gqa_attention() {
        let (device, queue) = gpu_or_skip!();
        let seq_len = 4;
        let num_q_heads = 4;
        let num_kv_heads = 2;
        let q_dim = 16;
        let v_dim = 16;

        let q: Vec<f32> = (0..seq_len * num_q_heads * q_dim)
            .map(|i| ((i as f32) * 0.05).sin() * 0.3)
            .collect();
        let k: Vec<f32> = (0..seq_len * num_kv_heads * q_dim)
            .map(|i| ((i as f32) * 0.07).cos() * 0.3)
            .collect();
        let v: Vec<f32> = (0..seq_len * num_kv_heads * v_dim)
            .map(|i| ((i as f32) * 0.03).sin() * 0.5)
            .collect();

        let expected = cpu_causal_gqa_attention(
            &q, &k, &v,
            num_q_heads, num_kv_heads, q_dim, v_dim, seq_len,
        );

        let gpu = CausalGqaNaiveAttentionWebgpu::new(&device, num_q_heads, num_kv_heads, q_dim, v_dim);
        let q_buf = upload_f32(&device, &queue, &q);
        let k_buf = upload_f32(&device, &queue, &k);
        let v_buf = upload_f32(&device, &queue, &v);
        let out_buf = create_f32_buffer(&device, seq_len * num_q_heads * v_dim);
        gpu.compute(&device, &queue, &q_buf, &k_buf, &v_buf, &out_buf, seq_len);
        let actual = download_f32(&device, &queue, &out_buf, seq_len * num_q_heads * v_dim);

        assert_approx_eq(&actual, &expected, 1e-3);
    }
}
