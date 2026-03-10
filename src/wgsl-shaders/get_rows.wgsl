// WGSL's f16 is IEEE 754 half-precision, but Qwen 3.5 uses bf16 (brain float 16 — same exponent range as f32 but only 8 bits of mantissa). These are different formats.

// Source matrix, stored as a flat array<u32> in row-major order.
// The u32 values are packed pairs of f16 (or bf16) values, so each u32 contains two elements of the logical matrix. The shader will unpack these into f32 in the destination.
// Logical shape: [num_rows, ne0] (e.g. [vocab_size, hidden_size]).
// Row `r` starts at src[offset_src + r * stride_src1] and contains ne0 contiguous f16 values.
@group(0) @binding(0)
var<storage, read> src: array<u32>;
// Index vector, stored as array<i32>.
// Logical shape: [n_rows] (optionally [idx1, idx2] for batched indexing).
// Each element is a row index into src, specifying which row to gather.
// For embedding lookup, these are token IDs from the tokenizer.
@group(0) @binding(1)
var<storage, read> idx: array<i32>;
// Destination matrix, stored as a flat array<f32> in row-major order.
// Logical shape: [n_rows, ne0] (e.g. [n_tokens, hidden_size]).
// Row `i` of dst receives a copy of src row idx[i].
@group(0) @binding(2)
var<storage, read_write> dst: array<f32>;

struct Params {
    offset_src: u32, // in elements
    offset_idx: u32, // in elements
    offset_dst: u32, // in elements

    // Strides (in elements)
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    stride_idx0: u32,
    stride_idx1: u32,
    stride_idx2: u32,

    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,

    // Shape of dst
    ne0: u32,
    n_rows: u32,
    ne2: u32,
    ne3: u32,

    // Shape of idx
    idx1: u32,
    idx2: u32,
};

@group(0) @binding(3)
var<uniform> params: Params;

// Copies one f16 element from src to dst, given the base addresses and the offset in elements.
fn copy_elements(src_base: u32, dst_base: u32, offset: u32) {
    let packed = src[src_base + offset / 2u];
    if (offset % 2 == 0u) {
        // Even offset: lower 16 bits contain a bf16 value.
        // bf16 is the upper 16 bits of an f32, so shift left by 16.
        let bf16_bits = packed & 0xFFFFu;
        dst[dst_base + offset] = bitcast<f32>(bf16_bits << 16u);
    } else {
        // Odd offset: upper 16 bits contain a bf16 value.
        // Already in the upper half, just mask off the lower bits.
        let bf16_bits = packed & 0xFFFF0000u;
        dst[dst_base + offset] = bitcast<f32>(bf16_bits);
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= params.n_rows * params.ne2 * params.ne3) {
        return;
    }
    let idx_dst3 = global_id.x / (params.n_rows * params.ne2);
    let idx_dst2 = (global_id.x % (params.n_rows * params.ne2)) / params.n_rows;
    let idx_dst1 = global_id.x % params.n_rows;

    let idx_idx2 = idx_dst3 % params.idx2;
    let idx_idx1 = idx_dst2 % params.idx1;
    let idx_idx0 = idx_dst1;

    let idx_idx = params.offset_idx
        + idx_idx0 * params.stride_idx0
        + idx_idx1 * params.stride_idx1
        + idx_idx2 * params.stride_idx2;
    
    let idx_val = u32(idx[idx_idx]);

    let idx_src = params.offset_src
        + idx_val * params.stride_src1
        + idx_dst2 * params.stride_src2
        + idx_dst3 * params.stride_src3;
    let dst_idx = params.offset_dst
        + idx_dst1 * params.stride_dst1
        + idx_dst2 * params.stride_dst2
        + idx_dst3 * params.stride_dst3;

    for (var i: u32 = 0u; i < params.ne0; i = i + 1u) {
        copy_elements(idx_src, dst_idx, i);
    }
}
