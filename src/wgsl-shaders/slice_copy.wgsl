// Copy a per-(token, head) sub-slice from one buffer into a tightly packed
// destination. Each thread copies one f32 element.
//
// TODO (perf): this kernel exists only to keep q_norm/RoPE/attention simple.
// Adding stride-aware variants of those three kernels would let us drop the
// extract pass entirely. See `wgsl-shaders/TODO.md`.

struct Params {
    src_offset: u32,
    src_token_stride: u32,
    src_head_stride: u32,

    dst_offset: u32,
    dst_token_stride: u32,
    dst_head_stride: u32,

    num_heads: u32,
    head_dim: u32,
    seq_len: u32,
}

@group(0) @binding(0)
var<storage, read> src: array<f32>;
@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;
@group(0) @binding(2)
var<uniform> params: Params;

override workgroup_size: u32;

@compute @workgroup_size(workgroup_size)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    if (wg_id.x >= params.seq_len) {
        return;
    }
    let total = params.num_heads * params.head_dim;
    for (var flat: u32 = local_id.x; flat < total; flat += workgroup_size) {
        let head = flat / params.head_dim;
        let i = flat % params.head_dim;
        let s_idx = params.src_offset + wg_id.x * params.src_token_stride + head * params.src_head_stride + i;
        let d_idx = params.dst_offset + wg_id.x * params.dst_token_stride + head * params.dst_head_stride + i;
        dst[d_idx] = src[s_idx];
    }
}
