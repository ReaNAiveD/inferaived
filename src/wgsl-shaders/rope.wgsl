// Half-split (HF / Llama style) RoPE applied in-place to Q and K.
//
// Q and K live in distinct buffers with potentially different head counts
// (GQA: num_q_heads >= num_k_heads). Each thread handles one (token, head, pair)
// triple and rotates Q if `head < num_q_heads` and/or K if `head < num_k_heads`.
//
// Position layout is 1D (`token + position_offset`). 3D MRoPE is not yet
// supported and is bit-exact equivalent to 1D for text-only inputs.

struct Params {
    q_offset: u32,
    q_token_stride: u32,
    q_head_stride: u32,
    k_offset: u32,
    k_token_stride: u32,
    k_head_stride: u32,

    num_q_heads: u32,
    num_k_heads: u32,
    seq_len: u32,
    num_rotated_dims: u32,

    theta_scale: f32,
    position_offset: u32,
}

@group(0) @binding(0)
var<storage, read_write> q: array<f32>;
@group(0) @binding(1)
var<storage, read_write> k: array<f32>;
@group(0) @binding(2)
var<uniform> params: Params;

fn q_index(token: u32, head: u32, i: u32) -> u32 {
    return params.q_offset + token * params.q_token_stride + head * params.q_head_stride + i;
}

fn k_index(token: u32, head: u32, i: u32) -> u32 {
    return params.k_offset + token * params.k_token_stride + head * params.k_head_stride + i;
}

@compute @workgroup_size(256)
fn rope(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pair_offset = params.num_rotated_dims / 2u;
    let max_heads = max(params.num_q_heads, params.num_k_heads);
    if (global_id.x >= pair_offset * max_heads * params.seq_len) {
        return;
    }
    let token = global_id.x / (pair_offset * max_heads);
    let head = (global_id.x % (pair_offset * max_heads)) / pair_offset;
    let pair = global_id.x % pair_offset;

    let pos = token + params.position_offset;
    let theta = f32(pos) * pow(params.theta_scale, f32(pair));
    let cos_theta = cos(theta);
    let sin_theta = sin(theta);

    if (head < params.num_q_heads) {
        let a_idx = q_index(token, head, pair);
        let b_idx = q_index(token, head, pair + pair_offset);
        let a = q[a_idx];
        let b = q[b_idx];
        q[a_idx] = a * cos_theta - b * sin_theta;
        q[b_idx] = a * sin_theta + b * cos_theta;
    }
    if (head < params.num_k_heads) {
        let a_idx = k_index(token, head, pair);
        let b_idx = k_index(token, head, pair + pair_offset);
        let a = k[a_idx];
        let b = k[b_idx];
        k[a_idx] = a * cos_theta - b * sin_theta;
        k[b_idx] = a * sin_theta + b * cos_theta;
    }
}
