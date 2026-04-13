// The shader currently implement NeoX-style RoPE without mRoPE.
// The shader does the replication in-place.

struct Params {
    q_offset: u32,
    k_offset: u32,
    stride_token: u32,
    stride_head: u32,
    head_dim: u32,
    num_heads: u32,
    seq_len: u32,
    n_dims: u32,
    theta_scale: f32,
    pos_offset: u32,
}

@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

@group(0) @binding(1)
var<uniform> params: Params;

@compute @workgroup_size(256)
fn rope(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pair_offset = params.n_dims / 2u;
    if (global_id.x >= pair_offset * params.num_heads * params.seq_len) {
        return;
    }
    let seq_num = global_id.x / (pair_offset * params.num_heads);
    let head_num = (global_id.x % (pair_offset * params.num_heads)) / pair_offset;
    let pair_num = global_id.x % pair_offset;
    let q_pair_a_offset = params.q_offset + seq_num * params.stride_token + head_num * params.stride_head + pair_num;
    let q_pair_b_offset = q_pair_a_offset + pair_offset;
    let k_pair_a_offset = params.k_offset + seq_num * params.stride_token + head_num * params.stride_head + pair_num;
    let k_pair_b_offset = k_pair_a_offset + pair_offset;

    let pos = seq_num + params.pos_offset;
    let theta = f32(pos) * pow(params.theta_scale, f32(pair_num));
    let cos_theta = cos(theta);
    let sin_theta = sin(theta);
    let q_a = src[q_pair_a_offset];
    let q_b = src[q_pair_b_offset];
    let new_q_a = q_a * cos_theta - q_b * sin_theta;
    let new_q_b = q_a * sin_theta + q_b * cos_theta;
    src[q_pair_a_offset] = new_q_a;
    src[q_pair_b_offset] = new_q_b;
    let k_a = src[k_pair_a_offset];
    let k_b = src[k_pair_b_offset];
    let new_k_a = k_a * cos_theta - k_b * sin_theta;
    let new_k_b = k_a * sin_theta + k_b * cos_theta;
    src[k_pair_a_offset] = new_k_a;
    src[k_pair_b_offset] = new_k_b;
}
