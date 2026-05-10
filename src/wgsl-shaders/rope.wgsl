// The shader currently implements NeoX-style RoPE without mRoPE.
// Q and K live in the same buffer (`qk`) at distinct offsets and share the
// same per-token / per-head stride. The shader rotates them in place.

struct Params {
    q_offset: u32,
    k_offset: u32,
    token_stride: u32,
    head_stride: u32,
    head_dim: u32,
    num_heads: u32,
    seq_len: u32,
    num_rotated_dims: u32,
    theta_scale: f32,
    position_offset: u32,
}

@group(0) @binding(0)
var<storage, read_write> qk: array<f32>;

@group(0) @binding(1)
var<uniform> params: Params;

@compute @workgroup_size(256)
fn rope(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pair_offset = params.num_rotated_dims / 2u;
    if (global_id.x >= pair_offset * params.num_heads * params.seq_len) {
        return;
    }
    let token = global_id.x / (pair_offset * params.num_heads);
    let head = (global_id.x % (pair_offset * params.num_heads)) / pair_offset;
    let pair = global_id.x % pair_offset;
    let q_pair_a = params.q_offset + token * params.token_stride + head * params.head_stride + pair;
    let q_pair_b = q_pair_a + pair_offset;
    let k_pair_a = params.k_offset + token * params.token_stride + head * params.head_stride + pair;
    let k_pair_b = k_pair_a + pair_offset;

    let pos = token + params.position_offset;
    let theta = f32(pos) * pow(params.theta_scale, f32(pair));
    let cos_theta = cos(theta);
    let sin_theta = sin(theta);
    let q_a = qk[q_pair_a];
    let q_b = qk[q_pair_b];
    qk[q_pair_a] = q_a * cos_theta - q_b * sin_theta;
    qk[q_pair_b] = q_a * sin_theta + q_b * cos_theta;
    let k_a = qk[k_pair_a];
    let k_b = qk[k_pair_b];
    qk[k_pair_a] = k_a * cos_theta - k_b * sin_theta;
    qk[k_pair_b] = k_a * sin_theta + k_b * cos_theta;
}
