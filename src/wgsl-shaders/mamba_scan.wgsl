struct Params {
    // Model dimensions
    num_key_heads: u32,
    key_head_dim: u32,
    value_head_dim: u32,
    seq_len: u32,

    // QKV source buffer layout
    q_offset: u32,
    k_offset: u32,
    v_offset: u32,
    qk_head_stride: u32,    // = key_head_dim
    v_head_stride: u32,     // = value_head_dim (may differ from key_head_dim)
    qkv_token_stride: u32,  // = num_key_heads * key_head_dim * 2 + num_value_heads * value_head_dim

    // Projection buffers (per-head scalars)
    proj_a_offset: u32,
    proj_a_token_stride: u32, // = num_key_heads
    proj_b_offset: u32,
    proj_b_token_stride: u32, // = num_key_heads

    // SSM params buffer (dt_bias and A_log packed together)
    dt_bias_offset: u32,
    a_log_offset: u32,

    // Output buffer
    output_token_stride: u32, // = num_key_heads * value_head_dim
    output_head_stride: u32,  // = value_head_dim

    // State buffer
    state_head_stride: u32, // = key_head_dim * value_head_dim
}

@group(0) @binding(0)
var<storage, read> qkv: array<f32>;
@group(0) @binding(1)
var<storage, read> proj_a: array<f32>;
@group(0) @binding(2)
var<storage, read> proj_b: array<f32>;
@group(0) @binding(3)
var<storage, read> ssm_params: array<f32>;  // [0..num_key_heads) for dt bias, [num_key_heads..2*num_key_heads) for A log]
@group(0) @binding(4)
var<storage, read_write> state: array<f32>;
@group(0) @binding(5)
var<storage, read_write> output: array<f32>;
@group(0) @binding(6)
var<uniform> params: Params;

override workgroup_size: u32;

fn softplus(x: f32) -> f32 {
    if (x > 20.0) { return x; }
    return log(1.0 + exp(x));
}

fn get_q(in_head_index: u32, head: u32, token: u32) -> f32 {
    return qkv[params.q_offset + in_head_index + head * params.qk_head_stride + token * params.qkv_token_stride];
}

fn get_k(in_head_index: u32, head: u32, token: u32) -> f32 {
    return qkv[params.k_offset + in_head_index + head * params.qk_head_stride + token * params.qkv_token_stride];
}

fn get_v(in_head_index: u32, head: u32, token: u32) -> f32 {
    return qkv[params.v_offset + in_head_index + head * params.v_head_stride + token * params.qkv_token_stride];
}

fn get_proj_a(head: u32, token: u32) -> f32 {
    return proj_a[params.proj_a_offset + head + token * params.proj_a_token_stride];
}

fn get_proj_b(head: u32, token: u32) -> f32 {
    return proj_b[params.proj_b_offset + head + token * params.proj_b_token_stride];
}

// MAX_KEY_HEAD_DIM defines the max total private state per thread.
// Supports up to floor(MAX_KEY_HEAD_DIM / key_head_dim) columns per thread.
// e.g. key_head_dim=128: 2 columns → workgroup_size >= value_head_dim / 2
const MAX_KEY_HEAD_DIM: u32 = 256u;
var<private> local_state: array<f32, MAX_KEY_HEAD_DIM>;

// Each workgroup computes on one head, and processes multiple tokens in that head.
// Each thread handles one or more value_dim positions (columns of the state matrix).
@compute @workgroup_size(workgroup_size)
fn mamba_scan(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>) {
    if (wg_id.x >= params.num_key_heads) {
        return;
    }
    let head = wg_id.x;
    let a_log = ssm_params[params.a_log_offset + head];
    let a = -exp(a_log);
    let dt_bias = ssm_params[params.dt_bias_offset + head];

    // Initialize local state (load from state buffer; zeroed for prefill)
    for (var value_index = local_id.x; value_index < params.value_head_dim; value_index += workgroup_size) {
        let state_base = (value_index / workgroup_size) * params.key_head_dim;
        for (var k = 0u; k < params.key_head_dim; k++) {
            local_state[state_base + k] = state[head * params.state_head_stride + k * params.value_head_dim + value_index];
        }
    }
    // Handle per token
    for (var token = 0u; token < params.seq_len; token += 1u) {
        let proj_a_val = get_proj_a(head, token);
        let dt = softplus(proj_a_val + dt_bias);
        let da = exp(a * dt);
        let proj_b_val = get_proj_b(head, token);
        let scale = dt * proj_b_val;
        for (var value_index = local_id.x; value_index < params.value_head_dim; value_index += workgroup_size) {
            let state_base = (value_index / workgroup_size) * params.key_head_dim;
            let v_val = get_v(value_index, head, token);
            // State update: h[k][v] = dA * h[k][v] + (dt * b) * K[k] * V[v]
            for (var k = 0u; k < params.key_head_dim; k++) {
                let k_val = get_k(k, head, token);
                local_state[state_base + k] = da * local_state[state_base + k] + scale * k_val * v_val;
            }
            // Output: y[v] = sum_k Q[k] * h[k][v]
            var acc = 0.0;
            for (var k = 0u; k < params.key_head_dim; k++) {
                let q_val = get_q(k, head, token);
                acc += q_val * local_state[state_base + k];
            }
            output[token * params.output_token_stride + head * params.output_head_stride + value_index] = acc;
        }
    }
    // Write back final state
    for (var value_index = local_id.x; value_index < params.value_head_dim; value_index += workgroup_size) {
        let state_base = (value_index / workgroup_size) * params.key_head_dim;
        for (var k = 0u; k < params.key_head_dim; k++) {
            state[head * params.state_head_stride + k * params.value_head_dim + value_index] = local_state[state_base + k];
        }
    }
}
