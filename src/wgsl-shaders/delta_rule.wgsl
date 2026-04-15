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
    stride_qk_head: u32,   // = key_head_dim
    stride_v_head: u32,    // = value_head_dim (may differ from key_head_dim)
    stride_qkv_token: u32, // = num_key_heads * key_head_dim * 2 + num_value_heads * value_head_dim

    // Projection buffers (per-head scalars)
    proj_a_offset: u32,
    stride_proj_a_token: u32, // = num_key_heads
    proj_b_offset: u32,
    stride_proj_b_token: u32, // = num_key_heads

    // Gate params buffer (dt_bias and A_log packed together)
    dt_bias_offset: u32,
    a_log_offset: u32,

    // Output buffer
    stride_dst_token: u32, // = num_key_heads * value_head_dim
    stride_dst_head: u32,  // = value_head_dim

    // State buffer
    stride_state_head: u32, // = key_head_dim * value_head_dim

    eps: f32,
}

@group(0) @binding(0)
var<storage, read_write> qkv_src: array<f32>;
@group(0) @binding(1)
var<storage, read> proj_a: array<f32>;
@group(0) @binding(2)
var<storage, read> proj_b: array<f32>;
@group(0) @binding(3)
var<storage, read> gate_params: array<f32>;  // [0..num_key_heads) dt_bias, [num_key_heads..2*num_key_heads) A_log
@group(0) @binding(4)
var<storage, read_write> state: array<f32>;
@group(0) @binding(5)
var<storage, read_write> dst: array<f32>;
@group(0) @binding(6)
var<uniform> params: Params;

override workgroup_size: u32;

fn softplus(x: f32) -> f32 {
    if (x > 20.0) { return x; }
    return log(1.0 + exp(x));
}

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

fn get_q(in_head_index: u32, head_num: u32, token_idx: u32) -> f32 {
    return qkv_src[params.q_offset + in_head_index + head_num * params.stride_qk_head + token_idx * params.stride_qkv_token];
}

fn set_q(in_head_index: u32, head_num: u32, token_idx: u32, value: f32) {
    qkv_src[params.q_offset + in_head_index + head_num * params.stride_qk_head + token_idx * params.stride_qkv_token] = value;
}

fn get_k(in_head_index: u32, head_num: u32, token_idx: u32) -> f32 {
    return qkv_src[params.k_offset + in_head_index + head_num * params.stride_qk_head + token_idx * params.stride_qkv_token];
}

fn set_k(in_head_index: u32, head_num: u32, token_idx: u32, value: f32) {
    qkv_src[params.k_offset + in_head_index + head_num * params.stride_qk_head + token_idx * params.stride_qkv_token] = value;
}

fn get_v(in_head_index: u32, head_num: u32, token_idx: u32) -> f32 {
    return qkv_src[params.v_offset + in_head_index + head_num * params.stride_v_head + token_idx * params.stride_qkv_token];
}

fn get_proj_a(head_num: u32, token_idx: u32) -> f32 {
    return proj_a[params.proj_a_offset + head_num + token_idx * params.stride_proj_a_token];
}

fn get_proj_b(head_num: u32, token_idx: u32) -> f32 {
    return proj_b[params.proj_b_offset + head_num + token_idx * params.stride_proj_b_token];
}

fn set_dst(head_num: u32, value_index: u32, token_idx: u32, value: f32) {
    dst[head_num * params.stride_dst_head + value_index + token_idx * params.stride_dst_token] = value;
}

// Workgroup scratch for parallel reduction
var<workgroup> reduce_scratch: array<f32, workgroup_size>;

// Private state: each thread holds ceil(value_head_dim / workgroup_size) columns of [key_head_dim].
// Max capacity: must be >= ceil(value_head_dim / workgroup_size) * key_head_dim.
// For workgroup_size=128, key_head_dim=128, value_head_dim=128: need 1*128 = 128.
const MAX_PRIVATE_STATE: u32 = 256u;
var<private> local_state: array<f32, MAX_PRIVATE_STATE>;

fn get_local_state(key_index: u32, value_index: u32) -> f32 {
    let row_offset = value_index / workgroup_size;
    return local_state[row_offset * params.key_head_dim + key_index];
}

fn set_local_state(key_index: u32, value_index: u32, value: f32) {
    let row_offset = value_index / workgroup_size;
    local_state[row_offset * params.key_head_dim + key_index] = value;
}

fn load_local_state(head_num: u32, value_index: u32) {
    for (var key_index = 0u; key_index < params.key_head_dim; key_index++) {
        set_local_state(key_index, value_index,
            state[head_num * params.stride_state_head + key_index * params.value_head_dim + value_index]);
    }
}

fn store_local_state(head_num: u32, value_index: u32) {
    for (var key_index = 0u; key_index < params.key_head_dim; key_index++) {
        state[head_num * params.stride_state_head + key_index * params.value_head_dim + value_index]
            = get_local_state(key_index, value_index);
    }
}

// Workgroup-wide sum reduction.
// Pre-condition: each thread has written its partial sum to reduce_scratch[lid].
fn workgroup_reduce_sum(lid: u32) -> f32 {
    var remaining = workgroup_size;
    while (remaining > 1u) {
        let half = (remaining + 1u) / 2u;
        workgroupBarrier();
        if (lid < remaining / 2u) {
            reduce_scratch[lid] += reduce_scratch[lid + half];
        }
        remaining = half;
    }
    workgroupBarrier();
    return reduce_scratch[0];
}

// One workgroup per head. Each thread handles one or more value_dim columns.
// Processes tokens sequentially within the head.
@compute @workgroup_size(workgroup_size)
fn delta_rule(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
    if (wg_id.x >= params.num_key_heads) {
        return;
    }
    let head_num = wg_id.x;
    let lid = local_id.x;

    // Load state from buffer (zero for fresh prefill, non-zero for continuation)
    for (var value_index = lid; value_index < params.value_head_dim; value_index += workgroup_size) {
        load_local_state(head_num, value_index);
    }

    let a_log = gate_params[params.a_log_offset + head_num];
    let dt_bias = gate_params[params.dt_bias_offset + head_num];

    for (var token_idx = 0u; token_idx < params.seq_len; token_idx += 1u) {
        let beta = sigmoid(get_proj_b(head_num, token_idx));
        let g = -exp(a_log) * softplus(get_proj_a(head_num, token_idx) + dt_bias);
        let gamma = exp(g);

        // ---- L2 norm for Q ----
        var q_partial: f32 = 0.0;
        for (var i = lid; i < params.key_head_dim; i += workgroup_size) {
            let val = get_q(i, head_num, token_idx);
            q_partial += val * val;
        }
        reduce_scratch[lid] = q_partial;
        let q_sq_sum = workgroup_reduce_sum(lid);

        // Write back: l2norm(q) * (1/sqrt(d_k))
        let q_scale = inverseSqrt((q_sq_sum + params.eps) * f32(params.key_head_dim));
        for (var i = lid; i < params.key_head_dim; i += workgroup_size) {
            let q_normalized = get_q(i, head_num, token_idx) * q_scale;
            set_q(i, head_num, token_idx, q_normalized);
        }

        // ---- L2 norm for K ----
        var k_partial: f32 = 0.0;
        for (var i = lid; i < params.key_head_dim; i += workgroup_size) {
            let val = get_k(i, head_num, token_idx);
            k_partial += val * val;
        }
        reduce_scratch[lid] = k_partial;
        let k_sq_sum = workgroup_reduce_sum(lid);

        let k_scale = inverseSqrt(k_sq_sum + params.eps);
        for (var i = lid; i < params.key_head_dim; i += workgroup_size) {
            let k_normalized = get_k(i, head_num, token_idx) * k_scale;
            set_k(i, head_num, token_idx, k_normalized);
        }
        workgroupBarrier();  // Ensure normalized Q/K visible to all threads

        // ---- Gated Delta Rule per value column ----
        for (var value_index = lid; value_index < params.value_head_dim; value_index += workgroup_size) {
            // Step 1: Decay state + retrieve prediction
            //   state[k][v] *= gamma
            //   kv_mem = sum_k( state[k][v] * k[k] )
            var kv_mem: f32 = 0.0;
            for (var key_index = 0u; key_index < params.key_head_dim; key_index++) {
                let gated = get_local_state(key_index, value_index) * gamma;
                set_local_state(key_index, value_index, gated);
                kv_mem += get_k(key_index, head_num, token_idx) * gated;
            }

            // Step 2: Error correction
            //   delta = beta * (v - kv_mem)
            let delta_v = (get_v(value_index, head_num, token_idx) - kv_mem) * beta;

            // Step 3: State update + output read
            //   state[k][v] += k[k] * delta
            //   output[v] = sum_k( q[k] * state[k][v] )
            var output: f32 = 0.0;
            for (var key_index = 0u; key_index < params.key_head_dim; key_index++) {
                let new_state = get_local_state(key_index, value_index)
                    + get_k(key_index, head_num, token_idx) * delta_v;
                set_local_state(key_index, value_index, new_state);
                output += get_q(key_index, head_num, token_idx) * new_state;
            }
            set_dst(head_num, value_index, token_idx, output);
        }
    }

    // Write back final state for KV cache
    for (var value_index = lid; value_index < params.value_head_dim; value_index += workgroup_size) {
        store_local_state(head_num, value_index);
    }
}
