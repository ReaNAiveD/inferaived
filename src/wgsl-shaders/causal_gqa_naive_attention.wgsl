// Causal Grouped-Query Attention. One workgroup per (q_token, q_head); each Q
// dots against all K up to its global position (q_token + q_token_offset),
// 3-pass safe softmax over kv_seq_len, then weighted sum of V. GQA index:
// kv_head = q_head / (num_q_heads / num_kv_heads). No KV cache management.
struct Params {
    q_offset: u32,
    q_token_stride: u32,
    q_head_stride: u32,
    k_offset: u32,
    k_token_stride: u32,
    k_head_stride: u32,
    v_offset: u32,
    v_token_stride: u32,
    v_head_stride: u32,
    output_offset: u32,
    output_token_stride: u32,
    output_head_stride: u32,

    num_q_heads: u32,
    num_kv_heads: u32,
    q_dim: u32,
    v_dim: u32,
    seq_len: u32,
}

@group(0) @binding(0)
var<storage, read> q: array<f32>;
@group(0) @binding(1)
var<storage, read> k: array<f32>;
@group(0) @binding(2)
var<storage, read> v: array<f32>;
@group(0) @binding(3)
var<storage, read_write> output: array<f32>;

@group(0) @binding(4)
var<uniform> params: Params;

fn get_q(token: u32, head: u32, in_head_index: u32) -> f32 {
    return q[params.q_offset + in_head_index + head * params.q_head_stride + token * params.q_token_stride];
}
fn get_k(token: u32, head: u32, in_head_index: u32) -> f32 {
    return k[params.k_offset + in_head_index + head * params.k_head_stride + token * params.k_token_stride];
}
fn get_v(token: u32, head: u32, in_head_index: u32) -> f32 {
    return v[params.v_offset + in_head_index + head * params.v_head_stride + token * params.v_token_stride];
}
fn set_output(token: u32, head: u32, in_head_index: u32, value: f32) {
    output[params.output_offset + in_head_index + head * params.output_head_stride + token * params.output_token_stride] = value;
}

override workgroup_size: u32;
var<workgroup> reduce_scratch: array<f32, workgroup_size>;

// Per-thread private accumulator for the V-weighted sum: each thread owns
// `ceil(v_dim / workgroup_size)` slots of the output head. With v_dim <= 256
// (Llama / Qwen / Gemma / Phi / DeepSeek all use head_dim in {128, 256}) and
// workgroup_size >= 64 this never exceeds 4.
const MAX_V_PER_THREAD: u32 = 4u;

fn workgroup_reduce_sum(local_id: vec3<u32>) -> f32 {
    var remaining: u32 = workgroup_size;
    while (remaining > 1) {
        let half = (remaining + 1) / 2;
        workgroupBarrier();
        if (local_id.x < remaining / 2) {
            reduce_scratch[local_id.x] += reduce_scratch[local_id.x + half];
        }
        remaining = half;
    }
    workgroupBarrier();
    return reduce_scratch[0];
}

@compute @workgroup_size(workgroup_size)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    if (wg_id.x >= params.seq_len * params.num_q_heads) {
        return;
    }
    let q_token = wg_id.x / params.num_q_heads;
    let q_head = wg_id.x % params.num_q_heads;
    let num_q_per_kv = params.num_q_heads / params.num_kv_heads;
    let kv_head = q_head / num_q_per_kv;
    let softmax_scale = inverseSqrt(f32(params.q_dim));

    // Scan max
    var max_score: f32 = -1e30;
    for (var k_token: u32 = 0; k_token <= q_token; k_token += 1u) {
        var score_acc: f32 = 0f;
        for (var d: u32 = local_id.x; d < params.q_dim; d += workgroup_size) {
            score_acc += get_q(q_token, q_head, d) * get_k(k_token, kv_head, d);
        }
        reduce_scratch[local_id.x] = score_acc;
        let score = workgroup_reduce_sum(local_id) * softmax_scale;
        max_score = max(score, max_score);
    }
    // Sum Exp
    var sum_exp: f32 = 0f;
    for (var k_token: u32 = 0; k_token <= q_token; k_token += 1u) {
        var score_acc: f32 = 0f;
        for (var d: u32 = local_id.x; d < params.q_dim; d += workgroup_size) {
            score_acc += get_q(q_token, q_head, d) * get_k(k_token, kv_head, d);
        }
        reduce_scratch[local_id.x] = score_acc;
        let score = workgroup_reduce_sum(local_id) * softmax_scale;
        sum_exp += exp(score - max_score);
    }
    // V add — Each thread keeps its share of the output head in a private
    // array of MAX_V_PER_THREAD slots.
    var v_acc: array<f32, MAX_V_PER_THREAD>;
    for (var i: u32 = 0u; i < MAX_V_PER_THREAD; i += 1u) {
        v_acc[i] = 0f;
    }
    for (var k_token: u32 = 0; k_token <= q_token; k_token += 1u) {
        var score_acc: f32 = 0f;
        for (var q_d: u32 = local_id.x; q_d < params.q_dim; q_d += workgroup_size) {
            score_acc += get_q(q_token, q_head, q_d) * get_k(k_token, kv_head, q_d);
        }
        reduce_scratch[local_id.x] = score_acc;
        let score = workgroup_reduce_sum(local_id) * softmax_scale;
        let weight = exp(score - max_score) / sum_exp;

        // Divergent inner loop is OK here: no barriers below this point in the
        // k_token iteration. The loop has a const trip count so the compiler
        // can fully unroll it and keep v_acc[i] in registers.
        for (var slot: u32 = 0u; slot < MAX_V_PER_THREAD; slot += 1u) {
            let v_d = local_id.x + slot * workgroup_size;
            if (v_d < params.v_dim) {
                v_acc[slot] += weight * get_v(k_token, kv_head, v_d);
            }
        }
    }

    for (var slot: u32 = 0u; slot < MAX_V_PER_THREAD; slot += 1u) {
        let v_d = local_id.x + slot * workgroup_size;
        if (v_d < params.v_dim) {
            set_output(q_token, q_head, v_d, v_acc[slot]);
        }
    }
}