// Masked Grouped-Query Attention over a shared KV row pool: one workgroup per
// (q_row, q_head) runs a 3-pass safe softmax over the rows selected by
// `visibility` + `scatter_position`, then weighted-sums V. Q is pre-rotated and
// K is stored pre-rotated, so the shader needs no RoPE positions.
//
// `visibility` is a flat range list (`[N, s0, e0, ..., s_{N-1}, _]`, N >= 1)
// shared by every q_head of a q_row. Ranges `0 .. N-1` are read as-is; the last
// range is the per-row causal tail ending at `scatter_position + 1 + q_row`, so
// it grows with K/V scatter without rewriting the buffer. The trailing stored
// end of the last range is unread.
struct Params {
    q_token_stride: u32,
    q_head_stride: u32,
    k_token_stride: u32,
    k_head_stride: u32,
    v_token_stride: u32,
    v_head_stride: u32,
    output_token_stride: u32,
    output_head_stride: u32,

    num_q_heads: u32,
    num_kv_heads: u32,
    q_dim: u32,
    v_dim: u32,
    num_q_rows: u32,
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

// Flat visible-range list; see the header comment for the layout.
@group(0) @binding(5)
var<storage, read> visibility: array<u32>;

// Pool row this batch's q_row 0 writes K/V into; the last visibility range ends
// at `scatter_position + 1 + q_row`, so updating this uniform grows the tail.
@group(0) @binding(6)
var<uniform> scatter_position: u32;

fn get_q(token: u32, head: u32, in_head_index: u32) -> f32 {
    return q[in_head_index + head * params.q_head_stride + token * params.q_token_stride];
}
fn get_k(token: u32, head: u32, in_head_index: u32) -> f32 {
    return k[in_head_index + head * params.k_head_stride + token * params.k_token_stride];
}
fn get_v(token: u32, head: u32, in_head_index: u32) -> f32 {
    return v[in_head_index + head * params.v_head_stride + token * params.v_token_stride];
}
fn set_output(token: u32, head: u32, in_head_index: u32, value: f32) {
    output[in_head_index + head * params.output_head_stride + token * params.output_token_stride] = value;
}

override workgroup_size: u32;
var<workgroup> reduce_scratch: array<f32, workgroup_size>;

// Per-thread private accumulator for the V-weighted sum: each thread owns
// `ceil(v_dim / workgroup_size)` slots, bounded by 4 in practice.
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

// Scaled Q·K score for one (q_row, q_head, k_token), reduced across the
// workgroup; must be called in uniform control flow (it barriers).
fn score_for(q_row: u32, q_head: u32, kv_head: u32, k_token: u32, local_id: vec3<u32>) -> f32 {
    var score_acc: f32 = 0f;
    for (var d: u32 = local_id.x; d < params.q_dim; d += workgroup_size) {
        score_acc += get_q(q_row, q_head, d) * get_k(k_token, kv_head, d);
    }
    reduce_scratch[local_id.x] = score_acc;
    return workgroup_reduce_sum(local_id) * inverseSqrt(f32(params.q_dim));
}

// Inclusive start (`.x`) and exclusive end (`.y`) of visible range `r` for this
// `q_row`; the last range's end is `scatter_position + 1 + q_row`.
fn visible_range(r: u32, num_ranges: u32, q_row: u32) -> vec2<u32> {
    let start = visibility[1u + r * 2u];
    if (r == num_ranges - 1u) {
        return vec2<u32>(start, scatter_position + 1u + q_row);
    }
    return vec2<u32>(start, visibility[2u + r * 2u]);
}

@compute @workgroup_size(workgroup_size)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>) {
    let linear_wg = wg_id.y * num_wg.x + wg_id.x;
    if (linear_wg >= params.num_q_rows * params.num_q_heads) {
        return;
    }
    let q_row = linear_wg / params.num_q_heads;
    let q_head = linear_wg % params.num_q_heads;
    let num_q_per_kv = params.num_q_heads / params.num_kv_heads;
    let kv_head = q_head / num_q_per_kv;
    let num_ranges = visibility[0];

    // Pass 1: max over visible rows.
    var max_score: f32 = -1e30;
    for (var r: u32 = 0u; r < num_ranges; r += 1u) {
        let range = visible_range(r, num_ranges, q_row);
        for (var k_token: u32 = range.x; k_token < range.y; k_token += 1u) {
            let score = score_for(q_row, q_head, kv_head, k_token, local_id);
            max_score = max(score, max_score);
        }
    }
    // Pass 2: sum of exponentials over visible rows.
    var sum_exp: f32 = 0f;
    for (var r: u32 = 0u; r < num_ranges; r += 1u) {
        let range = visible_range(r, num_ranges, q_row);
        for (var k_token: u32 = range.x; k_token < range.y; k_token += 1u) {
            let score = score_for(q_row, q_head, kv_head, k_token, local_id);
            sum_exp += exp(score - max_score);
        }
    }
    // Pass 3: V-weighted accumulation over visible rows.
    var v_acc: array<f32, MAX_V_PER_THREAD>;
    for (var i: u32 = 0u; i < MAX_V_PER_THREAD; i += 1u) {
        v_acc[i] = 0f;
    }
    for (var r: u32 = 0u; r < num_ranges; r += 1u) {
        let range = visible_range(r, num_ranges, q_row);
        for (var k_token: u32 = range.x; k_token < range.y; k_token += 1u) {
            let score = score_for(q_row, q_head, kv_head, k_token, local_id);
            let weight = exp(score - max_score) / sum_exp;
            for (var slot: u32 = 0u; slot < MAX_V_PER_THREAD; slot += 1u) {
                let v_d = local_id.x + slot * workgroup_size;
                if (v_d < params.v_dim) {
                    v_acc[slot] += weight * get_v(k_token, kv_head, v_d);
                }
            }
        }
    }

    for (var slot: u32 = 0u; slot < MAX_V_PER_THREAD; slot += 1u) {
        let v_d = local_id.x + slot * workgroup_size;
        if (v_d < params.v_dim) {
            set_output(q_row, q_head, v_d, v_acc[slot]);
        }
    }
}
