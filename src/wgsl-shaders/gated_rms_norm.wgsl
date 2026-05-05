struct Param {
    // Model dimensions
    num_value_heads: u32,
    value_head_dim: u32,
    seq_len: u32,

    src_offset: u32,
    stride_src_token: u32,
    stride_src_head: u32,

    gate_offset: u32,
    stride_gate_token: u32,
    stride_gate_head: u32,

    weight_offset: u32,

    eps: f32,
}

@group(0) @binding(0)
var<storage, read_write> src: array<f32>;
@group(0) @binding(1)
var<storage, read> gate: array<f32>;
@group(0) @binding(2)
var<storage, read> weight: array<f32>;
@group(0) @binding(3)
var<uniform> param: Param;

fn get_src(token_num: u32, head_num: u32, value_index: u32) -> f32 {
    return src[param.src_offset + param.stride_src_token * token_num + param.stride_src_head * head_num + value_index];
}

fn set_src(token_num: u32, head_num: u32, value_index: u32, value: f32) {
    src[param.src_offset + param.stride_src_token * token_num + param.stride_src_head * head_num + value_index] = value;
}

fn get_gate(token_num: u32, head_num: u32, value_index: u32) -> f32 {
    return gate[param.gate_offset + param.stride_gate_token * token_num + param.stride_gate_head * head_num + value_index];
}

fn silu(x: f32) -> f32 {
    return x / (1.0f + exp(-x));
}

override workgroup_size: u32;
var<workgroup> scratch: array<f32, workgroup_size>;

@compute @workgroup_size(workgroup_size)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    if (wid.x >= param.seq_len * param.num_value_heads) {
        return;
    }

    let token_num = wid.x / param.num_value_heads;
    let head_num = wid.x % param.num_value_heads;

    var sum: f32 = 0.0;
    for (var i: u32 = lid.x; i < param.value_head_dim; i += workgroup_size) {
        let val = get_src(token_num, head_num, i);
        sum += val * val;
    }
    scratch[lid.x] = sum;
    workgroupBarrier();

    var remain: u32 = workgroup_size;
    while (remain > 1u) {
        let half = (remain + 1u) / 2u;
        if (lid.x < remain / 2u) {
            scratch[lid.x] += scratch[lid.x + half];
        }
        remain = half;
        workgroupBarrier();
    }

    let wg_sum = scratch[0];
    let scale = inverseSqrt(wg_sum / f32(param.value_head_dim) + param.eps);

    for (var i: u32 = lid.x; i < param.value_head_dim; i += workgroup_size) {
        let val = get_src(token_num, head_num, i);
        let w = weight[param.weight_offset + i];
        let g = get_gate(token_num, head_num, i);
        let normalized = val * scale * w * silu(g);
        set_src(token_num, head_num, i, normalized);
    }
}
