struct Params {
    num_heads: u32,
    head_dim: u32,
    seq_len: u32,

    hidden_offset: u32,
    hidden_token_stride: u32,
    hidden_head_stride: u32,

    gate_offset: u32,
    gate_token_stride: u32,
    gate_head_stride: u32,

    weight_offset: u32,

    eps: f32,
}

@group(0) @binding(0)
var<storage, read_write> hidden: array<f32>;
@group(0) @binding(1)
var<storage, read> gate: array<f32>;
@group(0) @binding(2)
var<storage, read> weight: array<f32>;
@group(0) @binding(3)
var<uniform> params: Params;

fn get_hidden(token: u32, head: u32, i: u32) -> f32 {
    return hidden[params.hidden_offset + token * params.hidden_token_stride + head * params.hidden_head_stride + i];
}

fn set_hidden(token: u32, head: u32, i: u32, value: f32) {
    hidden[params.hidden_offset + token * params.hidden_token_stride + head * params.hidden_head_stride + i] = value;
}

fn get_gate(token: u32, head: u32, i: u32) -> f32 {
    return gate[params.gate_offset + token * params.gate_token_stride + head * params.gate_head_stride + i];
}

fn silu(x: f32) -> f32 {
    return x / (1.0f + exp(-x));
}

override workgroup_size: u32;
var<workgroup> scratch: array<f32, workgroup_size>;

@compute @workgroup_size(workgroup_size)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    if (wg_id.x >= params.seq_len * params.num_heads) {
        return;
    }

    let token = wg_id.x / params.num_heads;
    let head = wg_id.x % params.num_heads;

    var sum: f32 = 0.0;
    for (var i: u32 = local_id.x; i < params.head_dim; i += workgroup_size) {
        let val = get_hidden(token, head, i);
        sum += val * val;
    }
    scratch[local_id.x] = sum;
    workgroupBarrier();

    var remaining: u32 = workgroup_size;
    while (remaining > 1u) {
        let half = (remaining + 1u) / 2u;
        if (local_id.x < remaining / 2u) {
            scratch[local_id.x] += scratch[local_id.x + half];
        }
        remaining = half;
        workgroupBarrier();
    }

    let scale = inverseSqrt(scratch[0] / f32(params.head_dim) + params.eps);

    for (var i: u32 = local_id.x; i < params.head_dim; i += workgroup_size) {
        let val = get_hidden(token, head, i);
        let w = weight[params.weight_offset + i];
        let g = get_gate(token, head, i);
        set_hidden(token, head, i, val * scale * w * silu(g));
    }
}
