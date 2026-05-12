// In-place compute: hidden[token, head, i] *= sigmoid(gate[token, head, i])
struct Params {
    hidden_offset: u32,
    hidden_token_stride: u32,
    hidden_head_stride: u32,

    gate_offset: u32,
    gate_token_stride: u32,
    gate_head_stride: u32,

    num_heads: u32,
    head_dim: u32,
    seq_len: u32,
}

@group(0) @binding(0)
var<storage, read_write> hidden: array<f32>;
@group(0) @binding(1)
var<storage, read> gate: array<f32>;
@group(0) @binding(2)
var<uniform> params: Params;

fn hidden_index(token: u32, head: u32, i: u32) -> u32 {
    return params.hidden_offset + token * params.hidden_token_stride + head * params.hidden_head_stride + i;
}

fn gate_index(token: u32, head: u32, i: u32) -> u32 {
    return params.gate_offset + token * params.gate_token_stride + head * params.gate_head_stride + i;
}

fn sigmoid(value: f32) -> f32 {
    return 1.0 / (1.0 + exp(-value));
}

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
        let h_idx = hidden_index(wg_id.x, head, i);
        let g_idx = gate_index(wg_id.x, head, i);
        hidden[h_idx] = sigmoid(gate[g_idx]) * hidden[h_idx];
    }
}
