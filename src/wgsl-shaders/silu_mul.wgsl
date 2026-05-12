// In-place compute: hidden[token, i] *= silu(gate[token, i])
struct Params {
    hidden_offset: u32,
    hidden_token_stride: u32,
    gate_offset: u32,
    gate_token_stride: u32,

    hidden_size: u32,
    seq_len: u32,
}

@group(0) @binding(0)
var<storage, read_write> hidden: array<f32>;
@group(0) @binding(1)
var<storage, read> gate: array<f32>;
@group(0) @binding(2)
var<uniform> params: Params;

fn get_hidden(token: u32, i: u32) -> f32 {
    return hidden[params.hidden_offset + token * params.hidden_token_stride + i];
}

fn set_hidden(token: u32, i: u32, value: f32) {
    hidden[params.hidden_offset + token * params.hidden_token_stride + i] = value;
}

fn get_gate(token: u32, i: u32) -> f32 {
    return gate[params.gate_offset + token * params.gate_token_stride + i];
}

fn silu(value: f32) -> f32 {
    return value / (1.0 + exp(-value));
}

override workgroup_size: u32;

@compute @workgroup_size(workgroup_size)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    if (wg_id.x >= params.seq_len) {
        return;
    }
    for (var i: u32 = local_id.x; i < params.hidden_size; i += workgroup_size) {
        set_hidden(wg_id.x, i, silu(get_gate(wg_id.x, i)) * get_hidden(wg_id.x, i));
    }
}
