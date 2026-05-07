struct Param {
    src_offset: u32,
    stride_src_token: u32,
    gate_offset: u32,
    stride_gate_token: u32,

    hidden_size: u32,
    seq_len: u32,
}

@group(0) @binding(0)
var<storage, read_write> src: array<f32>;
@group(0) @binding(1)
var<storage, read> gate: array<f32>;
@group(0) @binding(2)
var<uniform> param: Param;

fn get_src(token_index: u32, hidden_index: u32) -> f32 {
    return src[param.src_offset + token_index * param.stride_src_token + hidden_index];
}

fn set_src(token_index: u32, hidden_index: u32, value: f32) {
    src[param.src_offset + token_index * param.stride_src_token + hidden_index] = value;
}

fn get_gate(token_index: u32, hidden_index: u32) -> f32 {
    return gate[param.gate_offset + token_index * param.stride_gate_token + hidden_index];
}

fn silu(value: f32) -> f32 {
    return value / (1 + exp(- value));
}

override workgroup_size: u32;

@compute @workgroup_size(workgroup_size)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    if (wid.x >= param.seq_len) {
        return;
    }

    for (var i: u32 = lid.x; i < param.hidden_size; i += workgroup_size) {
        let silu_gate = silu(get_gate(wid.x, i));
        set_src(wid.x, i, silu_gate * get_src(wid.x, i));
    }
}
