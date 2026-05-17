struct Params {
    hidden_token_stride: u32,
    addend_token_stride: u32,

    hidden_size: u32,
    seq_len: u32,
}

@group(0) @binding(0)
var<storage, read_write> hidden: array<f32>;
@group(0) @binding(1)
var<storage, read> addend: array<f32>;
@group(0) @binding(2)
var<uniform> params: Params;

fn get_hidden(token: u32, i: u32) -> f32 {
    return hidden[token * params.hidden_token_stride + i];
}

fn set_hidden(token: u32, i: u32, value: f32) {
    hidden[token * params.hidden_token_stride + i] = value;
}

fn get_addend(token: u32, i: u32) -> f32 {
    return addend[token * params.addend_token_stride + i];
}

override workgroup_size: u32;

@compute @workgroup_size(workgroup_size)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    if (wg_id.x >= params.seq_len) {
        return;
    }

    for (var i : u32 = local_id.x; i < params.hidden_size; i += workgroup_size) {
        set_hidden(wg_id.x, i, get_hidden(wg_id.x, i) + get_addend(wg_id.x, i));
    }
}
