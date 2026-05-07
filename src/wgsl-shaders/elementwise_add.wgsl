struct Param {
    stride_src_token: u32,
    stride_other_token: u32,
    src_offset: u32,
    other_offset: u32,

    hidden_size: u32,
    seq_len: u32,
}

@group(0) @binding(0)
var<storage, read_write> src: array<f32>;
@group(0) @binding(1)
var<storage, read> other: array<f32>;
@group(0) @binding(2)
var<uniform> param: Param;

fn get_src(token_index: u32, hidden_index: u32) -> f32 {
    return src[param.src_offset + token_index * param.stride_src_token + hidden_index];
}

fn set_src(token_index: u32, hidden_index: u32, value: f32) {
    src[param.src_offset + token_index * param.stride_src_token + hidden_index] = value;
}

fn get_other(token_index: u32, hidden_index: u32) -> f32 {
    return other[param.other_offset + token_index * param.stride_other_token + hidden_index];
}

override workgroup_size: u32;

@compute @workgroup_size(workgroup_size)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    if (wid.x >= param.seq_len) {
        return;
    }

    for (var i : u32 = lid.x; i < param.hidden_size; i += workgroup_size) {
        let added = get_src(wid.x, i) + get_other(wid.x, i);
        set_src(wid.x, i, added);
    }
}
