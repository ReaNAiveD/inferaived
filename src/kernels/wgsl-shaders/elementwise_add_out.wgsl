struct Params {
    dst_token_stride: u32,
    a_token_stride: u32,
    b_token_stride: u32,
    hidden_size: u32,
    seq_len: u32,
}

@group(0) @binding(0)
var<storage, read_write> dst: array<f32>;
@group(0) @binding(1)
var<storage, read> a: array<f32>;
@group(0) @binding(2)
var<storage, read> b: array<f32>;
@group(0) @binding(3)
var<uniform> params: Params;

override workgroup_size: u32;

@compute @workgroup_size(workgroup_size)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    if (wg_id.x >= params.seq_len) {
        return;
    }
    let t = wg_id.x;
    let dst_base = t * params.dst_token_stride;
    let a_base = t * params.a_token_stride;
    let b_base = t * params.b_token_stride;
    for (var i : u32 = local_id.x; i < params.hidden_size; i += workgroup_size) {
        dst[dst_base + i] = a[a_base + i] + b[b_base + i];
    }
}
