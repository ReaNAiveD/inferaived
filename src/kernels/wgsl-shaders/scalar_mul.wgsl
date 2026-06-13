struct Params {
    row_stride: u32,
    hidden_size: u32,
    seq_len: u32,
    scalar: f32,
}

@group(0) @binding(0)
var<storage, read_write> hidden: array<f32>;
@group(0) @binding(1)
var<uniform> params: Params;

override workgroup_size: u32;

@compute @workgroup_size(workgroup_size)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    if (wg_id.x >= params.seq_len) {
        return;
    }
    let base = wg_id.x * params.row_stride;
    for (var i : u32 = local_id.x; i < params.hidden_size; i += workgroup_size) {
        hidden[base + i] = hidden[base + i] * params.scalar;
    }
}
