struct Param {
    offset_src: u32,

    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    ne0: u32,
    ne1: u32,
    ne2: u32,
    ne3: u32,

    ne0_scale: u32,
    offset_scale: u32,
}

@group(0) @binding(0)
var<storage, read_write> src: array<f32>;
@group(0) @binding(1)
var<storage, read> scale: array<f32>;
@group(0) @binding(2)
var<uniform> params: Param;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.ne0 * params.ne1 * params.ne2 * params.ne3) {
        return;
    }
    let idx3 = gid.x / (params.ne0 * params.ne1 * params.ne2);
    let idx2 = (gid.x % (params.ne0 * params.ne1 * params.ne2)) / (params.ne0 * params.ne1);
    let idx1 = (gid.x % (params.ne0 * params.ne1)) / params.ne0;
    let idx0 = gid.x % params.ne0;
    let src_offset = params.offset_src + idx3 * params.stride_src3 + idx2 * params.stride_src2 + idx1 * params.stride_src1 + idx0;
    let scale_val = scale[params.offset_scale + (idx0 % params.ne0_scale)];
    src[src_offset] *= scale_val;
}