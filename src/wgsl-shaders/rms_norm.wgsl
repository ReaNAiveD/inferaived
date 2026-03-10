struct Params {
    offset_src: u32, // in elements

    // Strides (in elements)
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    // Shape of src/dst
    ne0: u32,
    ne1: u32,
    ne2: u32,
    ne3: u32,

    eps: f32
};

@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

@group(0) @binding(1)
var<uniform> params: Params;

override workgroup_size: u32;
var<workgroup> scratch: array<f32, workgroup_size>;

@compute @workgroup_size(workgroup_size)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx3 = wid.x / (params.ne1 * params.ne2);
    let idx2 = (wid.x % (params.ne1 * params.ne2)) / params.ne1;
    let idx1 = wid.x % params.ne1;
    let offset = params.offset_src + idx3 * params.stride_src3 + idx2 * params.stride_src2 + idx1 * params.stride_src1;

    var sum: f32 = 0.0;
    for (var i: u32 = lid.x; i < params.ne0; i += workgroup_size) {
        let val = src[offset + i];
        sum += val * val;
    }
    scratch[lid.x] = sum;
    workgroupBarrier();

    var remaining = workgroup_size;
    while (remaining > 1u) {
        let half = (remaining + 1u) / 2u;
        if (lid.x < remaining / 2u) {
            scratch[lid.x] += scratch[lid.x + half];
        }
        remaining = half;
        workgroupBarrier();
    }

    let wg_sum = scratch[0];
    let scale = inverseSqrt(wg_sum / f32(params.ne0) + params.eps);

    for (var i: u32 = lid.x; i < params.ne0; i += workgroup_size) {
        let val = src[offset + i];
        src[offset + i] = val * scale;
    }
}