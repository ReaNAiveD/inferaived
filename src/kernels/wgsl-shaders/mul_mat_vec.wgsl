// Matrix-vector multiply for the decode case (N = 1). One workgroup per
// output row; threads split K and tree-reduce the partials.
@group(0) @binding(0)
var<storage, read> weight: array<u32>;

@group(0) @binding(1)
var<storage, read> input: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

struct Params {
    m: u32,
    n: u32,                    // unused; assumed 1
    k: u32,

    weight_row_stride: u32,    // bf16 elements
    input_row_stride: u32,     // unused; only one input row is read
}

@group(0) @binding(3)
var<uniform> params: Params;

override workgroup_size: u32;
var<workgroup> scratch: array<f32, workgroup_size>;

@compute @workgroup_size(workgroup_size)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>) {
    let m_idx = wg_id.y * num_wg.x + wg_id.x;
    if (m_idx >= params.m) {
        return;
    }
    let row_bf16_start = m_idx * params.weight_row_stride;

    var partial: f32 = 0.0;
    for (var k: u32 = local_id.x; k < params.k; k += workgroup_size) {
        let bf16_idx = row_bf16_start + k;
        let packed = weight[bf16_idx / 2u];
        var w: f32;
        if (bf16_idx % 2u == 0u) {
            w = bitcast<f32>((packed & 0xFFFFu) << 16u);
        } else {
            w = bitcast<f32>(packed & 0xFFFF0000u);
        }
        partial += w * input[k];
    }

    scratch[local_id.x] = partial;
    workgroupBarrier();

    var remaining = workgroup_size;
    while (remaining > 1u) {
        let half = (remaining + 1u) / 2u;
        if (local_id.x < remaining / 2u) {
            scratch[local_id.x] += scratch[local_id.x + half];
        }
        remaining = half;
        workgroupBarrier();
    }

    if (local_id.x == 0u) {
        output[m_idx] = scratch[0];
    }
}
