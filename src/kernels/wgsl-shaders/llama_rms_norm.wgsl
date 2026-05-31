// Llama-style RMSNorm: gain is plain `weight` (Llama / MiniCPM
// `LlamaRMSNorm`). The Gemma-style `1 + weight` variant lives in
// `gemma_rms_norm.wgsl`.
struct Params {
    input_row_stride: u32,    // in elements
    output_row_stride: u32,   // in elements

    hidden_size: u32,
    seq_len: u32,

    eps: f32,
};

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@group(0) @binding(2)
var<storage, read> weight: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

override workgroup_size: u32;
var<workgroup> scratch: array<f32, workgroup_size>;

@compute @workgroup_size(workgroup_size)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    if (wg_id.x >= params.seq_len) {
        return;
    }
    let input_row_offset = wg_id.x * params.input_row_stride;
    let output_row_offset = wg_id.x * params.output_row_stride;

    var sum: f32 = 0.0;
    for (var i: u32 = local_id.x; i < params.hidden_size; i += workgroup_size) {
        let val = input[input_row_offset + i];
        sum += val * val;
    }
    scratch[local_id.x] = sum;
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

    let scale = inverseSqrt(scratch[0] / f32(params.hidden_size) + params.eps);

    for (var i: u32 = local_id.x; i < params.hidden_size; i += workgroup_size) {
        let val = input[input_row_offset + i];
        output[output_row_offset + i] = val * scale * weight[i];
    }
}