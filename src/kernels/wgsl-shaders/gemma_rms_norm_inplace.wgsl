// Gemma-style in-place RMSNorm: gain is `1 + weight` (centered weights),
// used by this repo's Qwen3.5 checkpoint. The plain-weight variant
// (Llama / MiniCPM `LlamaRMSNorm`) lives in `rms_norm_inplace.wgsl`.
struct Params {
    hidden_row_stride: u32,   // in elements

    hidden_size: u32,
    seq_len: u32,

    eps: f32,
};

@group(0) @binding(0)
var<storage, read_write> hidden: array<f32>;

@group(0) @binding(1)
var<storage, read> weight: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

override workgroup_size: u32;
var<workgroup> scratch: array<f32, workgroup_size>;

@compute @workgroup_size(workgroup_size)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    if (wg_id.x >= params.seq_len) {
        return;
    }
    let row_offset = wg_id.x * params.hidden_row_stride;

    var sum: f32 = 0.0;
    for (var i: u32 = local_id.x; i < params.hidden_size; i += workgroup_size) {
        let val = hidden[row_offset + i];
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
        let val = hidden[row_offset + i];
        hidden[row_offset + i] = val * scale * (1.0 + weight[i]);
    }
}
