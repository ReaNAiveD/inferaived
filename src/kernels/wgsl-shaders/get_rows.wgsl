// Source matrix: stored as a flat array<u32> in row-major order.
// Each u32 packs two bf16 values, so each u32 holds two elements of the logical matrix.
// Logical shape: [num_source_rows, hidden_size] (e.g. [vocab_size, hidden_size]).
// Row `r` starts at `source[source_offset + r * source_row_stride]`.
// `source_row_stride` is in u32 units = hidden_size / 2.
@group(0) @binding(0)
var<storage, read> source: array<u32>;
// Index vector: array<i32> of length `num_tokens`. Each entry is a row index into `source`.
@group(0) @binding(1)
var<storage, read> indices: array<i32>;
// Destination matrix: [num_tokens, hidden_size] in f32, row-major.
// Row i of `output` receives a copy of source row `indices[i]`.
@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

struct Params {
    source_offset: u32,       // in u32 elements
    source_row_stride: u32,   // in u32 elements (= hidden_size / 2)
    indices_offset: u32,      // in i32 elements
    output_offset: u32,       // in f32 elements
    output_row_stride: u32,   // in f32 elements

    hidden_size: u32,
    num_tokens: u32,
};

@group(0) @binding(3)
var<uniform> params: Params;

// Copies one bf16 element from source to output, given the base addresses and the offset in elements.
fn copy_element(source_base: u32, output_base: u32, offset: u32) {
    let packed = source[source_base + offset / 2u];
    if (offset % 2 == 0u) {
        // Even offset: lower 16 bits contain a bf16 value.
        // bf16 is the upper 16 bits of an f32, so shift left by 16.
        let bf16_bits = packed & 0xFFFFu;
        output[output_base + offset] = bitcast<f32>(bf16_bits << 16u);
    } else {
        // Odd offset: upper 16 bits contain a bf16 value.
        let bf16_bits = packed & 0xFFFF0000u;
        output[output_base + offset] = bitcast<f32>(bf16_bits);
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let token = global_id.x;
    if (token >= params.num_tokens) {
        return;
    }

    let row_idx = u32(indices[params.indices_offset + token]);
    let source_base = params.source_offset + row_idx * params.source_row_stride;
    let output_base = params.output_offset + token * params.output_row_stride;

    for (var i: u32 = 0u; i < params.hidden_size; i = i + 1u) {
        copy_element(source_base, output_base, i);
    }
}
