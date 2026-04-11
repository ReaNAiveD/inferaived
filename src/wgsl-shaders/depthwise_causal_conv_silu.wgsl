@group(0) @binding(0)
var<storage, read> src: array<f32>;
@group(0) @binding(1)
var<storage, read> conv_weights: array<u32>; // Packed bf16 convolution weights, in the same format as get_rows.wgsl
@group(0) @binding(2)
var<storage, read_write> dst: array<f32>;

struct Params {
    offset_src: u32, // in elements
    offset_weights: u32, // in elements
    offset_dst: u32, // in elements

    // Strides (in elements)
    stride_src1: u32,

    stride_weights1: u32, // in bf16 elements

    stride_dst1: u32,

    // Shape of src/dst
    ne0: u32,
    n_rows: u32,

    // Shape of weights
    kernel_size: u32,
    // The number of channels is the same as ne0
}

@group(0) @binding(3)
var<uniform> params: Params;

fn get_value(row: u32, channel: u32, lag: u32) -> f32 {
    if (lag > row) {
        return 0.0; // Out of bounds, treat as zero padding
    }
    let src_offset = params.offset_src + (row - lag) * params.stride_src1 + channel;
    return src[src_offset];
}

fn get_weight(channel: u32, k: u32) -> f32 {
    let weight_offset = params.offset_weights + channel * params.stride_weights1 + k;
    let packed = conv_weights[weight_offset / 2u];
    if (weight_offset % 2 == 0u) {
        // Even offset: lower 16 bits contain a bf16 value.
        // bf16 is the upper 16 bits of an f32, so shift left by 16.
        let bf16_bits = packed & 0xFFFFu;
        return bitcast<f32>(bf16_bits << 16u);
    } else {
        // Odd offset: upper 16 bits contain a bf16 value.
        // Already in the upper half, just mask off the lower bits.
        let bf16_bits = packed & 0xFFFF0000u;
        return bitcast<f32>(bf16_bits);
    }
}

@compute @workgroup_size(256)
fn conv1d_silu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= params.n_rows * params.ne0) {
        return;
    }
    let row = global_id.x / params.ne0;
    let channel = global_id.x % params.ne0;
    var sum: f32 = 0.0;
    for (var k = 0u; k < params.kernel_size; k++) {
        let lag = params.kernel_size - 1u - k;
        sum += get_value(row, channel, lag) * get_weight(channel, k);
    }
    let dst_offset = params.offset_dst + row * params.stride_dst1 + channel;
    let sigmoid = 1.0 / (1.0 + exp(-sum));
    dst[dst_offset] = sum * sigmoid;
}
