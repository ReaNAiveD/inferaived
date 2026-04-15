@group(0) @binding(0)
var<storage, read> src: array<f32>;
@group(0) @binding(1)
var<storage, read> conv_weights: array<u32>; // Packed bf16 convolution weights
@group(0) @binding(2)
var<storage, read_write> dst: array<f32>;

struct Params {
    // Channel group dimensions: layout is [Q, K, V] contiguous per token
    q_dim: u32,
    k_dim: u32,
    v_dim: u32,

    seq_len: u32,       // Number of tokens
    kernel_size: u32,   // Temporal kernel size (e.g. 4)

    // Elements between consecutive tokens (>= q_dim + k_dim + v_dim when padded)
    stride_src_token: u32,
    stride_dst_token: u32,

    // Per-group mode: 0 = copy (passthrough), 1 = conv1d + silu
    q_mode: u32,
    k_mode: u32,
    v_mode: u32,
}

@group(0) @binding(3)
var<uniform> params: Params;

fn num_channels() -> u32 {
    return params.q_dim + params.k_dim + params.v_dim;
}

fn get_src(token: u32, channel: u32, lag: u32) -> f32 {
    if (lag > token) {
        return 0.0; // Causal zero padding
    }
    return src[(token - lag) * params.stride_src_token + channel];
}

fn get_weight(channel: u32, k: u32) -> f32 {
    // Weights are packed bf16: each u32 holds two bf16 values.
    // Layout: [num_channels, kernel_size] in bf16 elements.
    let idx = channel * params.kernel_size + k;
    let packed = conv_weights[idx / 2u];
    if (idx % 2 == 0u) {
        let bf16_bits = packed & 0xFFFFu;
        return bitcast<f32>(bf16_bits << 16u);
    } else {
        let bf16_bits = packed & 0xFFFF0000u;
        return bitcast<f32>(bf16_bits);
    }
}

@compute @workgroup_size(256)
fn conv1d_silu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total = params.seq_len * num_channels();
    if (global_id.x >= total) {
        return;
    }
    let nc = num_channels();
    let token = global_id.x / nc;
    let channel = global_id.x % nc;

    // Determine mode for this channel's group
    var mode: u32;
    if (channel < params.q_dim) {
        mode = params.q_mode;
    } else if (channel < params.q_dim + params.k_dim) {
        mode = params.k_mode;
    } else {
        mode = params.v_mode;
    }

    let dst_idx = token * params.stride_dst_token + channel;

    if (mode == 0u) {
        // Copy mode: passthrough
        dst[dst_idx] = src[token * params.stride_src_token + channel];
    } else {
        // Conv1D + SiLU mode
        var sum: f32 = 0.0;
        for (var k = 0u; k < params.kernel_size; k++) {
            let lag = params.kernel_size - 1u - k;
            sum += get_src(token, channel, lag) * get_weight(channel, k);
        }
        let sigmoid = 1.0 / (1.0 + exp(-sum));
        dst[dst_idx] = sum * sigmoid;
    }
}
