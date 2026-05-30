// Depthwise causal 1D conv + SiLU, with a rolling-window conv state cache.
//
// Per-channel kernel of size K. The lookback for each output token is the
// K most recent raw inputs of the *combined* stream `[conv_state, input]`;
// values that fall before `conv_state[0]` are treated as causal zeros.
//
// `conv_state` is the rolling window of the last `K - 1` raw inputs seen
// before the current call. After `conv_state_update`, it slides forward
// by `seq_len` tokens so the next call's lookback is correct.
//
// For a fresh session (no prior context), allocate `conv_state` zero-filled
// and the shader reproduces the original causal-zero-pad behavior.

@group(0) @binding(0)
var<storage, read> input: array<f32>;
@group(0) @binding(1)
var<storage, read> weight: array<u32>; // Packed bf16 convolution weights
@group(0) @binding(2)
var<storage, read_write> output: array<f32>;
@group(0) @binding(3)
var<storage, read_write> conv_state: array<f32>; // (K-1) rows × num_channels f32

struct Params {
    // Channel group dimensions: layout is [Q, K, V] contiguous per token
    q_dim: u32,
    k_dim: u32,
    v_dim: u32,

    seq_len: u32,       // Number of tokens
    kernel_size: u32,   // Temporal kernel size (e.g. 4)

    // Elements between consecutive tokens (>= q_dim + k_dim + v_dim when padded)
    input_token_stride: u32,
    output_token_stride: u32,
    state_token_stride: u32,

    // Per-group flag: 0 = passthrough copy, 1 = conv1d + silu
    q_apply_conv: u32,
    k_apply_conv: u32,
    v_apply_conv: u32,
}

@group(0) @binding(4)
var<uniform> params: Params;

fn num_channels() -> u32 {
    return params.q_dim + params.k_dim + params.v_dim;
}

// Lookback for token `token` at lag `lag`. When the lookback reaches before
// the current batch, fall through to `conv_state[K-1-(lag-token)]` (the
// `(lag-token)`-th most recent past input). If even the state doesn't reach
// that far, return causal zero.
fn get_input(token: u32, channel: u32, lag: u32) -> f32 {
    if (lag <= token) {
        return input[(token - lag) * params.input_token_stride + channel];
    }
    let back = lag - token; // 1 .. K-1 normally
    if (back >= params.kernel_size) {
        return 0.0; // beyond state's reach
    }
    let state_idx = params.kernel_size - 1u - back; // 0 .. K-2
    return conv_state[state_idx * params.state_token_stride + channel];
}

fn get_weight(channel: u32, k: u32) -> f32 {
    // Weights are packed bf16: each u32 holds two bf16 values.
    // Layout: [num_channels, kernel_size] in bf16 elements.
    let idx = channel * params.kernel_size + k;
    let packed = weight[idx / 2u];
    if (idx % 2 == 0u) {
        let bf16_bits = packed & 0xFFFFu;
        return bitcast<f32>(bf16_bits << 16u);
    } else {
        let bf16_bits = packed & 0xFFFF0000u;
        return bitcast<f32>(bf16_bits);
    }
}

@compute @workgroup_size(256)
fn conv1d_silu(@builtin(global_invocation_id) global_id: vec3<u32>,
               @builtin(num_workgroups) num_wg: vec3<u32>) {
    let threads_per_row = num_wg.x * 256u;
    let linear_wg = global_id.y * threads_per_row + global_id.x;
    let total = params.seq_len * num_channels();
    if (linear_wg >= total) {
        return;
    }
    let nc = num_channels();
    let token = linear_wg / nc;
    let channel = linear_wg % nc;

    // Determine flag for this channel's group
    var apply_conv: u32;
    if (channel < params.q_dim) {
        apply_conv = params.q_apply_conv;
    } else if (channel < params.q_dim + params.k_dim) {
        apply_conv = params.k_apply_conv;
    } else {
        apply_conv = params.v_apply_conv;
    }

    let output_idx = token * params.output_token_stride + channel;

    if (apply_conv == 0u) {
        // Passthrough copy
        output[output_idx] = input[token * params.input_token_stride + channel];
    } else {
        // Conv1D + SiLU
        var sum: f32 = 0.0;
        for (var k = 0u; k < params.kernel_size; k++) {
            let lag = params.kernel_size - 1u - k;
            sum += get_input(token, channel, lag) * get_weight(channel, k);
        }
        let sigmoid = 1.0 / (1.0 + exp(-sum));
        output[output_idx] = sum * sigmoid;
    }
}

// Refresh `conv_state` to hold the last `K - 1` raw inputs of the combined
// `[old conv_state, input]` stream. Run AFTER `conv1d_silu` in the same
// compute pass.
//
// One invocation per channel; that invocation walks i = 0 .. K-2 ascending
// and either copies `conv_state[seq_len + i]` (the "shift-left" case when
// the batch is shorter than K-1) or `input[seq_len + i - (K-1)]` (the
// "append from input" case). Reads precede writes index-wise within the
// loop, so the in-place update is safe per channel; channels don't share
// memory across invocations.
@compute @workgroup_size(256)
fn conv_state_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let channel = global_id.x;
    if (channel >= num_channels()) {
        return;
    }
    if (params.kernel_size < 2u) {
        return; // no state to maintain
    }
    let k_minus_1 = params.kernel_size - 1u;
    for (var i: u32 = 0u; i < k_minus_1; i++) {
        let combined_idx = params.seq_len + i;
        var val: f32;
        if (combined_idx >= k_minus_1) {
            let input_idx = combined_idx - k_minus_1;
            val = input[input_idx * params.input_token_stride + channel];
        } else {
            val = conv_state[combined_idx * params.state_token_stride + channel];
        }
        conv_state[i * params.state_token_stride + channel] = val;
    }
}
