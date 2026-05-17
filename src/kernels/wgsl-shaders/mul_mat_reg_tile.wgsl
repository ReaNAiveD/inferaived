// weight: weight matrix, shape M × K, row-major (K is contiguous).
// Each row is one output neuron's weight vector of length K.
// Stored as u32 because each u32 packs two bf16 values.
// Access: weight[(m * K + k) / 2], where k varies fastest.
@group(0) @binding(0)
var<storage, read> weight: array<u32>; // M rows, K columns, row-major
// input: input activation, shape N × K, row-major (K is contiguous).
// Each row is one token's hidden vector of length K (transposed from the math perspective).
// Access: input[n * K + k], where k varies fastest.
@group(0) @binding(1)
var<storage, read> input: array<f32>; // N rows, K columns, row-major (transposed input)
// output: output activation, shape N × M, column-major (M is contiguous).
// Each column corresponds to one output neuron; consecutive elements are different tokens.
// Access: output[n * M + m], where m varies fastest.
@group(0) @binding(2)
var<storage, read_write> output: array<f32>; // N rows, M columns, column-major (M contiguous)

struct Params {
    m: u32,
    n: u32,
    k: u32,

    weight_row_stride: u32,
    input_row_stride: u32,
}

@group(0) @binding(3)
var<uniform> params: Params;

// The number of elements of weight and input per tile.
// Each thread is responsible for computing a tile of size tile_m x tile_n, and needs to load tile_k elements from weight and input.
// tile_m and tile_n are const (not override) because they size per-thread arrays
// which must be constructible types in WGSL.
const tile_m: u32 = 4u;
const tile_n: u32 = 4u;
override tile_k: u32;
// The number of threads in the M and N dimensions of the workgroup
override workgroup_size_m: u32;
override workgroup_size_n: u32;

// Shared memory for weight tile: (workgroup_size_m * tile_m) rows × tile_k columns, row-major (tile_k contiguous).
var<workgroup> shmem_weight: array<f32, tile_k * workgroup_size_m * tile_m>;
// Shared memory for input tile: (workgroup_size_n * tile_n) rows × tile_k columns, row-major (tile_k contiguous).
var<workgroup> shmem_input: array<f32, tile_k * workgroup_size_n * tile_n>;

// Per-invocation accumulators.
var<private> acc: array<f32, tile_m * tile_n>;
var<private> weight_reg: array<f32, tile_m>;

// Initialize the shared memory tile for weight.
//
// The shmem_weight tile has (workgroup_size_m * tile_m) rows and tile_k columns.
// Each thread (local_idx_m, local_idx_n) loads a vertical strip of tile_m rows,
// strided across K by workgroup_size_n.
//
// Example: workgroup_size_m=2, workgroup_size_n=4, tile_m=3, tile_k=8
//
// shmem_weight layout (6 rows × 8 cols):
//
//          K dimension (tile_k = 8, contiguous in memory) →
//          k=0   k=1   k=2   k=3   k=4   k=5   k=6   k=7
//        ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
// row 0  │(0,0)│(0,1)│(0,2)│(0,3)│(0,0)│(0,1)│(0,2)│(0,3)│  ← thread m=0 owns
// row 1  │(0,0)│(0,1)│(0,2)│(0,3)│(0,0)│(0,1)│(0,2)│(0,3)│    rows 0,1,2
// row 2  │(0,0)│(0,1)│(0,2)│(0,3)│(0,0)│(0,1)│(0,2)│(0,3)│    (inner_m = 0..3)
//        ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
// row 3  │(1,0)│(1,1)│(1,2)│(1,3)│(1,0)│(1,1)│(1,2)│(1,3)│  ← thread m=1 owns
// row 4  │(1,0)│(1,1)│(1,2)│(1,3)│(1,0)│(1,1)│(1,2)│(1,3)│    rows 3,4,5
// row 5  │(1,0)│(1,1)│(1,2)│(1,3)│(1,0)│(1,1)│(1,2)│(1,3)│    (inner_m = 3..6)
//        └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
//
// (m,n) = (local_idx_m, local_idx_n) of the thread that loads that element.
//
// M dimension: thread m loads rows [m*tile_m .. (m+1)*tile_m), no overlap.
// K dimension: thread n loads cols n, n+workgroup_size_n, n+2*workgroup_size_n, ...
//              e.g. thread n=0 loads k=0,4; thread n=1 loads k=1,5; etc.
//
fn init_shmem_weight(wg_idx_m: u32, local_idx_m: u32, k_offset: u32, local_idx_n: u32) {
    for (var inner_k = local_idx_n; inner_k < tile_k; inner_k += workgroup_size_n) {
        for (var inner_m = local_idx_m * tile_m; inner_m < local_idx_m * tile_m + tile_m; inner_m += 1u) {
            let global_m = wg_idx_m * workgroup_size_m * tile_m + inner_m;
            let global_k = k_offset + inner_k;
            if (global_m >= params.m || global_k >= params.k) {
                shmem_weight[inner_k + inner_m * tile_k] = 0.0;
                continue;
            }
            let src_offset = global_m * params.weight_row_stride + global_k;
            let packed = weight[src_offset / 2u];
            let target_offset = inner_k + inner_m * tile_k;
            if (src_offset % 2u == 0u) {
                let bf16_bits = packed & 0xFFFFu;
                shmem_weight[target_offset] = bitcast<f32>(bf16_bits << 16u);
            } else {
                let bf16_bits = packed & 0xFFFF0000u;
                shmem_weight[target_offset] = bitcast<f32>(bf16_bits);
            }
        }
    }
}

fn init_shmem_input(wg_idx_n: u32, local_idx_n: u32, k_offset: u32, local_idx_m: u32) {
    for (var inner_k = local_idx_m; inner_k < tile_k; inner_k += workgroup_size_m) {
        for (var inner_n = local_idx_n * tile_n; inner_n < local_idx_n * tile_n + tile_n; inner_n += 1u) {
            let global_n = wg_idx_n * workgroup_size_n * tile_n + inner_n;
            let global_k = k_offset + inner_k;
            if (global_n >= params.n || global_k >= params.k) {
                shmem_input[inner_k + inner_n * tile_k] = 0.0;
                continue;
            }
            let src_offset = global_n * params.input_row_stride + global_k;
            shmem_input[inner_k + inner_n * tile_k] = input[src_offset];
        }
    }
}

@compute @workgroup_size(workgroup_size_m * workgroup_size_n)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>) {
    let wg_num_m = (params.m + workgroup_size_m * tile_m - 1u) / (workgroup_size_m * tile_m);
    let wg_num_n = (params.n + workgroup_size_n * tile_n - 1u) / (workgroup_size_n * tile_n);
    let wg_idx_n = wg_id.x / wg_num_m;
    let wg_idx_m = wg_id.x % wg_num_m;

    let local_idx_m = local_id.x % workgroup_size_m;
    let local_idx_n = local_id.x / workgroup_size_m;

    for (var i = 0u; i < tile_m * tile_n; i++) {
        acc[i] = 0.0;
    }
    for (var k_offset = 0u; k_offset < params.k; k_offset += tile_k) {
        init_shmem_weight(wg_idx_m, local_idx_m, k_offset, local_idx_n);
        init_shmem_input(wg_idx_n, local_idx_n, k_offset, local_idx_m);

        workgroupBarrier();

        let k_end = min(tile_k, params.k - k_offset);
        for (var inner_k = 0u; inner_k < k_end; inner_k++) {
            // Load tile_m weight values once
            for (var inner_m = 0u; inner_m < tile_m; inner_m++) {
                weight_reg[inner_m] = shmem_weight[inner_k + (local_idx_m * tile_m + inner_m) * tile_k];
            }
            // For each N, multiply and accumulate
            for (var inner_n = 0u; inner_n < tile_n; inner_n++) {
                let input_val = shmem_input[inner_k + (local_idx_n * tile_n + inner_n) * tile_k];
                for (var inner_m = 0u; inner_m < tile_m; inner_m++) {
                    acc[inner_n * tile_m + inner_m] += weight_reg[inner_m] * input_val;
                }
            }
        }

        workgroupBarrier();
    }

    let output_offset_m = wg_idx_m * workgroup_size_m * tile_m + local_idx_m * tile_m;
    let output_offset_n = wg_idx_n * workgroup_size_n * tile_n + local_idx_n * tile_n;
    for (var inner_n = 0u; inner_n < tile_n; inner_n++) {
        for (var inner_m = 0u; inner_m < tile_m; inner_m++) {
            let global_m = output_offset_m + inner_m;
            let global_n = output_offset_n + inner_n;
            if (global_m >= params.m || global_n >= params.n) {
                continue;
            }
            let dst_offset = global_n * params.m + global_m;
            output[dst_offset] = acc[inner_n * tile_m + inner_m];
        }
    }
}