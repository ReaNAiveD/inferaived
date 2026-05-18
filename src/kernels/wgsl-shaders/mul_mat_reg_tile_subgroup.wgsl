// Subgroup-optimised dense GEMM tile (N > 1). Companion to
// mul_mat_vec_decode.wgsl, which handles the GEMV (N = 1) case.
// Improvements over mul_mat_reg_tile.wgsl:
//   1. ROWS_PER_WG × COLS_PER_WG output cells computed per workgroup.
//   2. All threads of the workgroup cooperatively cache COLS_PER_WG input
//      rows for the current K-tile; weights are read directly from global
//      memory in u32-packed bf16 pairs (one u32 -> two weights, no k%2
//      branch).
//   3. subgroupAdd partial-sum reduction across the K-split + short
//      cross-subgroup tree reduction for the final write-back.
//
// Same bind-group layout as mul_mat_reg_tile.wgsl. Output is column-major
// with M contiguous: output[n * M + m].
//
// Dispatch:
//   wg_count_m = ceil(M / ROWS_PER_WG)
//   wg_count_n = ceil(N / COLS_PER_WG)
//   dispatch_workgroups(wg_count_m * wg_count_n, 1, 1)
//   wg_id.x = wg_idx_n * wg_count_m + wg_idx_m   (m varies fastest)
//
// Requires Features::SUBGROUP on the device; the `enable subgroups;`
// directive must be OMITTED on native wgpu (see
// https://github.com/gfx-rs/wgpu/issues/5555).

@group(0) @binding(0) var<storage, read> weight: array<u32>;        // M×K bf16, 2 per u32
@group(0) @binding(1) var<storage, read> input: array<f32>;          // N×K f32, row-major
@group(0) @binding(2) var<storage, read_write> output: array<f32>;   // N×M f32, col-major (M contiguous)

struct Params {
    m: u32,
    n: u32,
    k: u32,
    weight_row_stride: u32,   // bf16 elements per row
    input_row_stride: u32,    // f32 elements per row
}
@group(0) @binding(3) var<uniform> params: Params;

override workgroup_size: u32;

// Output sub-tile per workgroup. ROWS_PER_WG * COLS_PER_WG MUST be <=
// workgroup_size; only that many threads cooperate in the final write-back.
const ROWS_PER_WG: u32 = 4u;
const COLS_PER_WG: u32 = 4u;
const TILE_CELLS: u32 = ROWS_PER_WG * COLS_PER_WG;

// Two f32 inputs per thread per tile per N-row (one pair per packed u32 word).
var<workgroup> cached_input: array<f32, COLS_PER_WG * 2u * workgroup_size>;
// reduce_shmem[sg_id * TILE_CELLS + r * COLS_PER_WG + c]; sized for the
// worst case subgroup_size == 1 (in practice subgroup_size >= 8 so most
// entries are unused).
var<workgroup> reduce_shmem: array<f32, workgroup_size * ROWS_PER_WG * COLS_PER_WG>;

@compute @workgroup_size(workgroup_size)
fn main(
    @builtin(workgroup_id)          wg_id:    vec3<u32>,
    @builtin(local_invocation_id)   local_id: vec3<u32>,
    @builtin(subgroup_id)           sg_id:    u32,
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_size)         sg_size:  u32,
) {
    let thread_id    = local_id.x;

    let wg_count_m   = (params.m + ROWS_PER_WG - 1u) / ROWS_PER_WG;
    let wg_idx_m     = wg_id.x % wg_count_m;
    let wg_idx_n     = wg_id.x / wg_count_m;
    let wg_row_base  = wg_idx_m * ROWS_PER_WG;
    let wg_col_base  = wg_idx_n * COLS_PER_WG;

    // Per-thread accumulators for all TILE_CELLS output cells. Indexed
    // as partial[r * COLS_PER_WG + c]. Every thread must track ALL cells
    // so the subgroupAdd reductions are correct regardless of how
    // threads partition into subgroups.
    var partial: array<f32, TILE_CELLS>;
    for (var i = 0u; i < TILE_CELLS; i++) {
        partial[i] = 0.0;
    }

    let weight_words_per_row = params.weight_row_stride / 2u;
    let k_words              = (params.k + 1u) / 2u;

    for (var tile_word_base = 0u; tile_word_base < k_words; tile_word_base += workgroup_size) {
        let ki0 = tile_word_base + thread_id;
        let k0  = 2u * ki0;
        let k1  = k0 + 1u;

        // Cooperatively load COLS_PER_WG × 2 input floats per thread (one
        // pair per N-row).  cached_input layout (per N-row, then per
        // K-pair, contiguous on K):
        //   cached_input[c * 2*workgroup_size + 2*thread_id + {0,1}]
        for (var c = 0u; c < COLS_PER_WG; c++) {
            let global_col = wg_col_base + c;
            let base       = c * (2u * workgroup_size) + 2u * thread_id;
            let valid_col  = global_col < params.n;
            // Avoid OOB reads on input when the column is past N; the
            // resulting cell is masked at write-back anyway.
            let row_offset = select(0u, global_col * params.input_row_stride, valid_col);
            cached_input[base]      = select(0.0, input[row_offset + k0], valid_col && k0 < params.k);
            cached_input[base + 1u] = select(0.0, input[row_offset + k1], valid_col && k1 < params.k);
        }

        workgroupBarrier();

        if (ki0 < k_words) {
            // Each thread MACs its owned K-pair against ROWS_PER_WG rows
            // and COLS_PER_WG columns of the cached input.
            for (var r = 0u; r < ROWS_PER_WG; r++) {
                let global_row = wg_row_base + r;
                if (global_row < params.m) {
                    let word = weight[global_row * weight_words_per_row + ki0];
                    let w_lo = bitcast<f32>((word & 0xFFFFu) << 16u);
                    let w_hi = bitcast<f32>(word & 0xFFFF0000u);
                    for (var c = 0u; c < COLS_PER_WG; c++) {
                        let base = c * (2u * workgroup_size) + 2u * thread_id;
                        let in0  = cached_input[base];
                        let in1  = cached_input[base + 1u];
                        partial[r * COLS_PER_WG + c] += w_lo * in0 + w_hi * in1;
                    }
                }
            }
        }

        workgroupBarrier();
    }

    // Subgroup-wide reduction of each per-cell partial.
    for (var i = 0u; i < TILE_CELLS; i++) {
        let sg_sum = subgroupAdd(partial[i]);
        if (sg_lane == 0u) {
            reduce_shmem[sg_id * TILE_CELLS + i] = sg_sum;
        }
    }

    workgroupBarrier();

    // Cross-subgroup tree reduction + write-back. TILE_CELLS threads
    // each finalize one output cell.
    let num_sg = workgroup_size / sg_size;
    if (thread_id < TILE_CELLS) {
        let cell = thread_id;
        let r = cell / COLS_PER_WG;
        let c = cell % COLS_PER_WG;
        var total = 0.0;
        for (var s = 0u; s < num_sg; s++) {
            total += reduce_shmem[s * TILE_CELLS + cell];
        }
        let global_row = wg_row_base + r;
        let global_col = wg_col_base + c;
        if (global_row < params.m && global_col < params.n) {
            output[global_col * params.m + global_row] = total;
        }
    }
}
