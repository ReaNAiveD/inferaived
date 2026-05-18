// Decode-optimised skinny GEMV (N = 1). Improvements over mul_mat_vec.wgsl:
//   1. ROWS_PER_WG rows share one cooperative input-vector tile load.
//   2. Paired bf16 unpacking: one u32 -> two weights, no k%2 branch.
//   3. subgroupAdd partial-sum reduction + short cross-subgroup tree.
//
// Same bind-group layout as mul_mat_vec.wgsl. Dispatch: ceil(M / ROWS_PER_WG).
// Requires Features::SUBGROUP on the device; the `enable subgroups;` directive
// must be OMITTED on native wgpu (see https://github.com/gfx-rs/wgpu/issues/5555).

@group(0) @binding(0) var<storage, read> weight: array<u32>;        // M×K bf16, 2 per u32
@group(0) @binding(1) var<storage, read> input: array<f32>;          // K f32 (single row)
@group(0) @binding(2) var<storage, read_write> output: array<f32>;   // M f32

struct Params {
    m: u32,
    n: u32,                   // always 1
    k: u32,
    weight_row_stride: u32,   // bf16 elements per row
    input_row_stride: u32,    // unused
}
@group(0) @binding(3) var<uniform> params: Params;

override workgroup_size: u32;

// MUST divide workgroup_size evenly.
const ROWS_PER_WG: u32 = 4u;

// Two f32 inputs per thread per tile (one pair per u32 weight word).
var<workgroup> cached_input: array<f32, 2u * workgroup_size>;
// reduce_shmem[sg_id * ROWS_PER_WG + row]; sized for the worst case
// subgroup_size == 1 (in practice subgroup_size >= 8 so most entries are unused).
var<workgroup> reduce_shmem: array<f32, workgroup_size * ROWS_PER_WG>;

@compute @workgroup_size(workgroup_size)
fn main(
    @builtin(workgroup_id)          wg_id:   vec3<u32>,
    @builtin(local_invocation_id)   local_id: vec3<u32>,
    @builtin(subgroup_id)           sg_id:   u32,
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_size)         sg_size: u32,
) {
    let thread_id    = local_id.x;
    let wg_row_base  = wg_id.x * ROWS_PER_WG;

    // Every thread tracks ALL ROWS_PER_WG partials so subgroupAdd is correct
    // regardless of how threads are partitioned into subgroups.
    var partial: array<f32, ROWS_PER_WG>;
    for (var r = 0u; r < ROWS_PER_WG; r++) {
        partial[r] = 0.0;
    }

    let weight_words_per_row = params.weight_row_stride / 2u;
    let k_words = (params.k + 1u) / 2u;

    for (var tile_word_base = 0u; tile_word_base < k_words; tile_word_base += workgroup_size) {
        let ki0 = tile_word_base + thread_id;
        let k0  = 2u * ki0;
        let k1  = k0 + 1u;
        cached_input[2u * thread_id]      = select(0.0, input[k0], k0 < params.k);
        cached_input[2u * thread_id + 1u] = select(0.0, input[k1], k1 < params.k);

        workgroupBarrier();

        if (ki0 < k_words) {
            let in0 = cached_input[2u * thread_id];
            let in1 = cached_input[2u * thread_id + 1u];

            for (var r = 0u; r < ROWS_PER_WG; r++) {
                let global_row = wg_row_base + r;
                if (global_row < params.m) {
                    let word  = weight[global_row * weight_words_per_row + ki0];
                    let w_lo  = bitcast<f32>((word & 0xFFFFu) << 16u);
                    let w_hi  = bitcast<f32>(word & 0xFFFF0000u);
                    partial[r] += w_lo * in0 + w_hi * in1;
                }
            }
        }

        workgroupBarrier();
    }

    for (var r = 0u; r < ROWS_PER_WG; r++) {
        let sg_sum = subgroupAdd(partial[r]);
        if (sg_lane == 0u) {
            reduce_shmem[sg_id * ROWS_PER_WG + r] = sg_sum;
        }
    }

    workgroupBarrier();

    let num_sg = workgroup_size / sg_size;
    if (thread_id < ROWS_PER_WG) {
        let r = thread_id;
        var total = 0.0;
        for (var s = 0u; s < num_sg; s++) {
            total += reduce_shmem[s * ROWS_PER_WG + r];
        }
        let global_row = wg_row_base + r;
        if (global_row < params.m) {
            output[global_row] = total;
        }
    }
}
