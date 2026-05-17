// mul_mat_vec_decode.wgsl — Decode-optimised skinny GEMV (N = 1).
//
// Key improvements over mul_mat_vec.wgsl:
//
//  1. ROWS_PER_WG output rows share one cooperative input-vector load per K-tile,
//     reducing input DRAM traffic by ROWS_PER_WG× compared to the one-workgroup-
//     per-row dispatch in mul_mat_vec.wgsl.
//
//  2. bf16 weights are read two-at-a-time from each packed u32 word (lo + hi lanes),
//     eliminating the k%2 branch and issuing half as many weight load instructions.
//
//  3. K-partial reduction uses subgroupAdd (a single GPU instruction with no shared-
//     memory traffic inside the subgroup), followed by a O(log₂ num_subgroups)-step
//     tree over subgroup leaders in workgroup memory — far fewer barriers than the
//     full O(log₂ workgroup_size)-step tree in mul_mat_vec.wgsl.
//
// Bindings: identical to mul_mat_vec.wgsl — the same bind-group layout is reused.
//
// Dispatch: ceil(M / ROWS_PER_WG) workgroups, 1-D.
//   workgroup_count = (params.m + ROWS_PER_WG - 1) / ROWS_PER_WG
//
// Requires: wgpu Features::SUBGROUP (enable subgroups; is mandatory below).

enable subgroups;

// ── Bindings ──────────────────────────────────────────────────────────────────
@group(0) @binding(0) var<storage, read> weight: array<u32>;        // M×K bf16, 2 per u32
@group(0) @binding(1) var<storage, read> input: array<f32>;          // K f32 (single row)
@group(0) @binding(2) var<storage, read_write> output: array<f32>;   // M f32

struct Params {
    m: u32,
    n: u32,                   // always 1 for this shader
    k: u32,
    weight_row_stride: u32,   // bf16 elements per weight row  (== k_dim)
    input_row_stride: u32,    // unused — only one input row
}
@group(0) @binding(3) var<uniform> params: Params;

// ── Override / const parameters ───────────────────────────────────────────────
override workgroup_size: u32;   // total threads per workgroup (must be power of 2)

// Number of consecutive output rows each workgroup computes.
// Increasing this amortises the cooperative input-vector load across more rows.
// MUST divide workgroup_size evenly; 4 is a good default (32 threads/row with
// workgroup_size=128, nicely matching a 32-wide NVIDIA subgroup).
const ROWS_PER_WG: u32 = 4u;

// ── Workgroup memory ──────────────────────────────────────────────────────────
// Input-vector tile: each thread holds two consecutive f32 values per tile
// (the lo and hi f32 values that pair with one u32 weight word).
// Total tile width = 2 × workgroup_size k-elements.
var<workgroup> cached_input: array<f32, 2u * workgroup_size>;

// Cross-subgroup reduction scratch.
// Layout: reduce_shmem[subgroup_id * ROWS_PER_WG + row].
// Bound: workgroup_size subgroups is a safe upper bound (subgroup_size ≥ 1);
// in practice the GPU has subgroup_size ≥ 8 so far fewer entries are live.
var<workgroup> reduce_shmem: array<f32, workgroup_size * ROWS_PER_WG>;

// ── Main ──────────────────────────────────────────────────────────────────────
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

    // Per-thread partial sums — one accumulator per output row.
    // All threads compute partial sums for ALL ROWS_PER_WG rows so that
    // subgroupAdd(partial[r]) is always correct regardless of which threads
    // land in the same subgroup.
    var partial: array<f32, ROWS_PER_WG>;
    for (var r = 0u; r < ROWS_PER_WG; r++) {
        partial[r] = 0.0;
    }

    // Number of u32 words per weight row (2 bf16 per word).
    let weight_words_per_row = params.weight_row_stride / 2u;

    // ── Tile loop ──────────────────────────────────────────────────────────
    // Each tile covers 2×workgroup_size k-elements (workgroup_size u32 words).
    // Thread `t` is responsible for weight word (and input pair) at pair-index
    //   ki0 = tile_word_base + t   (a global index into the u32 weight row).
    let k_words = (params.k + 1u) / 2u;   // total u32 words per row (rounded up)
    for (var tile_word_base = 0u; tile_word_base < k_words; tile_word_base += workgroup_size) {

        // ── Step 1: Cooperative input-vector load ──────────────────────────
        // Thread t loads the two f32 values that correspond to its pair index.
        //   k0 = 2 * (tile_word_base + t)   (even k-element)
        //   k1 = k0 + 1                      (odd  k-element)
        // Out-of-range elements are filled with 0 so they contribute nothing.
        let ki0 = tile_word_base + thread_id;
        let k0  = 2u * ki0;
        let k1  = k0 + 1u;
        cached_input[2u * thread_id]      = select(0.0, input[k0], k0 < params.k);
        cached_input[2u * thread_id + 1u] = select(0.0, input[k1], k1 < params.k);

        workgroupBarrier();

        // ── Step 2: Partial dot products for all ROWS_PER_WG rows ─────────
        // Thread t reads ONE packed u32 word per output row and accumulates
        // both the lo (k0) and hi (k1) bf16 contributions.
        // The inner loop over ROWS_PER_WG is the "unrolled" dimension described
        // in the TODO: the compiler can fully unroll it (ROWS_PER_WG is const).
        if (ki0 < k_words) {
            let in0 = cached_input[2u * thread_id];
            let in1 = cached_input[2u * thread_id + 1u];

            for (var r = 0u; r < ROWS_PER_WG; r++) {
                let global_row = wg_row_base + r;
                if (global_row < params.m) {
                    // One u32 read covers both bf16 weights for k0 and k1.
                    let word  = weight[global_row * weight_words_per_row + ki0];
                    let w_lo  = bitcast<f32>((word & 0xFFFFu) << 16u);   // k0 weight
                    let w_hi  = bitcast<f32>(word & 0xFFFF0000u);         // k1 weight
                    partial[r] += w_lo * in0 + w_hi * in1;
                }
            }
        }

        workgroupBarrier();   // protect cached_input before the next tile load
    }

    // ── Step 3: Within-subgroup reduction via subgroupAdd ─────────────────
    // For each row r, subgroupAdd sums partial[r] across every lane in the
    // subgroup.  Because every thread tracks ALL ROWS_PER_WG partials (each
    // for the same global rows, just different k-slices), this reduction is
    // correct for ANY subgroup size — no thread carries a partial from a
    // "wrong" row.
    for (var r = 0u; r < ROWS_PER_WG; r++) {
        let sg_sum = subgroupAdd(partial[r]);
        if (sg_lane == 0u) {
            reduce_shmem[sg_id * ROWS_PER_WG + r] = sg_sum;
        }
    }

    workgroupBarrier();

    // ── Step 4: Cross-subgroup reduction & store ───────────────────────────
    // num_sg = workgroup_size / sg_size  (known at runtime).
    // Threads 0..ROWS_PER_WG-1 each own one output row and sum the subgroup
    // partial sums for that row in a tight sequential loop (≤ 32 iters for
    // workgroup_size=128 and minimum subgroup_size=4).
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
