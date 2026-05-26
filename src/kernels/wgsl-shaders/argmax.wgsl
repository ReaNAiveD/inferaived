// argmax.wgsl — single-workgroup greedy argmax over a vocab-sized
// `array<f32>`. Used by `GpuSampler` as the terminal stage of the
// Backend::Gpu pipeline.
//
// One workgroup of 256 threads. Each thread strides through the input
// in steps of 256, accumulating its own (max, argmax) pair, then a
// shared-memory tree reduction combines them. Ties are broken on
// smallest index (deterministic, but not bit-identical to CPU greedy
// which prefers the last-seen max — ties are vanishingly rare on real
// logits, see argmax.rs for the rationale).

@group(0) @binding(0)
var<storage, read> logits: array<f32>;

// One-element output: `output[0]` receives the winning token id.
@group(0) @binding(1)
var<storage, read_write> output: array<u32>;

struct Params {
    vocab_size: u32,
};

@group(0) @binding(2)
var<uniform> params: Params;

const WG_SIZE: u32 = 256u;

var<workgroup> shared_vals: array<f32, WG_SIZE>;
var<workgroup> shared_idxs: array<u32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let vocab = params.vocab_size;

    // Stripe through the vocab: thread `tid` owns indices
    // tid, tid + WG_SIZE, tid + 2*WG_SIZE, ...
    var local_max: f32 = -3.40282347e+38;       // f32::MIN
    var local_idx: u32 = 0xFFFFFFFFu;           // sentinel: "no value seen"
    var i: u32 = tid;
    while (i < vocab) {
        let v = logits[i];
        // Strict > favors smaller index on ties (the first to set
        // local_max wins).
        if (v > local_max) {
            local_max = v;
            local_idx = i;
        }
        i = i + WG_SIZE;
    }
    shared_vals[tid] = local_max;
    shared_idxs[tid] = local_idx;
    workgroupBarrier();

    // Tree reduction in shared memory. At each step, the lower half of
    // active lanes consumes the upper half. Smaller index wins on ties.
    var stride: u32 = WG_SIZE >> 1u;
    while (stride > 0u) {
        if (tid < stride) {
            let other_val = shared_vals[tid + stride];
            let other_idx = shared_idxs[tid + stride];
            let mine_val = shared_vals[tid];
            let mine_idx = shared_idxs[tid];
            // Strict > for the value; on equality, prefer smaller idx.
            if (other_val > mine_val || (other_val == mine_val && other_idx < mine_idx)) {
                shared_vals[tid] = other_val;
                shared_idxs[tid] = other_idx;
            }
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if (tid == 0u) {
        output[0] = shared_idxs[0];
    }
}
