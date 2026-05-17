# Why a single-buffer GPU arena does not work under `wgpu`

Status: **decision** — keep `BufferView`, drop the single-buffer `GpuArena`
prototype that originally sat next to it in `src/buffer_arena.rs`. This note
records the constraint that killed the design so we don't relearn it next
time the idea looks attractive.

## What we wanted

A per-`forward()` bump arena backed by **one** `wgpu::Buffer`:

```text
[ resid | q_gate | q_norm_out | k | v | attn_out | mlp_gate | mlp_up | ... ]
  ^                                                                       ^
  bump=0 (reset at top of forward)                              capacity boundary
```

Motivation: each `LayerSession::forward` does roughly ten
`device.create_buffer` calls. Across the 24-layer stack that is ~240 buffer
creates per decoded token. A bump arena would turn that into "one big buffer
created once, bump cursor reset per forward, sub-regions handed out as
`BufferView`s". Same pattern as ggml / vLLM workspace allocators.

The arena prototype (`GpuArena` in `src/buffer_arena.rs`, since deleted)
worked in isolation: unit tests built `BindGroup`s from two disjoint
`alloc()`-returned views and `wgpu` accepted them. The trouble only appears
when those views are bound to a **single dispatch with mixed read/write
usage**.

## The constraint: per-buffer usage tracking

`wgpu` validates buffer state at the granularity of the **whole
`wgpu::Buffer`**, not byte ranges. The relevant code lives in
[`wgpu-core/src/track/buffer.rs`](https://github.com/gfx-rs/wgpu/blob/v28.0.1/wgpu-core/src/track/buffer.rs)
(version 28.0.1, cached locally at
`~/.cargo/registry/src/index.crates.io-*/wgpu-core-28.0.1/src/track/buffer.rs`).

State is stored as a `Vec<BufferUses>` indexed by `buffer.tracker_index()` —
one slot per buffer, no sub-range information. When a bind group references
a buffer, its `BufferUses` flags get merged into the usage scope by
`UsageScope::merge_single` (line ~197) which delegates to a `merge` helper
(line ~738):

```rust
let merged_state = *current_state | new_state;

if invalid_resource_state(merged_state) {
    return Err(ResourceUsageCompatibilityError::from_buffer(
        unsafe { metadata_provider.get(index) },
        *current_state,
        new_state,
    ));
}
```

`invalid_resource_state` returns true for any combination of writable
storage with another usage of the same buffer in the same usage scope (the
standard WebGPU "exclusive write" rule). The merge is purely bitwise OR
across all bindings of that buffer in the scope; byte offsets in
`BufferBinding` never participate.

Concretely, this means a dispatch whose bind group contains both:

- `binding=0`: `Storage { read_only: true }` → `BufferBinding { buffer: arena_buf, offset: 0, size: 4096 }`
- `binding=1`: `Storage { read_only: false }` → `BufferBinding { buffer: arena_buf, offset: 8192, size: 4096 }`

fails validation with a usage conflict, even though the two ranges are
disjoint. Every kernel in our forward that reads one tensor and writes
another (i.e. essentially all of them except true in-place ops) hits this
the moment both tensors live in the same arena.

This is not a `wgpu` bug. WebGPU was deliberately specified with
buffer-granularity tracking because finer ranges would force every backend
(D3D12, Metal, Vulkan, GLES) to do per-range hazard analysis. The
constraint propagates all the way up through `wgpu-core`.

## What we kept

`BufferView` — the non-owning `{ buffer, byte_offset, row_byte_size,
byte_stride_per_row, row_count }` descriptor — is genuinely useful
independent of the arena, because it lets a kernel read a strided slice of
a buffer that some other code created. The slice need not be tightly
packed, and the same kernel signature accepts:

- a whole standalone buffer (`BufferView::whole`),
- a row range of a persistent buffer like the KV cache (`BufferView::rows`),
- or a strided "every other head" view into a fused buffer
  (`BufferView::strided`).

The third constructor is what let us delete `SliceCopyWebgpu`: RoPE /
q_norm / attention now read Q directly from the fused
`[q_h0 | gate_h0 | q_h1 | gate_h1 | ...]` projection output, with per-head
byte stride `2 * head_dim * 4`. No copy dispatch, no scratch buffer.

So `BufferView` carried its own weight even after the arena was reverted.

## What we did not keep

`GpuArena` (bump allocator + `BindingKind` alignment + `alloc` / `alloc_2d`
/ `reset` / capacity tracking) is gone. The unit tests passed in isolation
but the type cannot be wired into `LayerSession::forward` without rewriting
every kernel to be in-place — which defeats the point.

Layers continue to call `device.create_buffer` for per-forward scratch.
This is the same behaviour we had before the BufferView work, just with
nicer wrappers around the resulting buffers.

## What would actually work

Two viable directions, both deferred:

1. **Multiple arenas + graph coloring.** Partition scratch tensors into
   `N` groups such that no single dispatch reads and writes tensors from
   the same group. Each group gets its own backing `wgpu::Buffer`. For our
   forward `N=3` is enough (input / output / persistent), bounded by the
   maximum number of distinct read/write roles any kernel exposes. Coloring
   can be precomputed once at session construction since the dispatch
   sequence is static. This is what production engines do.

2. **Sub-allocation via separate `wgpu::Buffer`s, pooled.** Keep a free-list
   of recently-released `wgpu::Buffer`s keyed by size class. `forward()`
   pops from the pool instead of calling `device.create_buffer`. This
   sidesteps the tracker entirely (different buffers, no usage merge) at
   the cost of more API objects. Cheaper to implement than option 1, weaker
   memory locality.

If we ever revisit this, the failure mode to look out for in tests is
**`ResourceUsageCompatibilityError`** from `wgpu`'s validation layer when
two bindings of the same buffer reach a single dispatch with mixed
read/write. It is reported with a clear "Buffer is missing the
`COPY_DST`/`STORAGE_BINDING`/..." prefix; if you see one of those during
arena work, the diagnosis is almost certainly per-buffer tracking biting,
not a real usage-flag bug.

## References

- `wgpu-core/src/track/buffer.rs`, version 28.0.1: `UsageScope`,
  `merge_single`, `merge`, `invalid_resource_state`.
- WebGPU spec §3.4.5 "Usage Scope" and §3.6.6.2 "Buffer Binding": each
  binding contributes the binding's full `usage` to the scope; conflicts
  are computed on the union per buffer.
- `bumpalo` for the `&self`-takes-allocation pattern used by `BufferView`'s
  arena constructors (now removed) — `Cell<u64>` for the bump cursor so
  multiple views with non-overlapping lifetimes can coexist without a
  `&mut` chain.
