# WGSL Shaders TODO

## Switch to multi-dimensional dispatch

**Affected shaders**: All shaders currently using 1D `gid.x` as the sole thread index (e.g. `get_rows.wgsl`, future shaders).

**Problem**: When total elements exceed `u32::MAX` (~4.3B), single-dimension indexing overflows. This can happen with 1M+ context lengths on large hidden sizes (e.g. 1M × 12288 = 12.9B elements).

**Solution**: Use 2D/3D dispatch to map different tensor dimensions to different `gid` axes:
- `gid.x` → column (ne0 / hidden_size)
- `gid.y` → row (token index)
- `gid.z` → batch or higher dimensions if needed
- `dispatch_workgroups(ceil(ne0 / wg_size), n_tokens, 1)`

This also eliminates expensive 4D index decoding arithmetic (repeated division/modulo chains).

## Switch to WebGPU native buffer offsets

**Affected shaders**: All shaders currently using manual byte offset calculations to access buffer data.

**Problem**: Manual byte offset calculations are error-prone and less efficient than using WebGPU's native buffer binding with offsets.

**Solution**: Use WebGPU's native buffer binding with offsets to directly access the relevant data without manual calculations.

## Use unified parameter naming

Previously, different shaders have used varying parameter names for similar concepts (e.g. `ne0`, `hidden_size`, `n_tokens`). This can lead to confusion and inconsistency across shaders.

## FlashAttention v2 for `causal_gqa_naive_attention`

**Affected shader**: `causal_gqa_naive_attention.wgsl` (currently a naive 3-pass implementation).

**Problem**: The naive kernel re-reads K (and recomputes scores) three times per `(q_token, q_head)` and is bandwidth-bound. Materializing the full attention matrix is also `O(seq_len^2)` memory, which becomes prohibitive past a few thousand tokens.

**Solution**: Rewrite as a FlashAttention v2-style kernel — outer loop over Q blocks, inner loop over K/V blocks, with online softmax updating `(m, ell, O)` incrementally in shared memory. This collapses to a single pass over K/V, never materializes the attention matrix, and avoids atomics (each workgroup owns its Q block exclusively).

References:
- [Online softmax](https://arxiv.org/abs/1805.02867) (Milakov & Gimelshein 2018) — math foundation.
- [FlashAttention](https://arxiv.org/abs/2205.14135) (Dao et al. 2022) — Algorithm 1.
- [FlashAttention v2](https://arxiv.org/abs/2307.08691) (Dao 2023) — the Q-outer / KV-inner reordering we want.
- Reference implementation: [Triton fused attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html).

Notes:
- We cannot do v3 (WGMMA / TMA require H100-class hardware not exposed by wgpu); v2 is the achievable ceiling.
- Should be a separate shader file (e.g. `causal_gqa_flash_attention.wgsl`) per the "many small single-purpose kernels" principle, not a flag on the existing kernel.
- Validate against the naive kernel within a small epsilon before swapping in.
