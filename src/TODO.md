# Host-side TODO

Optimization directions for the Rust orchestration layer (kernels-side TODOs live in `wgsl-shaders/TODO.md`). Listed in rough priority order.

## ScratchPool for per-forward scratch buffers  *(done — 2026-05)*

- Implemented as `ScratchPool` in [`src/scratch_pool.rs`](scratch_pool.rs): one `wgpu::Buffer` per `ScratchSlot` (14 slots covering the union of self-attention, linear-attention, and MLP scratch tensors), each sized to `max_seq_len × feature_dim(slot) × 4`. Owned by `Qwen35Session`, plumbed through `LayerSession::forward` as `&ScratchPool`. Layers call `scratch.view_2d(slot, num_tokens)` or `scratch.buffer(slot)` instead of `device.create_buffer` — eliminates ~222 per-token buffer creates across the 24-layer stack.
- Why named slots, not a bump arena: `wgpu`'s validation tracks buffer state at whole-buffer granularity, so a single arena with disjoint sub-ranges as read+write on the same dispatch trips `ResourceUsageCompatibilityError`. One distinct `wgpu::Buffer` per slot sidesteps the merge entirely. See [`docs/wgpu-single-buffer-arena.md`](../docs/wgpu-single-buffer-arena.md) for the failed bump-arena prototype.
- Performance: decode tok/s unchanged within run-to-run noise on the test hardware — buffer creates were not the bottleneck on this driver (see `benches/baselines.local.csv` row `master-scratch-pool`). The change is structural prep for downstream items (bf16 scratch, pre-built bind groups, possible liveness coloring).
- Deferred follow-ups:
  - **Liveness coloring → N < 14 physical buffers.** Saves ~70% of scratch memory at large `max_seq_len`. Needs a hand-authored or recorded slot→bucket table and a debug-mode validator that walks an annotated dispatch sequence to assert no read+write conflicts per dispatch. Both layer types' dispatch sequences 3-color cleanly per the design report; the `view_2d()` / `buffer()` accessors are the choke point and would absorb the offset arithmetic transparently. Wait until scratch memory shows up in a real profile.
  - **Pre-built bind groups.** Now that scratch buffers are session-lifetime, every kernel's bind group could also be built once at session construction instead of every dispatch — likely a bigger win than the buffer-create elimination itself.

## `BufferView` for strided / offset access  *(done — 2025)*

- Implemented as `BufferView { buffer, byte_offset, row_byte_size, byte_stride_per_row, row_count }` in `src/buffer_arena.rs`, with `whole` / `rows` / `strided` constructors and an `as_binding()` that folds the byte offset into a `wgpu::BufferBinding`.
- All kernels (`norm`, `sigmoid_mul`, `silu_mul`, `binary`, `mul_mat`, `rope`, `attention`) take `BufferView` arguments; offset/length arithmetic moved out of every shader uniform and into one place.
- Resolution of the original q-extract motivation: `RmsNormInplaceWebgpu` / `RopeInplaceWebgpu` / `CausalGqaNaiveAttentionWebgpu` now take a `q_head_byte_stride: u32` and read Q directly from the fused `q_gate_proj_buffer` via a strided `BufferView`. `SliceCopyWebgpu` and its WGSL shader have been deleted.

## Load model dims from `config.json`

- `main.rs` hardcodes `hidden_size`, `intermediate_size`, head dims, rope theta, layer-type pattern, etc. — every one of these is a silent footgun if we ever load another Qwen3-Next checkpoint.
- Direction: parse `model/Qwen3.5-0.8B/config.json` via `serde_json` into a typed `Qwen35Config`, derive `LayerStackConfig` from it. Also move `max_seq_len` (currently hardcoded `32`) into config / `Qwen35Session::new` arguments instead of a constant.

## Sampling beyond argmax

- Today only `ArgmaxSamplerCpu` exists, so generation from any base model degenerates to repeated high-prior tokens (current smoke test: `"Hello "` → `"1"` then four spaces). Hard to evaluate output quality this way.
- Direction: add temperature scaling, top-k, top-p (nucleus). Keep argmax as a `temperature == 0` special case. Sampler trait so the session takes `&mut dyn Sampler`.

## GPU LM head + GPU sampler

- `LmHeadCpu` does the full `vocab_size × hidden_size` bf16 matmul on CPU per decode step, after a `hidden_size` f32 readback. With `vocab=248_320, hidden=1024` this is the dominant decode latency once kernels are tight.
- Direction: a `MulMatWebgpu`-shaped GPU pipeline that consumes bf16 weights directly, followed by an on-GPU top-k / sampler kernel. End state: per-step readback becomes a single `u32` token id instead of a 248K f32 logits vector. Subsumes the next two items.

## LM head SIMD + pre-allocated logits buffer

- Only relevant if we keep the CPU LM head path around as a reference / debug fallback. If GPU LM head lands, delete this entry.
- Direction (CPU fallback): `half::slice::HalfFloatSliceExt::convert_to_f32_slice` + `wide::f32x8` for the inner dot product, enable `half`'s `use-intrinsics` feature. Have `compute()` write into a caller-supplied `&mut [f32]` instead of returning `Vec<f32>` (~1 MB alloc/step).

## bf16 / smaller KV cache and hidden states

- `SelfAttentionLayerSession::{k_cache_buffer, v_cache_buffer}` are f32. At long context the KV cache dominates VRAM (24 layers × seq × 2 × kv_dim × 4 B). bf16 halves it for ~zero quality loss. Same applies to `hidden_states_buffer` and most scratch tensors.
- Direction: change storage to bf16 (`vec2<u32>` packed in WGSL or a half extension), upcast to f32 at the FMA boundary inside kernels. Needs touching every kernel that reads/writes these tensors, so wait until ScratchArena + BufferView land so the API churn is one-shot.

## File-structure reorganization

- `src/` is currently flat with ~15 sibling files mixing orchestration (`language_model.rs`, `layer_loop.rs`, `main.rs`) with low-level kernels (`mul_mat.rs`, `norm.rs`, `rope.rs`, `delta_rule.rs`, `conv_silu.rs`, `gated_rms_norm.rs`, `sigmoid_mul.rs`, `binary.rs`, `mamba_scan.rs`, …).
- Direction: pull kernels into `src/kernels/` (one file per pipeline), keep the top level for model / session / sampler / config. Likely also: `src/layers/` for `linear_attention.rs` + `self_attention.rs` + `layer_stack.rs` + `layer_session.rs` split out from the current `layer_loop.rs`. Best done before the bf16 churn so renames don't collide.

## Quantization of weights (long term)

- bf16 weights = ~1.5 GB. Int8 sym halves it; int4 quarters it. Biggest absolute VRAM win, biggest engineering cost.
- Direction: per-row symmetric quantization for `MulMatWebgpu` and the GPU LM head. Needs new shader variants and a weight-loading path that quantizes on the fly (or consumes pre-quantized files).

## Chat template + multi-turn UX

- `Qwen35Session` already supports `forward(input_ids, …)` with arbitrary `num_new_tokens`, which is exactly what multi-turn extend needs. But there's no glue: no chat-template formatting, no tokenizer-side streaming detokenization, no high-level `chat(user_message) -> assistant_message` API.
- Direction: load `chat_template.jinja`, apply via minijinja; wrap the tokenizer for incremental decode; expose a small `Chat` struct over `Qwen35Session`. Mostly meaningful once an instruct-tuned checkpoint is in.

## Per-layer hidden-state dump infrastructure

- The dump / validation harness lives on the `feat/per-layer-dump-validation` branch from earlier work and hasn't been forward-ported through the session refactor.
- Direction: when the next correctness-sensitive change comes (quantization, bf16 KV, new kernel), rebase / redo this on top of the current `LayerSession::forward` API so we can diff against a Hugging Face reference cheaply.