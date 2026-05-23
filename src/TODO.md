# Host-side TODO

Optimization directions for the Rust orchestration layer (kernels-side TODOs live in `wgsl-shaders/TODO.md`). Open items are ordered by priority; completed and dropped items are listed at the bottom.

## 1. GPU LM head + GPU sampler

- `LmHeadCpu` does the full `vocab_size × hidden_size` bf16 matmul on CPU per decode step, after a `hidden_size` f32 readback. With `vocab=248_320, hidden=1024` this is the dominant decode latency now that the per-token buffer-create overhead is gone (see Done: per-step Runner pattern).
- Direction: a `MulMatWebgpu`-shaped GPU pipeline that consumes bf16 weights directly, followed by an on-GPU top-k / sampler kernel. End state: per-step readback becomes a single `u32` token id instead of a 248K f32 logits vector.
- Subsumes the dropped "LM head SIMD" entry — once GPU LM head lands, the CPU path becomes a debug-only fallback that doesn't need further optimization.

## 2. Sampling beyond argmax

- Today only `ArgmaxSamplerCpu` exists, so generation from any base model degenerates to repeated high-prior tokens (current smoke test: `"Hello "` → `"1"` then four spaces). Hard to evaluate output quality without this.
- Direction: add temperature scaling, top-k, top-p (nucleus). Keep argmax as a `temperature == 0` special case. Sampler trait so the session takes `&mut dyn Sampler`.
- Coordinate with item 1: if the GPU sampler lands first, this becomes a GPU-kernel task instead of a CPU one.

## 3. Load model dims from `config.json`

- `main.rs` hardcodes `hidden_size`, `intermediate_size`, head dims, rope theta, layer-type pattern, etc. — every one of these is a silent footgun if we ever load another Qwen3-Next checkpoint. `max_seq_len` is also a hardcoded constant (`32`).
- Direction: parse `model/Qwen3.5-0.8B/config.json` via `serde_json` into a typed `Qwen35Config`, derive `LayerStackConfig` from it. Move `max_seq_len` into config / `Qwen35Session::new` arguments instead of a constant.
- Small, mechanical, unblocks loading any other Qwen3-Next checkpoint.

## 4. bf16 / smaller KV cache and hidden states

- `SelfAttentionLayerSession::{k_cache_buffer, v_cache_buffer}` are f32. At long context the KV cache dominates VRAM (24 layers × seq × 2 × kv_dim × 4 B). bf16 halves it for ~zero quality loss. Same applies to `hidden_states_buffer` and most scratch tensors.
- Direction: change storage to bf16 (`vec2<u32>` packed in WGSL), upcast to f32 at the FMA boundary inside kernels. Touches every kernel that reads/writes these tensors — large, one-shot churn.

## 5. Chat template + multi-turn UX

- `Qwen35Session` already supports `forward(input_ids, …)` with arbitrary `num_new_tokens`, which is exactly what multi-turn extend needs. But there's no glue: no chat-template formatting, no tokenizer-side streaming detokenization, no high-level `chat(user_message) -> assistant_message` API.
- Direction: load `chat_template.jinja`, apply via minijinja; wrap the tokenizer for incremental decode; expose a small `Chat` struct over `Qwen35Session`. Mostly meaningful once an instruct-tuned checkpoint is in.

## 6. Cached prefill runner

- `build_prefill_runner` in [language_model.rs](language_model.rs) re-runs `LayerStackSession::plan` on every prefill call (~240 `create_buffer` per call, since each layer's `plan` allocates ~10 scratch buffers sized to `num_new_tokens`). One-shot per chat turn rather than per token, so not on the hot path, but easy to drop.
- Direction: cache one prefill runner sized to `max_seq_len` and have each prefill call narrow the `BufferView` slot to the actual `num_new_tokens` instead of rebuilding the plan.
- Subsumes the residual "share scratch across layers" idea from the old ScratchArena entry — at the session/prefill-runner level there is no read/write aliasing problem because every layer's scratch is private to that layer.

## 7. Per-layer hidden-state dump infrastructure

- The dump / validation harness lives on the `feat/per-layer-dump-validation` branch from earlier work and hasn't been forward-ported through the session refactor. Pull back when needed, not before.
- Direction: when the next correctness-sensitive change comes (quantization, bf16 KV, new kernel), rebase / redo this on top of the current `LayerSession::forward` API so we can diff against a Hugging Face reference cheaply.

## 8. Quantization of weights (long term)

- bf16 weights = ~1.5 GB. Int8 sym halves it; int4 quarters it. Biggest absolute VRAM win, biggest engineering cost — defer until everything above either ships or is consciously skipped.
- Direction: per-row symmetric quantization for `MulMatWebgpu` and the GPU LM head. Needs new shader variants and a weight-loading path that quantizes on the fly (or consumes pre-quantized files).

---

## Done

- **`BufferView` for strided / offset access** *(2025)* — Implemented as `BufferView` in [src/buffer_view.rs](buffer_view.rs), with `whole` / `rows` / `strided` constructors and an `as_binding()` that folds the byte offset into a `wgpu::BufferBinding`. All kernels (`norm`, `sigmoid_mul`, `silu_mul`, `binary`, `mul_mat`, `rope`, `attention`) take `BufferView` arguments; offset/length arithmetic moved out of every shader uniform and into one place. Q-extract motivation resolved: `RmsNormInplaceWebgpu` / `RopeInplaceWebgpu` / `CausalGqaNaiveAttentionWebgpu` now read Q directly from the fused `q_gate_proj_buffer` via a strided view; `SliceCopyWebgpu` and its WGSL shader are deleted.
- **File-structure reorganization** — Kernels now live under [src/kernels/](kernels/) (one file per pipeline) with their shaders in [src/kernels/wgsl-shaders/](kernels/wgsl-shaders/). Layer-level orchestration lives under [src/layers/](layers/) (`layer_stack.rs`, `linear_attention.rs`, `self_attention.rs`, `mlp.rs`). The top level keeps model / session / sampler / config.
- **Per-step Runner / bake pattern** *(commit `3b38d3c`)* — Every kernel and layer moved from an encoder-and-submit `forward(device, queue, ...)` convenience to a two-stage `plan(...) -> Runner` API. `Qwen35Session::new` builds the decode runner once via `DecodeRig::build`; `decode_step` is now `queue.write_buffer` + `runner.forward(cpass)` with **zero `create_buffer` calls per token**. This closes the original "ScratchArena" item ("240 buffer creates / token in decode"); the residual prefill-side cost is tracked separately as item 6.

## Dropped (no longer worth doing)

- **ScratchArena with `N`-bucket graph coloring** — The headline motivation was "every `LayerSession::forward` call still does ~10 `device.create_buffer` (×24 layers ⇒ ~240 / token in decode)". The per-step Runner refactor moved all of those allocations to one-time session construction, so the per-token cost is now zero. The remaining VRAM-fragmentation argument doesn't justify a graph-coloring pass; if it ever does, the cheaper per-size-class free-list variant is the right starting point. See [docs/wgpu-single-buffer-arena.md](../docs/wgpu-single-buffer-arena.md) for the post-mortem on why a single arena buffer doesn't work under `wgpu`'s whole-buffer usage tracking.
- **LM head SIMD + pre-allocated logits buffer** — Was conditional on keeping the CPU LM head as a hot path. The plan of record is item 1 (GPU LM head + GPU sampler), which eliminates the per-step f32 readback entirely; once that ships, `LmHeadCpu` is a debug-only fallback and its inner-loop throughput stops mattering. Not worth the `half` / `wide` SIMD work in the interim.