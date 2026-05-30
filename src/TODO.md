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

## 3. Load model dims from `config.json` — partial

- **Done:** `Qwen35Config::from_json` / `from_json_file` in [language_model/config.rs](language_model/config.rs) parse `model/Qwen3.5-0.8B/config.json` via `serde_json` (`Qwen35Config` derives `Deserialize` directly, with a `ConfigLoadError` enum and a unit test against the shipped file). Both `examples/generate.rs` and `examples/bench_decode.rs` now load the config from disk instead of hardcoding it.
- **Still to do:** `max_seq_len` lives on the caller (`Qwen35Session::new` argument); no change needed there, but a multi-arch refactor (per item 8a) will need to wrap `Qwen35Config` in a `ModelArch` enum.

## 4. bf16 / smaller KV cache and hidden states

- `SelfAttentionLayerSession::{k_cache_buffer, v_cache_buffer}` are f32. At long context the KV cache dominates VRAM (24 layers × seq × 2 × kv_dim × 4 B). bf16 halves it for ~zero quality loss. Same applies to `hidden_states_buffer` and most scratch tensors.
- Direction: change storage to bf16 (`vec2<u32>` packed in WGSL), upcast to f32 at the FMA boundary inside kernels. Touches every kernel that reads/writes these tensors — large, one-shot churn.

## 5. Chat template + multi-turn UX

- `Qwen35Session` already supports `forward(input_ids, …)` with arbitrary `num_new_tokens`, which is exactly what multi-turn extend needs. But there's no glue: no chat-template formatting, no tokenizer-side streaming detokenization, no high-level `chat(user_message) -> assistant_message` API.
- Direction: load `chat_template.jinja`, apply via minijinja; wrap the tokenizer for incremental decode; expose a small `Chat` struct over `Qwen35Session`. Mostly meaningful once an instruct-tuned checkpoint is in.

## 6. Cached prefill runner

- `build_prefill_runner` re-runs `LayerStackSession::plan` on every prefill call (~240 `create_buffer` per call, since each layer's `plan` allocates ~10 scratch buffers sized to `num_new_tokens`). One-shot per chat turn rather than per token, so not on the hot path, but easy to drop.
- ~~Previous direction:~~ "cache one prefill runner sized to `max_seq_len` and have each prefill call narrow the `BufferView` slot to the actual `num_new_tokens`". This is now blocked because `min_storage_buffer_offset_alignment` (32 on most desktop GPUs) makes narrowing into a strided sub-slice fail bind-group validation when the offset isn't 32-aligned (see the offset-0 fix in `Qwen35GpuSession::step` and the `prefill_tokens` / `prefill_hidden` field docs). The prefill staging buffers are now allocated per-call at the delta's actual size, which also makes the cached-runner idea less attractive (the buffers it would hold would still need to match `num_new`).
- Current direction (if worth pursuing): cache the per-layer scratch *shapes* and re-bind on each prefill, rather than re-planning. Most of the per-call cost is in the ~240 layer-scratch `create_buffer` calls, not in the plan logic itself.
- Subsumes the residual "share scratch across layers" idea from the old ScratchArena entry — at the session/prefill-runner level there is no read/write aliasing problem because every layer's scratch is private to that layer.

## 7. Per-layer hidden-state dump infrastructure

- The dump / validation harness lives on the `feat/per-layer-dump-validation` branch from earlier work and hasn't been forward-ported through the session refactor. Pull back when needed, not before.
- Direction: when the next correctness-sensitive change comes (quantization, bf16 KV, new kernel), rebase / redo this on top of the current `LayerSession::forward` API so we can diff against a Hugging Face reference cheaply.

## 8. New-model adoption: MiniCPM5-1B → Ministral-3-3B

Multi-model support, ordered by integration cost. Each step also exercises and hardens the generic loader/config scaffolding from items 3 and 5.

### 8a. `openbmb/MiniCPM5-1B` (first)

- Why first: pure `LlamaForCausalLM`, **strict subset** of what the engine already runs for Qwen3.5-0.8B. 24 layers × `hidden=1536` × `intermediate=4608`, GQA **16 Q / 2 KV** (same ratio as current Qwen3.5), `head_dim=128`, full RoPE θ=5M, RMSNorm + SwiGLU, vocab 130,560, bf16. Apache-2.0. Released May 2026 by OpenBMB (mid-tier but credible: MiniCPM-V lineage).
- Engineering deltas vs current Qwen3.5 path:
  - All 24 layers are `full_attention` → drop the linear-attention layer path; `delta_rule` / `mamba_scan` / `conv_silu` kernels go unused (still needed for Qwen3.5).
  - No `attn_output_gate` → simpler attention block than current Qwen3.5 (which uses `attn_output_gate=true`).
  - Full rotary, no `partial_rotary_factor` and no `mrope_interleaved` → simpler RoPE call.
  - `tie_word_embeddings: false` → LM head needs its own weight tensor (current Qwen3.5 ties them); small loader change.
  - Instruct-tuned with Jinja chat template + Think/No-Think modes → first model that will produce useful chat output (current smoke test degenerates to `"1    "`).
- Hard prerequisites: item 3 (typed `config.json` loader, so `Qwen35Config` generalizes into a per-architecture config enum) and item 5 (chat template + tokenizer-side streaming detokenize, so the instruct + thinking modes actually pay off).
- Step sequence:
  1. Land item 3. Introduce a `ModelArch` enum (`Qwen35`, `MiniCPM5`) + per-arch `Config` struct that emits a `LayerStackConfig`.
  2. Add `MiniCPM5Config` that emits 24 `full_attention` layers, no output gate, no partial rotary.
  3. Wire untied LM head path (load `lm_head.weight` as a separate tensor instead of reusing `embed_tokens.weight`).
  4. Land item 5 (chat template via minijinja + streaming detokenize).
  5. Smoke-test generate + chat on MiniCPM5-1B, compare logits against HF reference at a few token positions (use item 7's dump harness).

### 8b. `mistralai/Ministral-3-3B-Base-2512` (second)

- Why second: tier-1 lab (Mistral AI, Dec 2025), validates the engine on a frontier checkpoint and breaks the "everything is Llama" assumption in a controlled way. Text decoder is still standard RMSNorm / SwiGLU / GQA / RoPE — no new attention variant — but ships wrapped in `Mistral3ForConditionalGeneration` (3.4B LM + 0.4B vision encoder), uses `mistral-common` tokenizer (not HF-style), 256k context, BF16 native.
- Engineering deltas vs MiniCPM5-1B:
  - Weight loader has to **skip the vision encoder tensors** in the safetensors index and load only the text-decoder shard.
  - **`mistral-common` tokenizer**: either depend on a Rust port, or pre-tokenize with the official Python tokenizer and ship a fixed `tokenizer.json`-equivalent. Investigate before committing.
  - Use the `*-Base-2512` checkpoint, not `*-Instruct-2512` (Instruct ships **FP8 only**, would require an FP8→BF16 dequant on load that the loader doesn't currently do). Defer the Instruct variant until item 8 (quantization) ships, then revisit native FP8 support.
  - Base = no chat template, no thinking mode → demotes item-5-style chat UX work; this step is a "raw generate" milestone.
- Step sequence:
  1. Extend the `ModelArch` enum with `Ministral3Base`.
  2. Loader: read `safetensors.index.json`, ignore all tensors whose name doesn't start with the text-decoder prefix.
  3. Tokenizer: spike `mistral-common`-compatible tokenization (Rust port vs pre-tokenize offline). Pick one based on what's available at adoption time.
  4. Smoke-test continuation generation on the base model; compare logits against HF reference.

### 8c. Stretch goal: `google/gemma-4-E2B-it`

- Tier-1 (Google DeepMind), May 2026, but architecturally novel: per-layer embeddings (PLE), interleaved sliding+global attention with **different RoPE per layer-type** (sliding θ=10k vs global p-RoPE θ=1M, partial 0.25), `num_kv_shared_layers=20` (cross-layer KV cache sharing), `gelu_pytorch_tanh` MLP, double-wide MLP, GQA 8:1, final logit softcapping, multimodal (text+image+audio).
- Each of those is a new kernel or new orchestration mechanism. Land 8a and 8b first to validate the multi-arch scaffolding under non-trivial but tractable architectures before taking this on.

## 9. Quantization of weights (long term)

- bf16 weights = ~1.5 GB. Int8 sym halves it; int4 quarters it. Biggest absolute VRAM win, biggest engineering cost — defer until everything above either ships or is consciously skipped.
- Direction: per-row symmetric quantization for `MulMatWebgpu` and the GPU LM head. Needs new shader variants and a weight-loading path that quantizes on the fly (or consumes pre-quantized files).
- Connects to item 8b: native FP8 weight loading would unblock the `Ministral-3-3B-Instruct-2512` checkpoint (currently FP8-only, deferred to base for that reason).

---

## Done

- **`BufferView` for strided / offset access** *(2025)* — Implemented as `BufferView` in [src/buffer_view.rs](buffer_view.rs), with `whole` / `rows` / `strided` constructors and an `as_binding()` that folds the byte offset into a `wgpu::BufferBinding`. All kernels (`norm`, `sigmoid_mul`, `silu_mul`, `binary`, `mul_mat`, `rope`, `attention`) take `BufferView` arguments; offset/length arithmetic moved out of every shader uniform and into one place. Q-extract motivation resolved: `RmsNormInplaceWebgpu` / `RopeInplaceWebgpu` / `CausalGqaNaiveAttentionWebgpu` now read Q directly from the fused `q_gate_proj_buffer` via a strided view; `SliceCopyWebgpu` and its WGSL shader are deleted.
- **File-structure reorganization** — Kernels now live under [src/kernels/](kernels/) (one file per pipeline) with their shaders in [src/kernels/wgsl-shaders/](kernels/wgsl-shaders/). Layer-level orchestration lives under [src/layers/](layers/) (`layer_stack.rs`, `linear_attention.rs`, `self_attention.rs`, `mlp.rs`). The top level keeps model / session / sampler / config.
- **Per-step Runner / bake pattern** *(commit `3b38d3c`)* — Every kernel and layer moved from an encoder-and-submit `forward(device, queue, ...)` convenience to a two-stage `plan(...) -> Runner` API. `Qwen35Session::new` builds the decode runner once via `DecodeRig::build`; `decode_step` is now `queue.write_buffer` + `runner.forward(cpass)` with **zero `create_buffer` calls per token**. This closes the original "ScratchArena" item ("240 buffer creates / token in decode"); the residual prefill-side cost is tracked separately as item 6.

## Dropped (no longer worth doing)

- **ScratchArena with `N`-bucket graph coloring** — The headline motivation was "every `LayerSession::forward` call still does ~10 `device.create_buffer` (×24 layers ⇒ ~240 / token in decode)". The per-step Runner refactor moved all of those allocations to one-time session construction, so the per-token cost is now zero. The remaining VRAM-fragmentation argument doesn't justify a graph-coloring pass; if it ever does, the cheaper per-size-class free-list variant is the right starting point. See [docs/wgpu-single-buffer-arena.md](../docs/wgpu-single-buffer-arena.md) for the post-mortem on why a single arena buffer doesn't work under `wgpu`'s whole-buffer usage tracking.
- **LM head SIMD + pre-allocated logits buffer** — Was conditional on keeping the CPU LM head as a hot path. The plan of record is item 1 (GPU LM head + GPU sampler), which eliminates the per-step f32 readback entirely; once that ships, `LmHeadCpu` is a debug-only fallback and its inner-loop throughput stops mattering. Not worth the `half` / `wide` SIMD work in the interim.