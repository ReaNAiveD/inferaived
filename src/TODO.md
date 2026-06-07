# Host-side TODO

Optimization directions for the Rust orchestration layer (kernels-side TODOs live in `wgsl-shaders/TODO.md`). Open items are ordered by priority; completed and dropped items are listed at the bottom.

## 1. Sampling beyond argmax

- `Sampler::Argmax` is the only variant in [src/sampling.rs](sampling.rs) and `GpuSampler` ([src/gpu_sampler.rs](gpu_sampler.rs)) does GPU argmax only — `SamplingParams::{temperature, top_k, top_p}` are unused. Greedy is enough for the current correctness anchors but degrades quality on open-ended generation.
- Direction: extend the `GpuSampler` kernel (or stack a small `LogitsProcessor` GPU pass in front of it) for temperature scaling, top-k, and top-p. Keep argmax as the `temperature == 0` special case; route through the existing `Sampler` enum.
- The CPU `Sampler` / `LogitsProcessor` scaffolding in `sampling.rs` already encodes the API shape — the work is the WGSL plus the GPU-side prefix-sum / partial-sort.

## 2. bf16 / smaller KV cache and hidden states

- `Qwen35SelfAttentionLayerSession::{k_cache_buffer, v_cache_buffer}` (and the equivalent `MiniCPM5SelfAttentionLayerSession` fields, plus the parallel-path `MiniCPM5MaskedSelfAttentionLayerSession::{k_pool, v_pool}`) are f32. At long context the KV cache dominates VRAM (24 layers × seq × 2 × kv_dim × 4 B). bf16 halves it for ~zero quality loss. Same applies to `hidden_states_buffer` and most scratch tensors.
- Direction: change storage to bf16 (`vec2<u32>` packed in WGSL), upcast to f32 at the FMA boundary inside kernels. Touches every kernel that reads/writes these tensors — large, one-shot churn. Land alongside item 6 (`&mut wgpu::Buffer`) so the two signature waves combine.

## 3. Cached prefill runner

- `MiniCPM5GpuWorkspace::plan_prefill` / `Qwen35GpuWorkspace::plan_prefill` re-run `LayerStackSession::plan` on every prefill call (~240 `create_buffer` per call, since each layer's `plan` allocates ~10 scratch buffers sized to `num_new_tokens`). One-shot per chat turn rather than per token, so not on the hot path, but easy to drop.
- ~~Previous direction:~~ "cache one prefill runner sized to `max_seq_len` and have each prefill call narrow the `BufferView` slot to the actual `num_new_tokens`". Blocked because `min_storage_buffer_offset_alignment` (32 on most desktop GPUs) makes narrowing into a strided sub-slice fail bind-group validation when the offset isn't 32-aligned (see the offset-0 fix in `Qwen35GpuSession::step` and the `prefill_tokens` / `prefill_hidden` field docs). The prefill staging buffers are now allocated per-call at the delta's actual size inside `plan_prefill`, which also makes the cached-runner idea less attractive (the buffers it would hold would still need to match `num_new`).
- Current direction (if worth pursuing): cache the per-layer scratch *shapes* and re-bind on each prefill, rather than re-planning. Most of the per-call cost is in the ~240 layer-scratch `create_buffer` calls, not in the plan logic itself.
- Subsumes the residual "share scratch across layers" idea from the old ScratchArena entry — at the session/prefill-runner level there is no read/write aliasing problem because every layer's scratch is private to that layer.

## 4. Per-layer hidden-state dump infrastructure

- The dump / validation harness lives on the `feat/per-layer-dump-validation` branch from earlier work and hasn't been forward-ported through the session refactor. Pull back when needed, not before.
- Direction: when the next correctness-sensitive change comes (quantization, bf16 KV, new kernel), rebase / redo this on top of the current `LayerSession::forward` API so we can diff against a Hugging Face reference cheaply.

## 5. New-model adoption: Ministral-3-3B → Gemma-4

Multi-model support beyond the currently-supported Qwen3.5 / MiniCPM5 pair. Ordered by integration cost; the loader scaffolding hardened during MiniCPM5 adoption is the baseline (see Done).

### 5a. `mistralai/Ministral-3-3B-Base-2512` (first)

- Why first: tier-1 lab (Mistral AI, Dec 2025), validates the engine on a frontier checkpoint and breaks the "everything is Llama" assumption in a controlled way. Text decoder is still standard RMSNorm / SwiGLU / GQA / RoPE — no new attention variant — but ships wrapped in `Mistral3ForConditionalGeneration` (3.4B LM + 0.4B vision encoder), uses `mistral-common` tokenizer (not HF-style), 256k context, BF16 native.
- Engineering deltas vs MiniCPM5-1B:
  - Weight loader has to **skip the vision encoder tensors** in the safetensors index and load only the text-decoder shard.
  - **`mistral-common` tokenizer**: either depend on a Rust port, or pre-tokenize with the official Python tokenizer and ship a fixed `tokenizer.json`-equivalent. Investigate before committing.
  - Use the `*-Base-2512` checkpoint, not `*-Instruct-2512` (Instruct ships **FP8 only**, would require an FP8→BF16 dequant on load that the loader doesn't currently do). Defer the Instruct variant until item 7 (quantization) ships, then revisit native FP8 support.
  - Base = no chat template, no thinking mode — this step is a "raw generate" milestone.
- Step sequence:
  1. Add `Ministral3Config` next to `MiniCPM5Config` / `Qwen35Config`.
  2. Loader: read `safetensors.index.json`, ignore all tensors whose name doesn't start with the text-decoder prefix.
  3. Tokenizer: spike `mistral-common`-compatible tokenization (Rust port vs pre-tokenize offline). Pick one based on what's available at adoption time.
  4. Smoke-test continuation generation on the base model; compare logits against HF reference.

### 5b. Stretch goal: `google/gemma-4-E2B-it`

- Tier-1 (Google DeepMind), May 2026, but architecturally novel: per-layer embeddings (PLE), interleaved sliding+global attention with **different RoPE per layer-type** (sliding θ=10k vs global p-RoPE θ=1M, partial 0.25), `num_kv_shared_layers=20` (cross-layer KV cache sharing), `gelu_pytorch_tanh` MLP, double-wide MLP, GQA 8:1, final logit softcapping, multimodal (text+image+audio).
- Each of those is a new kernel or new orchestration mechanism. Land 5a first to validate the multi-arch scaffolding under a non-trivial but tractable architecture before taking this on.

## 6. Enforce write/read isolation on `wgpu::Buffer` via `&mut`

- Today every kernel and planner takes `&wgpu::Buffer` for both inputs and outputs (e.g. `MaskedGpuWorkspace::plan_kv` takes `input_hidden: &wgpu::Buffer` even though the layer stack writes residuals into it; `plan_kv_prefill` allocates a fresh `prefill_hidden` and hands it out as `&wgpu::Buffer` to a function that mutates it). Nothing in the type system prevents passing the same buffer as two outputs of one dispatch, or reading a buffer that an earlier stage is still writing to.
- Direction: switch every "this kernel writes here" parameter from `&wgpu::Buffer` to `&mut wgpu::Buffer`. `wgpu::Buffer` is `Arc`-backed so `Clone` is still available where one buffer is legitimately both an output and a later input — the caller takes one borrow, finishes the writing dispatch's planning, then takes a shared borrow for the read. Where Rust's single-mutable rule actually bites, we'll see it at the planning call site and can split the buffer into separate ones.
- A typed `BufferWrite` / `BufferRead` wrapper is a possible follow-up if `&mut wgpu::Buffer` proves insufficient, but is out of scope until we hit a concrete case it doesn't cover.
- Cost: invasive — every kernel `plan(...)` signature changes. Land alongside item 2 (bf16 KV cache touches all those signatures anyway) so the two churn waves combine.

## 7. Quantization of weights (long term)

- bf16 weights = ~1.5 GB. Int8 sym halves it; int4 quarters it. Biggest absolute VRAM win, biggest engineering cost — defer until everything above either ships or is consciously skipped.
- Direction: per-row symmetric quantization for `MulMatWebgpu` and the GPU LM head. Needs new shader variants and a weight-loading path that quantizes on the fly (or consumes pre-quantized files).
- Connects to item 5a: native FP8 weight loading would unblock the `Ministral-3-3B-Instruct-2512` checkpoint (currently FP8-only, deferred to base for that reason).

---

## Done

- **GPU LM head + GPU sampler** — `LmHeadWebgpu` ([src/lm_head.rs](lm_head.rs)) does the `vocab × hidden` matmul on GPU as part of the planned compute pass; `GpuSampler` ([src/gpu_sampler.rs](gpu_sampler.rs)) follows with a GPU argmax. The per-step readback shrank from a `vocab` f32 logits vector to a single `u32` token id (see `MiniCPM5GpuSession::run_and_read_back_token` and the Qwen35 equivalent). Top-k / top-p / temperature on top of the GPU sampler is tracked as item 1.
- **Typed `config.json` loaders for every supported model** — `Qwen35Config` ([src/language_model/qwen35/config.rs](language_model/qwen35/config.rs)) and `MiniCPM5Config` ([src/language_model/minicpm5/config.rs](language_model/minicpm5/config.rs)) both expose `from_json` / `from_json_file` (derives `Deserialize` with a `ConfigLoadError` enum). All four examples (`generate`, `bench_decode`, `chat_qwen35`, `chat_minicpm5`, `parallel_minicpm5`) load config from disk.
- **Chat template + multi-turn UX** — `examples/chat_minicpm5.rs` and `examples/chat_qwen35.rs` load each model's `chat_template.jinja` via minijinja (with `unknown_method_callback` + `raise_exception`), render an append-only delta per turn so subsequent prompts feed straight into the persistent KV cache, and stream tokens with on-the-fly detokenization. Multi-turn `/reset` returns the session to a fresh state. The parallel-context REPL (`examples/parallel_minicpm5.rs`) uses the same jinja path on top of the parallel session.
- **MiniCPM5-1B adoption** — Full `LlamaForCausalLM` path landed under [src/language_model/minicpm5/](language_model/minicpm5/) (`config.rs`, `core.rs`, `gpu.rs`) and [src/layers/minicpm5_self_attention.rs](layers/minicpm5_self_attention.rs) / [src/layers/minicpm5_layer_stack.rs](layers/minicpm5_layer_stack.rs). 24 full-attention layers, no output gate, full rotary, untied LM head; `chat_minicpm5` example is the smoke test.
- **Parallel context windows for MiniCPM5** — Precomputed Parallel Context Windows + Prompt-Cache-style compile/reuse: multiple context windows encoded once into the shared per-layer KV cache and reused by a rewindable generation stream that can be re-begun for many sessions. Lives in [src/language_model/minicpm5/parallel.rs](language_model/minicpm5/parallel.rs), [src/layers/minicpm5_masked_layer_stack.rs](layers/minicpm5_masked_layer_stack.rs), [src/parallel_window.rs](parallel_window.rs), [src/kernels/masked_block_attention.rs](kernels/masked_block_attention.rs); `examples/parallel_minicpm5.rs` is the interactive REPL with CLI-driven `--prefix` / `--context` / `/reset N..`.

- **`BufferView` for strided / offset access** *(2025)* — Implemented as `BufferView` in [src/buffer_view.rs](buffer_view.rs), with `whole` / `rows` / `strided` constructors and an `as_binding()` that folds the byte offset into a `wgpu::BufferBinding`. All kernels (`norm`, `sigmoid_mul`, `silu_mul`, `binary`, `mul_mat`, `rope`, `attention`) take `BufferView` arguments; offset/length arithmetic moved out of every shader uniform and into one place. Q-extract motivation resolved: `RmsNormInplaceWebgpu` / `RopeInplaceWebgpu` / `CausalGqaNaiveAttentionWebgpu` now read Q directly from the fused `q_gate_proj_buffer` via a strided view; `SliceCopyWebgpu` and its WGSL shader are deleted.
- **File-structure reorganization** — Kernels now live under [src/kernels/](kernels/) (one file per pipeline) with their shaders in [src/kernels/wgsl-shaders/](kernels/wgsl-shaders/). Layer-level orchestration lives under [src/layers/](layers/) (`layer_stack.rs`, `linear_attention.rs`, `qwen35_self_attention.rs`, `minicpm5_self_attention.rs`, `mlp.rs`). The top level keeps model / session / sampler / config.
- **Per-step Runner / bake pattern** *(commit `3b38d3c`)* — Every kernel and layer moved from an encoder-and-submit `forward(device, queue, ...)` convenience to a two-stage `plan(...) -> Runner` API. `Qwen35Session::new` builds the decode runner once via `DecodeRig::build`; `decode_step` is now `queue.write_buffer` + `runner.forward(cpass)` with **zero `create_buffer` calls per token**. This closes the original "ScratchArena" item ("240 buffer creates / token in decode"); the residual prefill-side cost is tracked separately as item 3.

## Dropped (no longer worth doing)

- **ScratchArena with `N`-bucket graph coloring** — The headline motivation was "every `LayerSession::forward` call still does ~10 `device.create_buffer` (×24 layers ⇒ ~240 / token in decode)". The per-step Runner refactor moved all of those allocations to one-time session construction, so the per-token cost is now zero. The remaining VRAM-fragmentation argument doesn't justify a graph-coloring pass; if it ever does, the cheaper per-size-class free-list variant is the right starting point. See [docs/wgpu-single-buffer-arena.md](../docs/wgpu-single-buffer-arena.md) for the post-mortem on why a single arena buffer doesn't work under `wgpu`'s whole-buffer usage tracking.
- **LM head SIMD + pre-allocated logits buffer** — Was conditional on keeping the CPU LM head as a hot path. With the GPU LM head + GPU sampler now landed (see Done), `LmHeadCpu` is a debug-only fallback and its inner-loop throughput stops mattering. Not worth the `half` / `wide` SIMD work after the fact.