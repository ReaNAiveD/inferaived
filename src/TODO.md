# Host-side TODO

Optimization directions for the Rust orchestration layer (kernels-side TODOs live in `wgsl-shaders/TODO.md`). Listed in rough priority order.

## ScratchArena for per-forward scratch buffers

- Every `LayerSession::forward` call still does ~10 `device.create_buffer` (×24 layers ⇒ ~240 buffer creates / token in decode). All of these have the same lifetime: from the start of `forward` to its end.
- Direction: a bump arena owned by `LayerStackSession`, sized for the worst-case `num_new_tokens`. Reset at the top of each `forward`. Layers borrow sub-buffers instead of allocating.
- Open question: do we expose sub-buffers as `&wgpu::Buffer` + `(offset, size)` pairs, or as a `BufferView` type (see next item)? The arena's internals likely want to be one big `wgpu::Buffer` with bump-allocated offsets, which composes naturally with `BufferView`.

## `BufferView` for strided / offset access

- Primary motivation: `SelfAttentionLayer` projects a fused `q_gate_combined` tensor laid out as `[q_head_0 | gate_head_0 | q_head_1 | gate_head_1 | ...]`. Today we run `SliceCopyWebgpu` to materialize a contiguous q tensor for the attention kernel — a pure shuffle dispatch that produces no new information.
- Direction: a `BufferView { buffer, offset, byte_stride_per_row, row_count, row_byte_size }` (or similar) that kernels accept in place of `&wgpu::Buffer`. RoPE / attention can then read `q` directly from the interleaved layout via per-head stride, killing the copy entirely.
- Also unblocks: arena sub-buffers, cleaner partial-row dispatches across the codebase.

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

- `src/` is currently flat with ~15 sibling files mixing orchestration (`language_model.rs`, `layer_loop.rs`, `main.rs`) with low-level kernels (`mul_mat.rs`, `norm.rs`, `rope.rs`, `delta_rule.rs`, `conv_silu.rs`, `gated_rms_norm.rs`, `sigmoid_mul.rs`, `slice_copy.rs`, `binary.rs`, `mamba_scan.rs`, …).
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