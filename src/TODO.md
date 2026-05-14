# Host-side TODO

Optimization directions for the Rust orchestration layer (kernels-side TODOs live in `wgsl-shaders/TODO.md`).

## Buffer reuse

- `LinearAttentionLayer::compute` and `SelfAttentionLayer::compute` currently allocate ~9 GPU buffers per call. For multi-token decode this becomes per-step churn.
- Direction: introduce a shared scratch arena owned by `LayerStack` (sized for the worst-case `seq_len`), pass it down to `compute()`. Layers borrow slices instead of allocating.
- Alternative: per-layer cache keyed by `seq_len` (lazily grown on first call).

## LM head SIMD / batch bf16→f32

- Current `LmHeadCpu::compute` does scalar `bf16::to_f32()` per element.
- Direction: use `half::slice::HalfFloatSliceExt::convert_to_f32_slice` + `wide::f32x8` for the inner dot product. Enable the `use-intrinsics` feature on `half`.

## LM head: pre-allocated logits buffer

- `compute()` returns `Vec<f32>` (~1 MB) every call.

## KV cache for full-attention layers

- `SelfAttentionLayer` currently recomputes K/V projections for every prompt token on every step.
- Direction: hold a per-layer KV cache buffer; `compute()` for decode steps only projects the new token and appends.

## Recurrent state cache for linear-attention layers

- `LinearAttentionLayer` already owns `recurrent_state_buffer`, but never persists it across calls (delta_rule reads + overwrites).
- Direction: split into a prefill path (initialize state) and a decode path (advance state by one token); hold state across `compute()` calls.

## Quantization (long term)

- bf16 weights = 1.5 GB total. Int8 sym would halve it; int4 quarters it.
- Direction: per-row symmetric quantization for `MulMatWebgpu` and `LmHeadCpu` weights. Would unlock significantly larger context / faster LM head.

## Load model dims from `config.json`

- `main.rs` hardcodes `hidden_size`, `intermediate_size`, head dims, etc.
- Direction: parse `model/Qwen3.5-0.8B/config.json` via `serde_json` into a typed `Qwen35Config`, drive `LayerStackConfig` from it.

## Replace env-var dump with proper Inspector abstraction

- Per-layer GPU readback for validation is currently driven by `INFERAIVED_DUMP_DIR` (env var, hardcoded path scheme, side-effecting from inside `LayerStack::compute`).
- Direction: design a clean inspector / tap interface (trait-based, zero-cost when unused) so the same hooks can serve dump, profile, KV-cache validation, and per-step logging without env-var coupling.
