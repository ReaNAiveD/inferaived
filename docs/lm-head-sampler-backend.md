# End-to-end pipeline: embed lookup + LM head + sampler

Status: **implemented** — migration steps 1–6 and 8 landed; step 7 (GPU temperature / top-k / top-p / min-p + GPU PRNG) deferred.

Follow-up performance work tracked in [`docs/gpu-continuous-decode.md`](gpu-continuous-decode.md), which removes the remaining per-step CPU↔GPU round-trips on the decode path.

## Outcome

- 54 lib tests pass; greedy tokens bit-identical between CPU- and GPU-backed sessions (`[1049, 369, 5995, 310, 381]` → " It is designed to be").
- `bench_decode` (greedy decode, max_seq_len = 256):

  | metric        | before (CPU sampler) | after (GPU end-to-end) |
  | ------------- | -------------------: | ---------------------: |
  | TTFT          | 324 ms               | 197 ms                 |
  | Decode tok/s  | 8.26                 | 30.73                  |

  Recorded in [`benches/baselines.local.csv`](../benches/baselines.local.csv) under `gpu-end-to-end-pipeline`.
- Zero runtime `panic!` / `unreachable!` from backend mismatch — every cross-product the original `enum Backend` would have allowed is now a compile error (see § "Backend choice via type, not value").

## Problem

Today the decode loop straddles CPU and GPU on **both ends** of the layer stack: a `hidden_size` f32 push on the way in, and a `vocab_size` f32 pull on the way out (≈ 970 kB / step for our model). Only argmax is supported on the CPU side.

Two coupled changes we want:

1. **End-to-end GPU pipeline** for the production decode path: embed lookup, LM head, sampler all on-device, with one shared `current_token: u32` buffer the sampler writes and the next step's embed reads. Critical-path readback drops to one `u32` per step in streaming mode, or zero in batch / non-streaming mode.
2. **A real sampler API** that supports the standard tricks (temperature, top-k, top-p, min-p, repetition penalty), with custom logit processors and samplers on the CPU path, and a fixed built-in pipeline on the GPU path.

## Survey: how modern engines decompose this

Every modern inference engine splits the sampling pipeline along three axes: logit transforms, terminal samplers, and stopping criteria. The shapes:

| Engine                                 | Logit transforms                                              | Terminal sampler                                      | Stopping                                | GPU pluggable? |
| -------------------------------------- | ------------------------------------------------------------- | ----------------------------------------------------- | --------------------------------------- | -------------- |
| HF Transformers `generation/`          | `LogitsProcessor` chain (`LogitsProcessorList`)               | `do_sample` + `GenerationConfig` (argmax / multinom / beam) | `StoppingCriteria` chain          | n/a (eager)    |
| vLLM `v1/sample/`                      | Built-in GPU ops + `logits_processor/` plugin interface       | Fused `topk_topp_sampler` GPU kernel + `RejectionSampler` for spec decode | `StoppingCriteria` (server-side)    | partial (custom processors fall back to CPU) |
| llama.cpp `llama_sampler` chain        | Chain of samplers (penalties, dry, logit_bias, grammar, …)    | Terminal element of chain (`greedy` / `dist` / `mirostat` / `adaptive_p`) | EOG token + caller loop      | **yes** (new `backend_apply` interface) |
| Candle `LogitsProcessor`               | Built into the struct via `Sampling` enum + `sample_f` hook   | Same struct (`ArgMax` / `All` / `TopK` / `TopP` / `GumbelSoftmax`) | caller loop                | n/a            |

Three things to take from this:

- **HF and vLLM both have a `LogitsProcessor` *chain* and a separate terminal sampler.** Everyone with any non-trivial sampling support splits the two. HF and llama.cpp expose them as a chain of pluggable units; Candle bundles them into one struct (and provides a `sample_f` escape hatch). vLLM does both: built-in fused GPU ops for the common case, plus a CPU-side `LogitsProcessor` plugin point for custom logic.
- **llama.cpp's `llama_sampler_i::backend_apply`** is the cleanest way to make one sampler implementation cover both CPU and GPU, but adds complexity that is not justified for a single-stream CPU+GPU engine.
- **Stopping criteria, streaming, and the KV cache** are *not* part of the sampler in any of these engines. Three orthogonal axes.

Not designed here, but the survey surfaced them: speculative decoding (`RejectionSampler`), structured / guided decoding (grammars, xgrammar, outlines), beam search, detokenization streaming, watermarking. All slot in on top of the design below — see "Future work".

## Pluggability boundary: do any of them compile user logic to GPU?

Short answer: **almost no engine compiles user-supplied `LogitsProcessor`s into a GPU kernel.** The pattern is universally "fixed library of built-in GPU kernels driven by `SamplingParams` + CPU-side plugin point for custom logic". The exceptions are narrow:

| Engine             | User processor on GPU?                                                                 |
| ------------------ | -------------------------------------------------------------------------------------- |
| vLLM               | No. Built-in Triton kernels for common knobs; custom processors are Python callables that run eager on the device the tensor lives on (each its own kernel launch). |
| HF Transformers    | No. `LogitsProcessor.__call__` is eager Python; `CompileConfig` only `torch.compile`s the model forward, not the generation loop. |
| llama.cpp          | Partial. The new `llama_sampler_i::backend_apply` lets a sampler **emit a GGML graph by hand** — built-ins (top-k/top-p/temp) have been ported. Not "compile arbitrary user logic", just a GPU-primitive DSL. |
| TensorRT-LLM / TGI | No. Fixed kernel chain. Extending = CUDA plugin.                                       |
| Candle             | No. CPU-only `sample_f` callback on already-readback probs.                            |
| MLX-LM / JAX       | Opportunistic. If the user writes the processor in MLX/JAX ops and the loop is under `mx.compile` / `jax.jit`, the framework's tracer fuses it. Property of the framework, not the engine.                                                |

Five reasons this is the universal answer:

1. **The sampler isn't the bottleneck.** One vocab-sized read + O(vocab log k) compute is bandwidth-bound. Fusing a custom transform in saves microseconds of launch overhead. The big GPU sampler win was keeping logits on the device, which the built-ins already deliver.
2. **The DSL barrier defeats the point.** "Pluggable" means a 20-line Python / Rust function. Forcing the user into WGSL/CUDA/Triton turns the plugin point into a contribution-to-the-engine workflow, which every engine already supports via "submit a PR with a new built-in".
3. **Many useful processors are data-structure heavy, not arithmetic-heavy.** Grammars, no-repeat-ngrams, watermarking with hashed contexts, and structured-output masks (xgrammar, outlines, lm-format-enforcer) **construct token-level allow/deny masks on CPU** using non-trivial data structures, then upload the mask and let a built-in GPU op apply it. The construction step doesn't compile cleanly to a kernel; the apply step is one fused multiply everyone already has.
4. **Composition pays for itself.** N pre-fused built-in kernels chained gives ~95% of the win for ~1% of the engineering. vLLM is the proof — a small library of single-purpose Triton kernels mixed and matched by `SamplingParams`.
5. **Validation cost.** Arbitrary GPU code is hard to validate against a CPU reference. Engines prefer a small vetted built-in set + an unvetted CPU-side plugin point.

**Implication for us.** wgpu / WGSL has *no* runtime kernel JIT analogous to `torch.compile` / `mx.compile` / Triton — shaders are static text compiled at build time. The GPU side is fixed by necessity. The CPU side is fixed by choice in v1 — closed enums of built-in processors and samplers, matching the project's existing `LayerConfig::{Linear, Full}` style. Extension points are listed under "Future work".

## Component split

The three axes the survey identified map onto three types in `src/sampling.rs`:

- **`LogitsProcessor`** — pure logit transform. Input: history of generated tokens and the live logits vector. Effect: in-place mutate the logits. Composable into a `Vec<LogitsProcessor>` chain. Examples: temperature, top-k mask, top-p mask, min-p mask, repetition penalty, logit bias.
- **`Sampler`** — terminal token selector. Input: post-processed logits. Output: a `SampledToken { id, logprob }`. Examples: greedy (argmax), multinomial draw with PRNG.
- **`StoppingCriteria`** — orthogonal session-level stop condition. Examples: EOS token match, max tokens reached.

The same `SamplingParams` struct — a flat bag of numeric knobs — drives both the GPU kernel (read directly at dispatch time) and the CPU path (translated to a concrete processor chain + sampler via `default_processors` / `default_sampler`).

## Recommended design

### Components

```rust
// src/sampling.rs (new)

/// All numeric knobs that the standard logit transforms + terminal
/// samplers consume. Both the CPU and GPU sessions take one of these.
///
/// `Default` is pure greedy (`temperature = 0`, every other knob at its
/// disabled value).
#[derive(Clone, Copy, Debug)]
pub struct SamplingParams {
    pub temperature: f32,        // <= 0 ⇒ argmax (no random draw)
    pub top_k: usize,            // 0 ⇒ disabled
    pub top_p: f32,              // 1.0 ⇒ disabled
    pub min_p: f32,              // 0.0 ⇒ disabled
    pub repetition_penalty: f32, // 1.0 ⇒ disabled
    pub presence_penalty: f32,   // 0.0 ⇒ disabled
    pub frequency_penalty: f32,  // 0.0 ⇒ disabled
    pub rng_seed: u64,
}

impl Default for SamplingParams { /* greedy: temperature = 0, rest disabled */ }

#[derive(Clone, Copy, Debug)]
pub struct SampledToken {
    pub id: u32,
    pub logprob: f32,            // post-softmax log-probability of `id`
}

/// Pure logit transform. CPU only. Composable into a chain. Closed
/// enum; extension points are listed under "Future work". The
/// signature mirrors HF: history of generated ids (for penalty
/// processors), live logits mutated in place.
pub enum LogitsProcessor {
    Temperature(f32),
    TopK(usize),
    TopP(f32),
    MinP(f32),
    RepetitionPenalty(f32),
    FreqPresencePenalty { freq: f32, presence: f32 },
    LogitBias(Vec<(u32, f32)>),
}

impl LogitsProcessor {
    /// Processors are pure transforms — no per-step mutable state — so
    /// `&self` is enough. Anything stateful (e.g. cached n-grams) goes
    /// in the `Custom` trapdoor described under "Future work".
    pub fn process(&self, generated: &[u32], logits: &mut [f32]) { /* match self */ }
}

/// Terminal token selector. CPU only. Owns its PRNG state where needed.
pub enum Sampler {
    Greedy,                       // argmax
    Multinomial { rng: SmallRng } // softmax + draw
}

impl Sampler {
    /// `logits` is the post-processor vector. Result includes the
    /// post-softmax logprob of the chosen token.
    pub fn sample(&mut self, generated: &[u32], logits: &[f32]) -> SampledToken { /* match */ }
}

/// Session-level orthogonal generation-stopping rule.
///
/// `MaxTokens` is **not** a variant: the per-call `max_tokens: usize`
/// argument on `generate(...)` is the explicit loop bound. Passing an
/// empty `stopping` slice with a large `max_tokens` is the supported
/// "generate exactly N tokens" idiom.
pub enum StoppingCriteria {
    Eos(Vec<u32>),
    // StopStrings { ... } — needs tokenizer to detokenize tail; defer.
}

impl StoppingCriteria {
    pub fn is_done(&self, generated: &[u32], last: SampledToken) -> bool { /* match */ }
}

/// Build the standard processor chain / sampler from a SamplingParams
/// value, used by `DecodingPolicy::cpu_default`. Exposed separately so
/// callers can construct their own variant on top of the defaults.
///
/// Standard chain order mirrors HF / vLLM: penalties first, then
/// truncation (top-k, top-p, min-p), then temperature. Empty vec when
/// every knob is at its disabled value.
pub fn default_processors(p: &SamplingParams) -> Vec<LogitsProcessor>;

/// `Sampler::Greedy` when `temperature <= 0`, else
/// `Sampler::Multinomial { rng }` seeded from `params.rng_seed`.
pub fn default_sampler   (p: &SamplingParams) -> Sampler;
```

### Backend choice via type, not value

With four parallel `Cpu` / `Gpu` enums (head, tail, session state, policy), only 2 of the 16 representable cross-products are valid; the other 14 are illegal states. Each one of them costs a runtime panic. We pay the small cost of two concrete types per role to eliminate all of them.

Shared mid-stack lives in `Qwen35ModelCore`; backend-specific embed / sampler / fields live on the two concrete model types. The GPU side carries **no `'data` parameter**: the embed table's bf16 bytes are uploaded to a device buffer at construction time and never referenced on the host again. The CPU side legitimately keeps `'data` because `EmbeddingLookupCpu<'data>` holds the `TensorView<'data>` for runtime row gather.

```rust
// src/language_model/{mod,core,cpu,gpu}.rs

/// Pieces both backends share. The GPU LM-head mat-mul is impractical
/// on CPU (vocab × hidden = 248K × 1024 bf16 muls per step), so it
/// belongs to the shared core. Lifetime-free — every referenced weight
/// is uploaded to a device buffer in `new`.
pub struct Qwen35ModelCore {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub layer_stack: LayerStack,
    pub final_norm: RmsNormInplaceWebgpu,
    pub lm_head: LmHeadWebgpu,
}

impl Qwen35ModelCore {
    /// `embed_tokens` is taken as an explicit parameter (not re-fetched
    /// from `tensors`) so the caller can also pass the same view to its
    /// backend-specific embed lookup without a duplicate borrow of
    /// `SafeTensors`. The view is consumed: its bf16 bytes are uploaded
    /// to the lm_head's weight buffer and never read again.
    pub(super) fn new<'data>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tensors: &SafeTensors<'data>,
        config: &Qwen35Config,
        embed_tokens: TensorView<'data>,
    ) -> Self;
}

/// CPU-backend model: CPU embed, GPU mid-stack, CPU sampler /
/// processor chain (per-request).
pub struct Qwen35CpuModel<'data> {
    pub core: Qwen35ModelCore,
    pub embed: EmbeddingLookupCpu<'data>,
}

/// GPU-backend model: GPU embed (thin wrapper over the generic
/// `kernels::get_rows::GetRows` kernel), GPU mid-stack, GPU sampler
/// kernel. No `'data` parameter — nothing here references host bytes.
pub struct Qwen35GpuModel {
    pub core: Qwen35ModelCore,
    pub embed: EmbeddingLookupWebgpu,
    pub sampler: GpuSampler,
}

impl<'data> Qwen35CpuModel<'data> { pub fn new(...) -> Self { /* */ } }
impl         Qwen35GpuModel        { pub fn new<'data>(...) -> Self { /* */ } }
```

Sessions follow the same split — each session type borrows the matching model type:

```rust
pub struct Qwen35CpuSession<'m, 'data> {
    model: &'m Qwen35CpuModel<'data>,
    /* CPU-mode buffers + runners */
}
pub struct Qwen35GpuSession<'m> {
    model: &'m Qwen35GpuModel,
    /* GPU-mode buffers + runners, including the persistent
       `current_token: wgpu::Buffer` */
}
```

No more `Backend` value-enum, no more `Qwen35Head`, no more `Qwen35Tail`, no more `SessionMode`, no more `DecodingPolicy`. The backend choice IS the type you construct. Mismatched-backend states are unrepresentable; every panic that used to express the invariant becomes a compile error.

A mild trade-off: code that wants to be backend-agnostic at runtime (e.g. a CLI that flips on a flag) has to wrap the two session types in a local `enum` or trait object. None of our code does today.

### Per-step data flow

The head + tail are selected as a unit (at the type level via the concrete model type); mixed CPU/GPU combinations are rejected because they reintroduce per-step round-trips.

A single `current_token: wgpu::Buffer` of size `1 × u32` lives on `Qwen35GpuSession` for the entire conversation: the sampler kernel writes it; the next step's decode embed-lookup reads it. Read-after-write across dispatches in one encoder is fine under wgpu's per-buffer usage tracker.

```text
                                  ─── decode step (Gpu backend) ───
        ┌── current_token (1 × u32 GPU buffer, persistent) ──┐
        │                                                    │
        ▼ read                                          write ▲
embed_lookup → layers → final_norm → lm_head → sampler kernel
                                                       │
                                                       ▼ (streaming only)
                                          readback (1 × u32 = 4 B)
```

The CPU backend keeps today's shape — `hidden_size` f32 push to GPU, `vocab_size` f32 pull, then the processor chain + sampler runs in Rust:

```text
                                  ─── decode step (Cpu backend) ───
generated: Vec<u32>      ─────┐
                              │
                   ┌──────────▼──────────┐    write_buffer (hidden_size × f32)
token_id → EmbeddingLookupCpu ─→ hidden ─┴────────────────────────────────►
                                                                          │
                              ┌───────────────────────────────────────────┘
                              ▼ (layers → final_norm → LM head on GPU)
                              logits readback (vocab_size × f32) → CPU
                              │
                              ▼
            LogitsProcessor chain ── (processes logits in-place)
                              │
                              ▼
            Sampler ─────────────► SampledToken { id, logprob }
```

### Per-step API

Two `step` methods, one per session type. Signatures are deliberately asymmetric because the underlying state is asymmetric: CPU sampling carries mutable RNG state ([`Sampler::Multinomial`]); GPU sampling is stateless from the host's perspective (kernel handles its own state).

```rust
impl<'m, 'data> Qwen35CpuSession<'m, 'data> {
    pub async fn step(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_ids: &[u32],
        processors: &[LogitsProcessor],
        sampler: &mut Sampler,
    ) -> SampledToken;

    pub async fn generate(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_ids: &[u32],
        processors: &[LogitsProcessor],
        sampler: &mut Sampler,
        max_tokens: usize,
        stopping: &[StoppingCriteria],
    ) -> Vec<SampledToken>;
}

impl<'m> Qwen35GpuSession<'m> {
    pub async fn step(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_ids: &[u32],
        params: &SamplingParams,
    ) -> SampledToken;

    pub async fn generate(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_ids: &[u32],
        params: &SamplingParams,
        max_tokens: usize,
        stopping: &[StoppingCriteria],
    ) -> Vec<SampledToken>;
}
```

The GPU `step` returns `SampledToken { id, logprob: f32::NAN }` today: the greedy GPU kernel doesn't compute logprobs. Consumers that need a real logprob use `Qwen35CpuSession`; once step 7 lands, the GPU kernel will fill `logprob` directly.

User code, GPU backend (just the knobs):

```rust
let model = Qwen35GpuModel::new(&device, &queue, &tensors, &config);
let mut session = Qwen35GpuSession::new(&model, &device, &queue, max_seq_len);
let params = SamplingParams { temperature: 0.7, top_k: 40, top_p: 0.9, ..Default::default() };
let tok = session.step(&device, &queue, &[next_input], &params).await;
```

User code, CPU backend with the standard chain derived from params:

```rust
let model = Qwen35CpuModel::new(&device, &queue, &tensors, &config);
let mut session = Qwen35CpuSession::new(&model, &device, &queue, max_seq_len);
let params = SamplingParams { temperature: 0.7, top_k: 40, top_p: 0.9, ..Default::default() };
let processors = default_processors(&params);
let mut sampler = default_sampler(&params);
let tok = session.step(&device, &queue, &[next_input], &processors, &mut sampler).await;
```

User code, CPU backend with a hand-rolled chain and an explicit sampler:

```rust
let processors = vec![
    LogitsProcessor::RepetitionPenalty(1.1),
    LogitsProcessor::TopK(40),
    LogitsProcessor::Temperature(0.7),
];
let mut sampler = Sampler::Greedy;
let tok = session.step(&device, &queue, &[next_input], &processors, &mut sampler).await;
```

### What we are deliberately *not* doing

- **Generic `Qwen35Session<B: Backend>` over a phantom backend type.** We use two concrete session / model types instead. Generics over a phantom can't change field types per `B` without GATs, so internal `match` arms on `Qwen35Head` / `Qwen35Tail` still need `unreachable!` branches — defeating the type-level invariant we wanted. Two concrete types pay a little duplication (factored into `Qwen35ModelCore`) and fully eliminate the panics.
- **A runtime `Backend` value-enum or wrapper `enum Qwen35Session { Cpu(...), Gpu(...) }`.** Reintroduces the runtime check we just removed. Add only if a caller actually needs runtime dispatch (none does today).
- **User-defined `LogitsProcessor` / `Sampler` / `StoppingCriteria`.** All three are closed enums; the project's pattern is static dispatch first, dyn dispatch only when a concrete extension point appears. Opening any one of them up later is a 30-line change (add `Custom(Box<dyn Trait>)` variant, define the trait next to the enum, existing built-in variants and call sites untouched).
- **Backend-pluggable samplers** (llama.cpp's `backend_apply` trick). Only worth doing if a third backend lands.
- **Built-in GPU custom-processor fallback.** vLLM does support this — custom CPU processors trigger a per-step readback even when the rest runs on GPU. Additive when needed as a separate `step_with_cpu_override` method on `Qwen35GpuSession` that does the readback + CPU chain.
- **Beam search / multi-sequence batching.** Single-stream session by design. The shared `current_token` buffer generalizes to `current_tokens: u32[batch]` without API-surface changes.

## Migration plan

Status legend: ✅ landed · ⏳ deferred · 🧭 follow-up doc.

1. ✅ **Land sampling primitives** \([src/sampling.rs](../src/sampling.rs)\). `SamplingParams`, `SampledToken`, the closed enums `LogitsProcessor`, `Sampler`, `StoppingCriteria` plus their built-in variants and `default_processors` / `default_sampler` helpers. 16 unit tests.
2. ✅ **Refit the CPU path.** `Qwen35CpuSession::step` runs the per-call processor chain + sampler after a vocab-f32 readback. Default (greedy, empty processor chain) is bit-identical to the pre-change hardwired argmax.
3. ✅ **CPU model + session as concrete types.** `Qwen35ModelCore` (shared GPU mid-stack) and `Qwen35CpuModel<'data>` / `Qwen35CpuSession<'m, 'data>` wrap it with the CPU embed. No backend enum, no `Qwen35Head` / `Qwen35Tail` enums — the backend choice IS the type you construct.
4. ✅ **GPU embed lookup, Runner-shape.** Generic kernel lives in [`src/kernels/get_rows.rs`](../src/kernels/get_rows.rs) as `GetRows` / `GetRowsRunner`. `EmbeddingLookupWebgpu` is a thin wrapper that validates the per-call `dst.shape[1] == hidden_size` on `plan(...)`. Two GPU tests in `kernels::get_rows::tests`.
5. ✅ **GPU greedy sampler kernel** \([src/kernels/argmax.rs](../src/kernels/argmax.rs)\). Single-workgroup tree reduction (WG_SIZE = 256) with a grid-stride loop; ties broken on smallest index. Wrapped as `GpuSampler` in [`src/gpu_sampler.rs`](../src/gpu_sampler.rs). 4 unit tests.
6. ✅ **GPU model + session as concrete types.** `Qwen35GpuModel` (no `'data` parameter — see § "Backend choice via type, not value") and `Qwen35GpuSession<'m>` with the persistent `current_token` buffer. The decode runner chains embed → layers → final_norm → lm_head → sampler into one compute pass; only the 4-byte token id is read back per step. `main.rs` and `examples/bench_decode.rs` flipped over.
7. ⏳ **Extend GPU sampler.** Temperature, top-k, top-p, min-p stages (mirroring vLLM's `topk_topp_sampler` op) plus a small GPU PRNG. All consume the same `SamplingParams`. Deferred until the per-step round-trip cleanup in [`docs/gpu-continuous-decode.md`](gpu-continuous-decode.md) lands — the latter doubles the effective tok/s budget and makes the relative cost of a fused sampler easier to measure.
8. ✅ **`generate(...)` convenience + `StoppingCriteria`.** Both sessions ship a `generate` method that loops `step`, applies the stopping slice on each iteration, and returns every sampled token (including the one that tripped EOS). `max_tokens` is an explicit loop bound rather than a `StoppingCriteria` variant.
9. 🧭 **Deferred-readback / continuous decode.** Tracked in [`docs/gpu-continuous-decode.md`](gpu-continuous-decode.md): eliminate the per-step `write_buffer(current_token, …)` and replace blocking `poll(Wait)` with a small async readback ring + lazy EOS draining (vLLM-style, ≤ 1 over-shoot token past EOS). Estimated 1.3–1.6× additional decode-rate speedup.
10. 🧭 **Custom-CPU-processor fallback on the GPU session** \(vLLM-style `step_with_cpu_override`\). Pure addition; surface listed under "Future work".

## Future work this design unlocks

- **User-extensible CPU processors / samplers.** When a real use case for custom CPU logic appears (grammars, watermarking, custom bias), open the enums up. Two ways, both local:
  - **Trapdoor variant.** Add `LogitsProcessor::Custom(Box<dyn LogitsProcessorFn>)` and define `trait LogitsProcessorFn { fn process(&mut self, &[u32], &mut [f32]); }`. Existing built-in variants and call sites stay; one new `match` arm dispatches to the boxed closure. Same recipe for `Sampler`. Zero churn on the GPU path. This is the recommended path.
  - **Full trait flip.** Replace the enums with traits and add the built-ins as separate structs. More invasive but matches HF / llama.cpp shape exactly. Only worth it if user-defined variants significantly outnumber built-ins (unlikely).
- **Custom CPU processors on the GPU backend** (vLLM style): once the trapdoor variant above exists, an additional `DecodingPolicy::GpuWithCpuOverride { params, extra_processors }` variant adds a one-shot full-vocab readback after the GPU `lm_head`, runs the CPU processor chain, then either picks on CPU or writes back to `current_token` for next-step consistency. Pure additive; no surface change to existing variants.
- **Speculative decoding.** Verifier kernel needs the sampler to expose a proper distribution, not just argmax — the `logprob` field on `SampledToken` is already the first step. Logits-stay-on-GPU is the enabling constraint, satisfied by the Gpu backend.
- **Structured / guided decoding (grammars, JSON schemas).** Two paths: add a `LogitsProcessor::Grammar(...)` variant to the enum (built-in, same as the rest), or land the trapdoor variant first and ship the grammar processor as a user-defined `LogitsProcessorFn`. The GPU backend would additionally need either a CPU-fallback hook (above) or a custom mask-apply kernel.
- **Backend-agnostic samplers** (llama.cpp's `backend_apply`). If we ever want one sampler impl that runs on both CPU and GPU, this needs a trait, not an enum — flip `Sampler` to a trait at that point. Not worth doing before there's a concrete use case.
- **Multi-sequence batching / paged KV / scheduler.** Out of scope here — vLLM-style. None of these need to touch the sampler API surface.
