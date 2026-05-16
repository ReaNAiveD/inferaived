When encountering uncertainty about the implementation details of any specific LLM inference architecture (e.g., attention mechanisms, KV-cache strategies, speculative decoding, quantization schemes, custom kernel designs, Mamba/SSM recurrences, delta-rule layers, RoPE variants, flash attention, grouped-query attention, mixture-of-experts routing, or any other deep domain knowledge), do not guess or hallucinate. Instead:

1. **Search the web** for authoritative sources — official papers, reference implementations, or well-known technical blog posts — to verify the correct algorithm, data layout, or numerical behavior.
2. **Read the relevant source code** in this repository (or in upstream reference repos such as Hugging Face Transformers, vLLM, llama.cpp, or the original author's code) to confirm how a component is actually wired together.
3. **Consult the original research paper or technical essay** (e.g., on arXiv) when the design rationale or mathematical formulation is unclear.

Always cite or link to the source you relied on so the information can be verified.

## Software Design Principles

These principles synthesize widely-accepted software engineering wisdom (SOLID, Rust API Guidelines, "parse don't validate", "make illegal states unrepresentable") and apply across all code in this project.

- **Single Responsibility.** Each module, type, or function should have one reason to change. When a unit accumulates multiple responsibilities, split it.
- **Make illegal states unrepresentable.** Encode invariants in the type system so the compiler rejects incorrect usage. Prefer distinct types over runtime flags, enums over booleans, and required parameters over optional ones whose meaning depends on context.
- **Parse, don't validate.** Convert untrusted input into a strongly-typed representation at the boundary, then operate on the typed form internally. Do not re-check the same invariant at every use site.
- **High cohesion, low coupling.** Group related data and behavior together. Minimize what each module needs to know about others. Depend on abstractions, not concrete implementations.
- **APIs should be hard to misuse.** Function signatures, type bounds, and ownership rules should make incorrect calls fail to compile. Surface errors as early as possible — preferably at compile time, otherwise at construction, never silently at runtime.
- **Name parameters by semantic role, not by typical-case value.** A matmul's contraction dim is `k`, not `hidden_size` — the latter happens to be correct for q/k/v projections but silently wrong for o_proj or FFN down-projection. Names that encode "where this value usually comes from" lure callers into supplying the wrong value when the typical case doesn't apply.
- **Validate every dimension of weight tensors at construction.** When a `Linear(in, out)` weight is loaded, assert BOTH `shape[0] == out` AND `shape[1] == in`. One-sided assertions catch only half of misconfigurations and let the other half degrade silently into wrong results.
- **Optimize for reading, not writing.** Code is read far more often than written. Choose names that describe intent over implementation. Keep functions short enough to understand at a glance.

## GPU Compute Design Principles

These principles synthesize widely-accepted GPU programming wisdom (NVIDIA CUDA Best Practices Guide, ggml/llama.cpp patterns, vLLM kernel design, grid-stride loops) and apply to all WGSL shaders and `wgpu` host code in this project.

- **Decouple kernels from memory layout via strides and offsets.** Shaders should accept stride/offset parameters in a uniform rather than assume tightly-packed contiguous data. This lets one shader operate on full tensors, sub-slices, or interleaved views without duplication. Production engines (vLLM, llama.cpp) all do this.
- **Match data layout to access pattern.** GPU performance is dominated by memory bandwidth. Adjacent threads should read adjacent memory. When the natural layout fights this, restructure the data at load time rather than fight it at every kernel invocation.
- **Minimize host↔device transfers.** Each transfer has fixed overhead. Keep intermediate tensors on the GPU; only transfer final results back. Batch small operations into larger dispatches when possible.
- **Validate intermediate outputs against a reference.** Floating-point parallelism produces non-bitwise-identical but numerically equivalent results. Compare against a CPU/PyTorch reference within a small epsilon, not bit-for-bit equality.
- **Profile before optimizing.** Don't add complexity (fused kernels, custom layouts, in-place variants) for performance reasons without measurement. Most overhead is hidden behind dispatch latency or memory bandwidth, not arithmetic.
- **Prefer many small, single-purpose kernels over one large parameterized kernel.** Each kernel should do one well-defined operation. Composability and debuggability outweigh dispatch overhead at modest scales.
- **One pipeline per shader; group pipelines by shared resource.** Each shader is its own pipeline. A host struct may own several pipelines when they share a large GPU resource; otherwise keep them in separate structs.
