When encountering uncertainty about the implementation details of any specific LLM inference architecture (e.g., attention mechanisms, KV-cache strategies, speculative decoding, quantization schemes, custom kernel designs, Mamba/SSM recurrences, delta-rule layers, RoPE variants, flash attention, grouped-query attention, mixture-of-experts routing, or any other deep domain knowledge), do not guess or hallucinate. Instead:

1. **Search the web** for authoritative sources — official papers, reference implementations, or well-known technical blog posts — to verify the correct algorithm, data layout, or numerical behavior.
2. **Read the relevant source code** in this repository (or in upstream reference repos such as Hugging Face Transformers, vLLM, llama.cpp, or the original author's code) to confirm how a component is actually wired together.
3. **Consult the original research paper or technical essay** (e.g., on arXiv) when the design rationale or mathematical formulation is unclear.

Always cite or link to the source you relied on so the information can be verified.
