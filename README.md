# inferaived

An LLM inference engine written in Rust on top of [wgpu](https://github.com/gfx-rs/wgpu).

`inferaived` runs small-to-mid-size transformer language models directly on the GPU through the WebGPU stack, with custom WGSL kernels for the hot paths (matmul, RoPE, RMSNorm, attention, sampling). It targets developers who want to embed local inference into Rust applications without pulling in CUDA, PyTorch, or a separate runtime — only what `wgpu` itself supports.

> Status: **early / experimental.** APIs will change. Expect rough edges.

## Supported models

- **Qwen3.5** (`Qwen3.5-0.8B` checkpoints)
- **MiniCPM5** (`MiniCPM5-1B` checkpoints, including the parallel hybrid layer stack)

Weights are loaded directly from Hugging Face `safetensors` files.

## Features

- Pure-Rust orchestration; GPU work runs through `wgpu` (Vulkan / Metal / DX12 / WebGPU).
- Custom WGSL kernels with a small dispatch / buffer-arena layer.
- GPU-resident KV cache and sampler (argmax today).
- Continuous decode loop with example streaming chat front-ends.

## Examples

The crate ships several runnable examples under [`examples/`](examples/):

| Example | What it does |
| --- | --- |
| `generate` | Minimal single-prompt generation. |
| `chat_qwen35` | Streaming chat with Qwen3.5. |
| `chat_minicpm5` | Streaming chat with MiniCPM5. |
| `parallel_minicpm5` | Parallel-window decoding for MiniCPM5. |
| `bench_decode` | Decode-throughput micro-benchmark. |

Place model weights under `model/<ModelName>/` (matching the layout in each example) and run, e.g.:

```sh
cargo run --release --example chat_qwen35
```

## Requirements

- Rust 2024 edition (`rustc` 1.85+)
- A GPU + driver that `wgpu` 29 supports (Vulkan, Metal, DX12, or browser WebGPU)

## License

Licensed under the [MIT License](LICENSE).
