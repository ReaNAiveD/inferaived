# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-07

Initial release.

### Added

- GPU-resident inference engine on top of `wgpu` 29 (Vulkan / Metal / DX12 / WebGPU).
- Support for Qwen3.5 and MiniCPM5 model families, loaded from Hugging Face `safetensors`.
- Custom WGSL kernels: matmul, RoPE, RMSNorm, masked attention, mamba scan, delta rule, samplers.
- GPU KV cache, continuous decode loop, and argmax sampler.
- Runnable examples: `generate`, `chat_qwen35`, `chat_minicpm5`, `parallel_minicpm5`, `bench_decode`.
