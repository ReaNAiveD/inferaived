---
name: wgsl-validation
description: "Validate WGSL shaders using naga-cli. Use when: checking shader correctness, debugging shader compilation errors, validating override constants, translating shaders to SPIR-V/HLSL/MSL/GLSL."
---

# WGSL Shader Validation with naga-cli

## When to Use
- Validate WGSL shader syntax and semantics offline
- Debug `create_shader_module` validation errors from wgpu
- Check shader correctness after edits before running the application
- Translate WGSL to other shader formats (SPIR-V, HLSL, MSL, GLSL)

## Prerequisites

```
cargo install naga-cli
```

## Validation Commands

### Basic validation (no output, just check)

```
naga path/to/shader.wgsl
```

### With override constants

Required when the shader uses `override` declarations. Pass every override used in the shader:

```
naga --override name1=value1 --override name2=value2 path/to/shader.wgsl
```

### Restrict to a single entry point

```
naga --entry-point main path/to/shader.wgsl
```

### Translate to another format (implies validation)

```
naga input.wgsl output.spv      # SPIR-V
naga input.wgsl output.metal    # MSL
naga input.wgsl output.hlsl     # HLSL
naga input.wgsl output.glsl     # GLSL (requires --profile and --shader-stage)
```

### Bulk validation

```
naga --bulk-validate shader1.wgsl shader2.wgsl shader3.wgsl
```

## WGSL Override Rules (naga/wgpu)

These are the key constraints enforced by naga that differ from what the WGSL spec text might suggest:

| Variable scope | Override-sized arrays allowed? |
|---|---|
| `var<workgroup>` | Yes |
| `var<private>` | **No** — requires constructible type |
| Function-local `var` | **No** — requires constructible type |

**Constructible** means the type's size is fully determined at shader parse time (uses only `const`, not `override`).

### Workaround patterns

When you need per-thread arrays sized by tuning parameters:

1. **Use `const` instead of `override`** for values that size per-thread arrays. Pass them to the shader via text substitution or hardcode them. Only use `override` for values that size `var<workgroup>` or are used in `@workgroup_size()`.

2. **Hardcode small register arrays** — if tile sizes are fixed (e.g., `tile_m=4`, `tile_n=4`), use `const tile_m: u32 = 4;` so `array<f32, tile_m * tile_n>` is constructible.

## Bounds Check Policies

```
naga --index-bounds-check-policy Restrict shader.wgsl output.spv
```

| Policy | Behavior |
|---|---|
| `Unchecked` | Default. No bounds checks. |
| `Restrict` | Clamp indices in-bounds. |
| `ReadZeroSkipWrite` | OOB reads return zero, OOB writes are dropped. |

`--buffer-bounds-check-policy` and `--image-load-bounds-check-policy` override the default for storage/uniform buffers and texture loads respectively.
