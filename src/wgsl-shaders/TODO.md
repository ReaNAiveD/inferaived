# WGSL Shaders TODO

## Switch to multi-dimensional dispatch

**Affected shaders**: All shaders currently using 1D `gid.x` as the sole thread index (e.g. `norm_scale.wgsl`, future shaders).

**Problem**: When total elements exceed `u32::MAX` (~4.3B), single-dimension indexing overflows. This can happen with 1M+ context lengths on large hidden sizes (e.g. 1M × 12288 = 12.9B elements).

**Solution**: Use 2D/3D dispatch to map different tensor dimensions to different `gid` axes:
- `gid.x` → column (ne0 / hidden_size)
- `gid.y` → row (token index)
- `gid.z` → batch or higher dimensions if needed
- `dispatch_workgroups(ceil(ne0 / wg_size), n_tokens, 1)`

This also eliminates expensive 4D index decoding arithmetic (repeated division/modulo chains).
