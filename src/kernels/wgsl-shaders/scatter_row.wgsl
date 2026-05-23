// Copy `num_rows` contiguous rows from a source buffer into a multi-row
// destination buffer starting at row `position_offset`.

struct Params {
    row_width: u32,
    num_rows: u32,
}

@group(0) @binding(0)
var<storage, read> src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

@group(0) @binding(3)
var<uniform> position_offset: u32;

override workgroup_size: u32;

@compute @workgroup_size(workgroup_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = params.row_width * params.num_rows;
    if (gid.x >= total) {
        return;
    }
    let row = gid.x / params.row_width;
    let col = gid.x % params.row_width;
    let src_idx = row * params.row_width + col;
    let dst_idx = (position_offset + row) * params.row_width + col;
    dst[dst_idx] = src[src_idx];
}
