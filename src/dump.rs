//! Debug-only GPU buffer dump helpers.
//!
//! Triggered by the `INFERAIVED_DUMP_DIR` environment variable. When set,
//! call sites inside `Qwen35Model::compute` / `LayerStack::compute` will
//! `copy_buffer_to_buffer` the relevant GPU buffer into a CPU-mapped
//! staging buffer, read it back as `&[f32]`, and write it as a single-tensor
//! `.safetensors` file under that directory.
//!
//! This is intentionally side-effecting and uses raw env vars — the proper
//! Inspector abstraction is tracked in `src/TODO.md`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use safetensors::{Dtype, serialize_to_file, tensor::TensorView};

/// Returns the directory specified by `INFERAIVED_DUMP_DIR`, if any.
pub fn dump_dir() -> Option<PathBuf> {
    std::env::var_os("INFERAIVED_DUMP_DIR").map(PathBuf::from)
}

/// Optional layer-index filter. When `INFERAIVED_DUMP_LAYER` is set to an
/// integer (e.g. `3`), only that layer's fine-grained dumps fire; otherwise
/// every layer dumps when `INFERAIVED_DUMP_DIR` is set.
pub fn dump_layer_filter() -> Option<usize> {
    std::env::var("INFERAIVED_DUMP_LAYER")
        .ok()
        .and_then(|s| s.parse().ok())
}

/// Returns the dump directory if dumping is enabled and the given `layer_index`
/// matches the optional filter.
pub fn layer_dump_dir(layer_index: usize) -> Option<PathBuf> {
    let dir = dump_dir()?;
    match dump_layer_filter() {
        Some(filter) if filter != layer_index => None,
        _ => Some(dir),
    }
}

/// Copy `num_floats` f32s out of `src` (starting at `src_offset` bytes) and
/// write them to `out_path` as a safetensors file with a single `"data"`
/// tensor of shape `shape`.
pub async fn dump_buffer_as_safetensors(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    src_offset: u64,
    num_floats: usize,
    shape: Vec<usize>,
    out_path: &Path,
) {
    debug_assert_eq!(
        shape.iter().product::<usize>(),
        num_floats,
        "shape product must equal num_floats"
    );
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent).expect("Failed to create dump directory");
    }

    let byte_size = (num_floats * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dump/staging"),
        size: byte_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("dump/encoder"),
    });
    encoder.copy_buffer_to_buffer(src, src_offset, &staging, 0, byte_size);
    let submission = queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = tokio::sync::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: Some(submission),
        timeout: None,
    });
    rx.await
        .expect("dump map_async channel dropped")
        .expect("dump map_async failed");

    let bytes: Vec<u8> = slice.get_mapped_range().to_vec();
    staging.unmap();

    let view = TensorView::new(Dtype::F32, shape, &bytes).expect("Failed to build TensorView");
    let mut tensors: HashMap<String, TensorView> = HashMap::new();
    tensors.insert("data".to_string(), view);
    serialize_to_file(&tensors, None, out_path).expect("Failed to write safetensors file");
}
