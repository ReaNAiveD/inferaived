/// Shared GPU test utilities for creating device/queue and transferring data
/// to/from GPU storage buffers. Gated behind `#[cfg(test)]` so it is only
/// compiled during `cargo test`.

/// Create a wgpu `Device` and `Queue` suitable for compute-only testing.
/// Returns `None` when no compatible adapter is available (e.g. headless CI
/// without a GPU), so that tests can be skipped gracefully.
pub async fn create_device_queue() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = match instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
    {
        Ok(a) => a,
        Err(e) => {
            eprintln!("GPU adapter request failed: {e}");
            return None;
        }
    };
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("test_device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        })
        .await
        .expect("Failed to create wgpu device");
    Some((device, queue))
}

/// Convenience macro that skips the test if no GPU is available.
macro_rules! gpu_or_skip {
    () => {
        match $crate::gpu_test_utils::create_device_queue().await {
            Some(dq) => dq,
            None => {
                eprintln!("No GPU adapter found — skipping test");
                return;
            }
        }
    };
}
pub(crate) use gpu_or_skip;

/// Upload an `f32` slice to a new GPU storage buffer.
pub fn upload_f32(device: &wgpu::Device, data: &[f32]) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("test/upload_f32"),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

/// Create a zero-initialised GPU storage buffer of the given f32 element count.
pub fn create_f32_buffer(device: &wgpu::Device, num_elements: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test/create_f32_buffer"),
        size: (num_elements * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Read back `num_elements` f32 values from a GPU buffer.
pub fn download_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    num_elements: usize,
) -> Vec<f32> {
    let byte_size = (num_elements * std::mem::size_of::<f32>()) as u64;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test/readback"),
        size: byte_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test/readback_encoder"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &readback, 0, byte_size);
    let sub_idx = queue.submit(Some(encoder.finish()));
    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: Some(sub_idx),
        timeout: None,
    });
    rx.recv().unwrap().unwrap();
    let data: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
    readback.unmap();
    data
}

pub fn run_blocking_compute(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    record: impl FnOnce(&mut wgpu::ComputePass<'_>),
) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test/encoder"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("test/compute_pass"),
            timestamp_writes: None,
        });
        record(&mut cpass);
    }
    queue.submit(Some(encoder.finish()));
}

/// Pack a slice of `f32` values into packed bf16 u32 pairs (2 bf16 per u32).
/// Odd-length inputs are padded with a single zero bf16 in the high lane of
/// the final u32 — this matches how `MulMatWebgpu` pads its weight buffer.
pub fn pack_f32_to_bf16_u32(data: &[f32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(data.len().div_ceil(2));
    let mut iter = data.chunks_exact(2);
    for pair in &mut iter {
        let lo = half::bf16::from_f32(pair[0]).to_bits() as u32;
        let hi = half::bf16::from_f32(pair[1]).to_bits() as u32;
        out.push(lo | (hi << 16));
    }
    if let [last] = iter.remainder() {
        let lo = half::bf16::from_f32(*last).to_bits() as u32;
        out.push(lo);
    }
    out
}

/// Unpack a bf16 value from a packed u32 at the given element index.
pub fn unpack_bf16(packed: &[u32], index: usize) -> f32 {
    let word = packed[index / 2];
    let bits = if index % 2 == 0 {
        (word & 0xFFFF) as u16
    } else {
        (word >> 16) as u16
    };
    half::bf16::from_bits(bits).to_f32()
}

/// Assert two f32 slices are approximately equal with given tolerance.
pub fn assert_approx_eq(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "Length mismatch: actual={} expected={}",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        let rel = if e.abs() > 1e-6 { diff / e.abs() } else { diff };
        assert!(
            diff <= tol || rel <= tol,
            "Mismatch at index {i}: actual={a} expected={e} (abs_diff={diff}, rel_diff={rel}, tol={tol})"
        );
    }
}
