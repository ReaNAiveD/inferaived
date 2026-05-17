pub const MAX_DIMS: usize = 4;

#[derive(Debug, Clone, Copy)]
pub struct BufferView<'a> {
    pub buffer: &'a wgpu::Buffer,
    pub byte_offset: wgpu::BufferAddress,
    pub rank: u8,
    pub shape: [u32; MAX_DIMS],
    pub stride: [u32; MAX_DIMS],
    pub elem_size: u32,
}

impl<'a> BufferView<'a> {
    pub fn new_1d(buffer: &'a wgpu::Buffer, elem_size: u32, len: u32) -> Self {
        debug_assert!(elem_size > 0 && len > 0, "new_1d: dims must be > 0");
        Self {
            buffer,
            byte_offset: 0,
            rank: 1,
            shape: [len, 1, 1, 1],
            stride: [1, 0, 0, 0],
            elem_size,
        }
    }

    pub fn new_2d_tight(buffer: &'a wgpu::Buffer, outer: u32, inner: u32, elem_size: u32) -> Self {
        debug_assert!(
            outer > 0 && inner > 0 && elem_size > 0,
            "new_2d_tight: dims must be > 0",
        );
        Self {
            buffer,
            byte_offset: 0,
            rank: 2,
            shape: [outer, inner, 1, 1],
            stride: [inner, 1, 0, 0],
            elem_size,
        }
    }

    pub fn new_3d_tight(
        buffer: &'a wgpu::Buffer,
        d0: u32,
        d1: u32,
        d2: u32,
        elem_size: u32,
    ) -> Self {
        debug_assert!(
            d0 > 0 && d1 > 0 && d2 > 0 && elem_size > 0,
            "new_3d_tight: dims must be > 0",
        );
        Self {
            buffer,
            byte_offset: 0,
            rank: 3,
            shape: [d0, d1, d2, 1],
            stride: [d1 * d2, d2, 1, 0],
            elem_size,
        }
    }

    pub fn new_4d_tight(
        buffer: &'a wgpu::Buffer,
        d0: u32,
        d1: u32,
        d2: u32,
        d3: u32,
        elem_size: u32,
    ) -> Self {
        debug_assert!(
            d0 > 0 && d1 > 0 && d2 > 0 && d3 > 0 && elem_size > 0,
            "new_4d_tight: dims must be > 0",
        );
        Self {
            buffer,
            byte_offset: 0,
            rank: 4,
            shape: [d0, d1, d2, d3],
            stride: [d1 * d2 * d3, d2 * d3, d3, 1],
            elem_size,
        }
    }

    pub fn from_raw(
        buffer: &'a wgpu::Buffer,
        byte_offset: wgpu::BufferAddress,
        rank: u8,
        shape: [u32; MAX_DIMS],
        stride: [u32; MAX_DIMS],
        elem_size: u32,
    ) -> Self {
        debug_assert!((1..=MAX_DIMS as u8).contains(&rank), "rank out of range");
        debug_assert!(elem_size > 0, "from_raw: elem_size must be > 0");
        debug_assert_eq!(
            byte_offset % elem_size as wgpu::BufferAddress,
            0,
            "from_raw: byte_offset ({}) must be a multiple of elem_size ({})",
            byte_offset,
            elem_size,
        );
        for i in 0..rank as usize {
            debug_assert!(shape[i] > 0, "from_raw: shape[{}] must be > 0", i);
        }
        Self {
            buffer,
            byte_offset,
            rank,
            shape,
            stride,
            elem_size,
        }
    }

    // -------------------------------------------------------------------
    // View ops — all cheap, no GPU work, no allocations.
    // -------------------------------------------------------------------

    /// Restrict dim `dim` to the contiguous range `[start, start + len)`.
    pub fn narrow(mut self, dim: usize, start: u32, len: u32) -> Self {
        debug_assert!(
            (dim as u8) < self.rank,
            "narrow: dim {} out of range for rank {}",
            dim,
            self.rank,
        );
        debug_assert!(
            start + len <= self.shape[dim],
            "narrow: start+len ({}) exceeds shape[{}] ({})",
            start + len,
            dim,
            self.shape[dim],
        );
        debug_assert!(len > 0, "narrow: len must be > 0");
        self.byte_offset += start as wgpu::BufferAddress
            * self.stride[dim] as wgpu::BufferAddress
            * self.elem_size as wgpu::BufferAddress;
        self.shape[dim] = len;
        self
    }

    /// Drop dim `dim` by fixing it at `index`.
    pub fn select(self, dim: usize, index: u32) -> Self {
        debug_assert!(
            (dim as u8) < self.rank,
            "select: dim {} out of range for rank {}",
            dim,
            self.rank,
        );
        debug_assert!(
            index < self.shape[dim],
            "select: index {} out of range for shape[{}]={}",
            index,
            dim,
            self.shape[dim],
        );
        let new_byte_offset = self.byte_offset
            + index as wgpu::BufferAddress
                * self.stride[dim] as wgpu::BufferAddress
                * self.elem_size as wgpu::BufferAddress;

        let mut shape = self.shape;
        let mut stride = self.stride;
        for i in dim..(self.rank as usize - 1) {
            shape[i] = shape[i + 1];
            stride[i] = stride[i + 1];
        }
        // Clear the now-unused slot.
        let last = self.rank as usize - 1;
        shape[last] = 1;
        stride[last] = 0;

        Self {
            buffer: self.buffer,
            byte_offset: new_byte_offset,
            rank: self.rank - 1,
            shape,
            stride,
            elem_size: self.elem_size,
        }
    }

    /// Merge the outermost `n` dims into one.
    pub fn flatten_outer(self, n: usize) -> Self {
        debug_assert!(n >= 1, "flatten_outer: n must be >= 1");
        if n == 1 {
            return self;
        }
        debug_assert!(
            n as u8 <= self.rank,
            "flatten_outer: cannot flatten {} dims of rank-{} view",
            n,
            self.rank,
        );
        for i in 0..n - 1 {
            let expected = self.shape[i + 1] as u64 * self.stride[i + 1] as u64;
            debug_assert_eq!(
                self.stride[i] as u64,
                expected,
                "flatten_outer: dims {}..{} are not stride-compatible \
                 (stride[{}]={}, shape[{}]={}, stride[{}]={})",
                i,
                i + 1,
                i,
                self.stride[i],
                i + 1,
                self.shape[i + 1],
                i + 1,
                self.stride[i + 1],
            );
        }
        let mut merged_shape: u64 = 1;
        for i in 0..n {
            merged_shape *= self.shape[i] as u64;
        }
        debug_assert!(
            merged_shape <= u32::MAX as u64,
            "flatten_outer: merged shape {} exceeds u32::MAX",
            merged_shape,
        );

        let merged_stride = self.stride[n - 1];
        let mut shape = [1u32; MAX_DIMS];
        let mut stride = [0u32; MAX_DIMS];
        shape[0] = merged_shape as u32;
        stride[0] = merged_stride;
        for i in n..self.rank as usize {
            shape[i - (n - 1)] = self.shape[i];
            stride[i - (n - 1)] = self.stride[i];
        }

        Self {
            buffer: self.buffer,
            byte_offset: self.byte_offset,
            rank: self.rank - (n as u8 - 1),
            shape,
            stride,
            elem_size: self.elem_size,
        }
    }

    // -------------------------------------------------------------------
    // Queries.
    // -------------------------------------------------------------------

    /// Smallest contiguous byte span of the underlying buffer that
    /// contains every element addressed by this view.
    pub fn total_byte_size(&self) -> wgpu::BufferAddress {
        let mut span_elements: u64 = 1;
        for i in 0..self.rank as usize {
            // (shape[i] - 1) * stride[i] reaches the start of the last
            // element along this dim; +1 element at the end covers the
            // innermost element itself.
            span_elements += (self.shape[i] as u64 - 1) * self.stride[i] as u64;
        }
        span_elements * self.elem_size as wgpu::BufferAddress
    }

    /// Produce a `wgpu::BindingResource` covering exactly this view's span.
    pub fn as_binding(&self) -> wgpu::BindingResource<'a> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: self.buffer,
            offset: self.byte_offset,
            size: wgpu::BufferSize::new(self.total_byte_size()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_test_utils::gpu_or_skip;
    use wgpu::{BufferDescriptor, BufferUsages};

    fn dummy_buffer(device: &wgpu::Device, byte_size: u64) -> wgpu::Buffer {
        device.create_buffer(&BufferDescriptor {
            label: Some("test/dummy_buffer"),
            size: byte_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    #[test]
    fn new_2d_tight_matches_row_layout() {
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let (device, _queue) = gpu_or_skip!();
            let buf = dummy_buffer(&device, 64 * 1024);
            let v = BufferView::new_2d_tight(&buf, 6, 64, 4);
            assert_eq!(v.rank, 2);
            assert_eq!(v.shape[0..2], [6, 64]);
            assert_eq!(v.stride[0..2], [64, 1]);
            assert_eq!(v.elem_size, 4);
            assert_eq!(v.total_byte_size(), 6 * 64 * 4);
        });
    }

    #[test]
    fn select_drops_dim_and_bumps_offset() {
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let (device, _queue) = gpu_or_skip!();
            let buf = dummy_buffer(&device, 64 * 1024);
            // Fused q_gate layout: [seq=3, heads=2, q_or_gate=2, head_dim=4] f32.
            let v4 = BufferView::new_4d_tight(&buf, 3, 2, 2, 4, 4);
            assert_eq!(v4.stride[0..4], [2 * 2 * 4, 2 * 4, 4, 1]);

            // Q half: select(dim=2, index=0).
            let q = v4.select(2, 0);
            assert_eq!(q.rank, 3);
            assert_eq!(q.byte_offset, 0);
            assert_eq!(q.shape[0..3], [3, 2, 4]);
            // Per-head element-stride from the dropped dim's neighbour:
            // it's the OLD stride[1] = 2 * 4 = 8 elements.
            assert_eq!(q.stride[0..3], [2 * 2 * 4, 2 * 4, 1]);

            // Gate half: select(dim=2, index=1) → offset bumped by
            // 1 * old_stride[2] * elem_size = 1 * 4 * 4 = 16 bytes.
            let gate = v4.select(2, 1);
            assert_eq!(gate.byte_offset, 16);
            assert_eq!(gate.shape[0..3], [3, 2, 4]);
            assert_eq!(gate.stride[0..3], q.stride[0..3]);
        });
    }

    #[test]
    fn narrow_shrinks_one_dim_no_copy() {
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let (device, _queue) = gpu_or_skip!();
            let buf = dummy_buffer(&device, 64 * 1024);
            // [10, 32] tight, f32.
            let v = BufferView::new_2d_tight(&buf, 10, 32, 4);
            // Take rows [4..7) — 3 rows.
            let sub = v.narrow(0, 4, 3);
            assert_eq!(sub.rank, 2);
            assert_eq!(sub.shape[0..2], [3, 32]);
            // Offset = 4 * stride[0] * elem_size = 4 * 32 * 4.
            assert_eq!(sub.byte_offset, 4 * 32 * 4);
            assert_eq!(sub.stride, v.stride);
        });
    }

    #[test]
    fn flatten_outer_merges_compatible_dims() {
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let (device, _queue) = gpu_or_skip!();
            let buf = dummy_buffer(&device, 64 * 1024);
            // Fused q half: select(2, 0) from [3, 2, 2, 4] →
            // [3, 2, 4] with stride = [16, 8, 1] (elements).
            let q = BufferView::new_4d_tight(&buf, 3, 2, 2, 4, 4).select(2, 0);
            // Outer two dims are stride-compatible:
            //   stride[0] (16) == shape[1] (2) * stride[1] (8)
            // so they can be flattened without copy.
            let flat = q.flatten_outer(2);
            assert_eq!(flat.rank, 2);
            assert_eq!(flat.shape[0..2], [3 * 2, 4]);
            assert_eq!(flat.stride[0..2], [8, 1]);
            assert_eq!(flat.byte_offset, q.byte_offset);
        });
    }

    #[test]
    #[should_panic(expected = "not stride-compatible")]
    fn flatten_outer_rejects_incompatible_strides() {
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let (device, _queue) = match crate::gpu_test_utils::create_device_queue().await {
                Some(dq) => dq,
                None => panic!("not stride-compatible: no-GPU stub"),
            };
            let buf = dummy_buffer(&device, 64 * 1024);
            // Incompatible: stride[0] != shape[1] * stride[1].
            let v = BufferView::from_raw(
                &buf,
                0,
                3,
                [3, 2, 4, 1],
                [100, 8, 1, 0], // 100 != 2 * 8
                4,
            );
            let _ = v.flatten_outer(2);
        });
    }

    #[test]
    fn select_then_select_navigates_4d_to_2d() {
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let (device, _queue) = gpu_or_skip!();
            let buf = dummy_buffer(&device, 64 * 1024);
            let v4 = BufferView::new_4d_tight(&buf, 3, 2, 2, 4, 4);
            // Pick gate half.
            let gate = v4.select(2, 1);
            // From the rank-3 [seq=3, heads=2, head_dim=4], pick head 0.
            let head0 = gate.select(1, 0);
            assert_eq!(head0.rank, 2);
            assert_eq!(head0.shape[0..2], [3, 4]);
            // Per-token element-stride preserved from the rank-3 view.
            assert_eq!(head0.stride[0..2], [2 * 2 * 4, 1]);
            // Offset stays at gate.byte_offset (16) — head 0 is at 0.
            assert_eq!(head0.byte_offset, 16);
        });
    }

    /// End-to-end: a `BufferView` produced by the new API binds cleanly.
    #[test]
    fn view_makes_a_valid_bind_group() {
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            let (device, _queue) = gpu_or_skip!();
            let buf = dummy_buffer(&device, 64 * 1024);
            let v_a = BufferView::new_2d_tight(&buf, 1, 1024, 4);
            let v_b = BufferView::new_2d_tight(&buf, 1, 1024, 4).narrow(1, 0, 256);

            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("test/view_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
            let _bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("test/view_bg"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: v_a.as_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: v_b.as_binding(),
                    },
                ],
            });
        });
    }
}
