use half::bf16;
use safetensors::tensor::TensorView;

use crate::{
    buffer_view::BufferView,
    kernels::get_rows::{GetRows, GetRowsRunner},
};

pub struct EmbeddingLookupCpu<'data> {
    hidden_size: usize,
    embed_tokens: TensorView<'data>,
}

impl<'data> EmbeddingLookupCpu<'data> {
    pub fn new(embed_tokens: TensorView<'data>) -> Self {
        let hidden_size = embed_tokens.shape()[1];
        Self {
            hidden_size,
            embed_tokens,
        }
    }

    pub fn compute(&self, input_encoding: &[u32]) -> Vec<f32> {
        let row_width = self.hidden_size * std::mem::size_of::<bf16>();
        let embedding_row_num = self.embed_tokens.data().len() / row_width;
        input_encoding
            .iter()
            .flat_map(|&idx| {
                let start = ((idx as usize) % embedding_row_num) * row_width;
                let end = start + row_width;
                let row_data = &self.embed_tokens.data()[start..end];
                // The safetensors store bf16 values as little-endian
                let row_floats: Vec<f32> = row_data
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        bf16::from_bits(bits).to_f32()
                    })
                    .collect();
                row_floats
            })
            .collect()
    }
}

pub struct EmbeddingLookupWebgpu {
    hidden_size: usize,
    kernel: GetRows,
}

impl EmbeddingLookupWebgpu {
    pub fn new<'data>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        embed_tokens: TensorView<'data>,
    ) -> Self {
        debug_assert_eq!(
            embed_tokens.shape().len(),
            2,
            "EmbeddingLookupWebgpu: embed_tokens must be 2-D [vocab, hidden], got shape {:?}",
            embed_tokens.shape(),
        );
        let hidden_size = embed_tokens.shape()[1];
        let kernel = GetRows::new(device, queue, embed_tokens);
        Self {
            hidden_size,
            kernel,
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn plan(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tokens: BufferView<'_>,
        dst: BufferView<'_>,
    ) -> EmbeddingLookupWebgpuRunner {
        debug_assert_eq!(
            dst.shape[1] as usize, self.hidden_size,
            "EmbeddingLookupWebgpu: dst inner dim {} != model hidden_size {}",
            dst.shape[1], self.hidden_size,
        );
        let runner = self.kernel.plan(device, queue, tokens, dst);
        EmbeddingLookupWebgpuRunner { runner }
    }
}

pub struct EmbeddingLookupWebgpuRunner {
    runner: GetRowsRunner,
}

impl EmbeddingLookupWebgpuRunner {
    pub fn forward(&self, cpass: &mut wgpu::ComputePass<'_>) {
        self.runner.forward(cpass);
    }
}
