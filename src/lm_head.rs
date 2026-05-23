use half::bf16;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use safetensors::tensor::TensorView;

use crate::{
    buffer_view::BufferView,
    kernels::mul_mat::{MulMatWebgpu, MulMatWebgpuRunner},
};

/// GPU LM head: a thin wrapper around [`MulMatWebgpu`] that owns the
/// `vocab_size × hidden_size` bf16 weight buffer (typically the tied
/// `embed_tokens` table) and produces a `1 × vocab_size` f32 logits row
/// per `forward`.
pub struct LmHeadWebgpu {
    vocab_size: usize,
    hidden_size: usize,
    mul_mat: MulMatWebgpu,
}

impl LmHeadWebgpu {
    pub fn new<'data>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        embed_tokens: TensorView<'data>,
    ) -> Self {
        debug_assert_eq!(
            embed_tokens.shape().len(),
            2,
            "LmHeadWebgpu: embed_tokens must be 2-D [vocab, hidden], got shape {:?}",
            embed_tokens.shape(),
        );
        let vocab_size = embed_tokens.shape()[0];
        let hidden_size = embed_tokens.shape()[1];
        let mul_mat = MulMatWebgpu::new(device, queue, embed_tokens);
        Self {
            vocab_size,
            hidden_size,
            mul_mat,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn plan(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        hidden_last_row: BufferView<'_>,
        logits: BufferView<'_>,
    ) -> LmHeadWebgpuRunner {
        debug_assert_eq!(
            hidden_last_row.shape[0], 1,
            "LmHeadWebgpu: input must have 1 row, got {}",
            hidden_last_row.shape[0],
        );
        debug_assert_eq!(
            hidden_last_row.shape[1] as usize, self.hidden_size,
            "LmHeadWebgpu: input hidden_size {} != model hidden_size {}",
            hidden_last_row.shape[1], self.hidden_size,
        );
        debug_assert_eq!(
            logits.shape[0], 1,
            "LmHeadWebgpu: logits must have 1 row, got {}",
            logits.shape[0],
        );
        debug_assert_eq!(
            logits.shape[1] as usize, self.vocab_size,
            "LmHeadWebgpu: logits vocab {} != model vocab {}",
            logits.shape[1], self.vocab_size,
        );
        let mul_mat_runner = self.mul_mat.plan(device, queue, hidden_last_row, logits);
        LmHeadWebgpuRunner { mul_mat_runner }
    }
}

pub struct LmHeadWebgpuRunner {
    mul_mat_runner: MulMatWebgpuRunner,
}

impl LmHeadWebgpuRunner {
    pub fn forward(&self, cpass: &mut wgpu::ComputePass<'_>) {
        self.mul_mat_runner.forward(cpass);
    }
}

/// CPU reference implementation.
pub struct LmHeadCpu<'data> {
    hidden_size: usize,
    vocab_size: usize,
    embed_tokens: TensorView<'data>,
}

impl<'data> LmHeadCpu<'data> {
    pub fn new(embed_tokens: TensorView<'data>) -> Self {
        let hidden_size = embed_tokens.shape()[1];
        let vocab_size = embed_tokens.shape()[0];
        Self {
            hidden_size,
            vocab_size,
            embed_tokens,
        }
    }

    pub fn compute(&self, input_vector: &[f32]) -> Vec<f32> {
        debug_assert_eq!(input_vector.len(), self.hidden_size);
        (0..self.vocab_size)
            .into_par_iter()
            .map(|i| {
                let start = i * self.hidden_size * std::mem::size_of::<bf16>();
                let end = start + self.hidden_size * std::mem::size_of::<bf16>();
                let row_data = &self.embed_tokens.data()[start..end];
                row_data
                    .chunks_exact(2)
                    .zip(input_vector.iter())
                    .map(|(chunk, &input_val)| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        bf16::from_bits(bits).to_f32() * input_val
                    })
                    .sum()
            })
            .collect()
    }
}
