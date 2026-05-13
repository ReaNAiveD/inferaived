use half::bf16;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use safetensors::tensor::TensorView;

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
        (0..self.vocab_size).into_par_iter().map(|i| {
            let start = i * self.hidden_size * std::mem::size_of::<bf16>();
            let end = start + self.hidden_size * std::mem::size_of::<bf16>();
            let row_data = &self.embed_tokens.data()[start..end];
            row_data.chunks_exact(2).zip(input_vector.iter()).map(|(chunk, &input_val)| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                bf16::from_bits(bits).to_f32() * input_val
            }).sum()
        }).collect()
    }
}
