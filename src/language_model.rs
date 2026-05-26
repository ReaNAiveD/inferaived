mod core;
mod cpu;
mod gpu;

pub use core::Qwen35ModelCore;
pub use cpu::{Qwen35CpuModel, Qwen35CpuSession};
pub use gpu::{Qwen35GpuModel, Qwen35GpuSession};

pub enum LayerType {
    Linear,
    Full,
}

pub struct Qwen35Config {
    pub hidden_size: usize,

    pub layer_types: Vec<LayerType>,

    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,

    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub intermediate_size: usize,
}
