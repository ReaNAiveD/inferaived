use std::path::Path;

use serde::Deserialize;

use super::super::ConfigLoadError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum AttentionType {
    #[serde(rename = "linear_attention")]
    Linear,
    #[serde(rename = "full_attention")]
    Full,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeConfig {
    pub rope_type: String,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    pub mrope_interleaved: bool,
    pub mrope_section: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35TextConfig {
    // -- transformer shape --
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub layer_types: Vec<AttentionType>,
    pub full_attention_interval: usize,
    pub mlp_only_layers: Vec<usize>,

    // -- full attention --
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub attention_bias: bool,
    pub attention_dropout: f32,
    pub attn_output_gate: bool,

    // -- linear (delta-rule) attention --
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub mamba_ssm_dtype: String,

    // -- norms / activations --
    pub rms_norm_eps: f64,
    pub hidden_act: String,

    // -- positional / context window --
    pub max_position_embeddings: usize,
    pub rope_parameters: RopeConfig,

    // -- vocab / embedding sharing --
    pub vocab_size: usize,
    pub tie_word_embeddings: bool,

    // -- multi-token prediction (speculative-decoding draft head) --
    pub mtp_num_hidden_layers: usize,
    pub mtp_use_dedicated_embeddings: bool,

    // -- runtime / training metadata --
    pub use_cache: bool,
    pub initializer_range: f32,
    pub model_type: String,
    pub dtype: String,
    pub eos_token_id: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub hidden_act: String,
    pub in_channels: usize,
    pub intermediate_size: usize,
    pub initializer_range: f32,
    pub num_heads: usize,
    pub num_position_embeddings: usize,
    pub out_hidden_size: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub deepstack_visual_indexes: Vec<usize>,
    pub model_type: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35Config {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub transformers_version: String,
    pub tie_word_embeddings: bool,
    pub text_config: Qwen35TextConfig,
    pub vision_config: Qwen35VisionConfig,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
}

impl Qwen35Config {
    /// Parse a Hugging Face `config.json` (Qwen3.5 schema) from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        Ok(serde_json::from_str(json)?)
    }

    /// Read and parse a Hugging Face `config.json` from disk.
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, ConfigLoadError> {
        let json = std::fs::read_to_string(path)?;
        Ok(Self::from_json(&json)?)
    }
}
