//! Typed configuration for the MiniCPM5 text decoder, plus the loader
//! that produces it.
//!
//! MiniCPM5 is a `LlamaForCausalLM` checkpoint: standard self-attention.

use std::path::Path;

use serde::Deserialize;

use super::super::ConfigLoadError;

/// Top-level `LlamaForCausalLM` config as it appears on disk.
#[derive(Debug, Clone, Deserialize)]
pub struct MiniCPM5Config {
    // -- transformer shape --
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,

    // -- attention --
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,

    // -- norm / activation --
    pub rms_norm_eps: f64,
    pub hidden_act: String,

    // -- positional / context window --
    pub max_position_embeddings: usize,
    pub rope_theta: f32,

    // -- vocab / embedding sharing --
    pub vocab_size: usize,
    /// MiniCPM5 sets this to `false` — `lm_head.weight` is a separate
    /// tensor in the safetensors, not aliased to `embed_tokens.weight`.
    pub tie_word_embeddings: bool,

    // -- runtime / training metadata --
    pub use_cache: bool,
    pub initializer_range: f32,
    pub model_type: String,
    pub torch_dtype: String,
    pub transformers_version: String,

    // -- tokens --
    pub bos_token_id: u32,
    /// MiniCPM5 ships `[1, 130073]` (`</s>` + `<|im_end|>`); either
    /// terminates generation.
    pub eos_token_id: Vec<u32>,
    pub pad_token_id: u32,

    // -- architecture marker (always `["LlamaForCausalLM"]`) --
    pub architectures: Vec<String>,
}

impl MiniCPM5Config {
    /// Parse a HuggingFace `config.json` (Llama / MiniCPM5 schema) from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Read and parse a HuggingFace `config.json` from disk.
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, ConfigLoadError> {
        let json = std::fs::read_to_string(path)?;
        Ok(Self::from_json(&json)?)
    }
}
