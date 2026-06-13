//! Typed configuration for the HRM-Text decoder, plus the disk loader.
//!
//! HRM-Text (`hrm_text` / `HrmTextForCausalLM`) is a dual-timescale recurrent
//! decoder: two weight-shared transformer stacks (`H_module` slow, `L_module`
//! fast) iterated `H_cycles × (L_cycles + 1)` times per forward.

use std::path::Path;

use serde::Deserialize;

use super::super::ConfigLoadError;

/// Top-level `hrm_text` config as it appears on disk.
#[derive(Debug, Clone, Deserialize)]
pub struct HrmTextConfig {
    // -- transformer shape (per H/L stack) --
    pub hidden_size: usize,
    pub intermediate_size: usize,
    /// Number of layers in EACH of the `H_module` and `L_module` stacks.
    pub num_hidden_layers: usize,

    // -- attention --
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,

    // -- recurrence --
    #[serde(rename = "H_cycles")]
    pub h_cycles: usize,
    #[serde(rename = "L_cycles")]
    pub l_cycles: usize,

    // -- norm / positional --
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,

    // -- embedding / head --
    pub vocab_size: usize,
    /// Multiplier applied to the looked-up token embeddings before the
    /// recurrent core (`z_H = embed(input_ids) * embedding_scale`).
    pub embedding_scale: f32,
    /// HRM-Text ships a separate `lm_head.weight` (not tied to embeddings).
    pub tie_word_embeddings: bool,

    // -- prefix-LM marker --
    pub prefix_lm: bool,

    // -- tokens --
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,

    // -- architecture marker (`["HrmTextForCausalLM"]`) --
    pub architectures: Vec<String>,
}

impl HrmTextConfig {
    /// Parse a HuggingFace `config.json` (`hrm_text` schema) from a JSON string.
    ///
    /// The on-disk file spells the recurrence keys `H_cycles` / `L_cycles`;
    /// serde maps them to the lowercase fields via the field renames declared
    /// above.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Read and parse a HuggingFace `config.json` from disk.
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self, ConfigLoadError> {
        let json = std::fs::read_to_string(path)?;
        Ok(Self::from_json(&json)?)
    }
}
