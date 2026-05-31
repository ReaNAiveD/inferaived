use safetensors::{SafeTensors, tensor::TensorView};

use crate::{
    kernels::norm::GemmaRmsNormInplaceWebgpu,
    layers::{
        linear_attention::LinearAttentionConfig,
        qwen35_layer_stack::{LayerConfig, LayerStack, LayerStackConfig},
        qwen35_self_attention::Qwen35SelfAttentionConfig,
    },
    lm_head::LmHeadWebgpu,
    log_tensor,
};

use super::{AttentionType, Qwen35TextConfig};

/// GPU mid-stack shared by both backends. Running the LM-head mat-mul
/// on CPU is impractical (vocab × hidden = 248K × 1024 bf16 muls per
/// step).
pub struct Qwen35ModelCore {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub layer_stack: LayerStack,
    pub final_norm: GemmaRmsNormInplaceWebgpu,
    pub lm_head: LmHeadWebgpu,
}

impl Qwen35ModelCore {
    /// Builds the shared GPU mid-stack.
    pub(super) fn new<'data>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tensors: &SafeTensors<'data>,
        config: &Qwen35TextConfig,
        embed_tokens: TensorView<'data>,
    ) -> Self {
        // The `Qwen35SelfAttentionLayer` bakes in the per-head Q/K norm and
        // the sigmoid output gate; both are intrinsic to Qwen3.5. Guard
        // against a checkpoint that disables the gate, which would need a
        // different layer type rather than a silently-wrong `q_proj` shape.
        assert!(
            config.attn_output_gate,
            "Qwen35SelfAttentionLayer requires attn_output_gate = true (got false)",
        );
        let self_attention_config = Qwen35SelfAttentionConfig {
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            rope_theta: config.rope_parameters.rope_theta,
            partial_rotary_factor: config.rope_parameters.partial_rotary_factor,
            intermediate_size: config.intermediate_size,
        };
        let linear_attention_config = LinearAttentionConfig {
            linear_num_key_heads: config.linear_num_key_heads,
            linear_num_value_heads: config.linear_num_value_heads,
            linear_key_head_dim: config.linear_key_head_dim,
            linear_value_head_dim: config.linear_value_head_dim,
            intermediate_size: config.intermediate_size,
        };
        let layers_config = config
            .layer_types
            .iter()
            .map(|layer_type| match layer_type {
                AttentionType::Linear => LayerConfig::Linear(linear_attention_config.clone()),
                AttentionType::Full => LayerConfig::Full(self_attention_config.clone()),
            })
            .collect();
        let layer_stack_config = LayerStackConfig {
            layers: layers_config,
        };
        let layer_stack = LayerStack::new(
            device,
            queue,
            tensors,
            "model.language_model",
            &layer_stack_config,
            config.hidden_size,
        );
        let final_norm_weight_name = "model.language_model.norm.weight";
        let final_norm_weight = tensors.tensor(final_norm_weight_name).expect(&format!(
            "Failed to get tensor for {}",
            final_norm_weight_name
        ));
        log_tensor(final_norm_weight_name, &final_norm_weight);
        let final_norm = GemmaRmsNormInplaceWebgpu::new(device, queue, final_norm_weight);
        let lm_head = LmHeadWebgpu::new(device, queue, embed_tokens);
        let vocab_size = lm_head.vocab_size();
        Self {
            hidden_size: config.hidden_size,
            vocab_size,
            layer_stack,
            final_norm,
            lm_head,
        }
    }
}
