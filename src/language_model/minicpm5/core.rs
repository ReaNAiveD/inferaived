use safetensors::{SafeTensors, tensor::TensorView};

use crate::{
    kernels::norm::LlamaRmsNormInplaceWebgpu,
    layers::{
        minicpm5_layer_stack::{MiniCPM5LayerStack, MiniCPM5LayerStackConfig},
        minicpm5_self_attention::MiniCPM5SelfAttentionConfig,
    },
    lm_head::LmHeadWebgpu,
    log_tensor,
};

use super::MiniCPM5Config;

/// Shared GPU mid-stack for MiniCPM5 (a pure `LlamaForCausalLM`):
/// 24 self-attention layers + final RMS norm + untied LM head.
pub struct MiniCPM5ModelCore {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub layer_stack: MiniCPM5LayerStack,
    pub final_norm: LlamaRmsNormInplaceWebgpu,
    pub lm_head: LmHeadWebgpu,
}

impl MiniCPM5ModelCore {
    pub(super) fn new<'data>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tensors: &SafeTensors<'data>,
        config: &MiniCPM5Config,
    ) -> Self {
        let self_attention_config = MiniCPM5SelfAttentionConfig {
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            rope_theta: config.rope_theta,
            intermediate_size: config.intermediate_size,
        };
        let layer_stack_config = MiniCPM5LayerStackConfig {
            num_hidden_layers: config.num_hidden_layers,
            layer: self_attention_config,
        };
        let layer_stack = MiniCPM5LayerStack::new(
            device,
            queue,
            tensors,
            "model",
            &layer_stack_config,
            config.hidden_size,
        );
        let final_norm_weight_name = "model.norm.weight";
        let final_norm_weight = tensors.tensor(final_norm_weight_name).expect(&format!(
            "Failed to get tensor for {}",
            final_norm_weight_name
        ));
        log_tensor(final_norm_weight_name, &final_norm_weight);

        let final_norm = LlamaRmsNormInplaceWebgpu::new(device, queue, final_norm_weight);

        let lm_head_weight = lm_head_tensor(tensors, config);
        let lm_head = LmHeadWebgpu::new(device, queue, lm_head_weight);
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

/// Pick the right tensor for the LM head matmul. `tie_word_embeddings`
/// determines whether the head reuses `embed_tokens.weight` or has its
/// own `lm_head.weight`. Both are exposed as `[vocab, hidden]`.
pub(super) fn lm_head_tensor<'data>(
    tensors: &SafeTensors<'data>,
    config: &MiniCPM5Config,
) -> TensorView<'data> {
    let name = if config.tie_word_embeddings {
        "model.embed_tokens.weight"
    } else {
        "lm_head.weight"
    };
    let tensor = tensors
        .tensor(name)
        .expect(&format!("Failed to get tensor for {}", name));
    log_tensor(name, &tensor);
    tensor
}
