use safetensors::{SafeTensors, tensor::TensorView};

use crate::{
    kernels::norm::ParameterlessRmsNormWebgpu,
    layers::{
        hrm_layer_stack::{HrmLayerStack, HrmLayerStackConfig},
        hrm_self_attention::HrmSelfAttentionConfig,
    },
    lm_head::LmHeadWebgpu,
    log_tensor,
};

use super::HrmTextConfig;

/// Decode a 1-D bf16 tensor into f32.
fn load_bf16_vec(weight: &TensorView<'_>, what: &str) -> Vec<f32> {
    debug_assert_eq!(
        weight.shape().len(),
        1,
        "{what} must be 1-D, got shape {:?}",
        weight.shape(),
    );
    weight
        .data()
        .chunks_exact(2)
        .map(|chunk| half::bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
        .collect()
}

/// Shared GPU components for HRM-Text: the two weight-shared recurrent stacks
/// (`H_module` slow, `L_module` fast), the learned `z_L_init` fast-loop seed,
/// the parameterless module-final RMSNorm (`norm_f`), and the untied LM head.
///
/// The recurrent driver that loops the stacks with additive `z_L + z_H`
/// injection lives in [`super::gpu`].
pub struct HrmTextModelCore {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub h_cycles: usize,
    pub l_cycles: usize,
    pub embedding_scale: f32,

    pub h_stack: HrmLayerStack,
    pub l_stack: HrmLayerStack,
    /// `[hidden]` f32 seed for the fast-loop state, broadcast across positions.
    pub z_l_init: wgpu::Buffer,
    /// Parameterless RMSNorm applied at the end of every stack invocation
    /// (`Transformer.norm_f` in the reference, shared by H and L because it has
    /// no learnable gain).
    pub norm_f: ParameterlessRmsNormWebgpu,
    pub lm_head: LmHeadWebgpu,
}

impl HrmTextModelCore {
    pub(super) fn new<'data>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tensors: &SafeTensors<'data>,
        config: &HrmTextConfig,
    ) -> Self {
        let layer_config = HrmSelfAttentionConfig {
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            rope_theta: config.rope_theta,
            intermediate_size: config.intermediate_size,
        };
        let stack_config = HrmLayerStackConfig {
            num_hidden_layers: config.num_hidden_layers,
            layer: layer_config,
        };
        let h_stack = HrmLayerStack::new(
            device,
            queue,
            tensors,
            "model.H_module",
            &stack_config,
            config.hidden_size,
        );
        let l_stack = HrmLayerStack::new(
            device,
            queue,
            tensors,
            "model.L_module",
            &stack_config,
            config.hidden_size,
        );

        let z_l_init_name = "model.z_L_init";
        let z_l_init_tensor = tensors
            .tensor(z_l_init_name)
            .expect(&format!("Failed to get tensor for {}", z_l_init_name));
        log_tensor(z_l_init_name, &z_l_init_tensor);
        let z_l_init_f32 = load_bf16_vec(&z_l_init_tensor, z_l_init_name);
        debug_assert_eq!(
            z_l_init_f32.len(),
            config.hidden_size,
            "{} length {} != hidden_size {}",
            z_l_init_name,
            z_l_init_f32.len(),
            config.hidden_size,
        );
        let z_l_init = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hrm_text/z_L_init"),
            size: (z_l_init_f32.len() * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&z_l_init, 0, bytemuck::cast_slice(&z_l_init_f32));

        let norm_f = ParameterlessRmsNormWebgpu::new(device, queue, config.hidden_size);

        let lm_head_weight = lm_head_tensor(tensors, config);
        let lm_head = LmHeadWebgpu::new(device, queue, lm_head_weight);
        let vocab_size = lm_head.vocab_size();

        Self {
            hidden_size: config.hidden_size,
            vocab_size,
            h_cycles: config.h_cycles,
            l_cycles: config.l_cycles,
            embedding_scale: config.embedding_scale,
            h_stack,
            l_stack,
            z_l_init,
            norm_f,
            lm_head,
        }
    }
}

/// Pick the LM-head weight. HRM-Text sets `tie_word_embeddings = false`, so the
/// head normally has its own `lm_head.weight`; the tied branch is kept for
/// completeness. Both are `[vocab, hidden]`.
pub(super) fn lm_head_tensor<'data>(
    tensors: &SafeTensors<'data>,
    config: &HrmTextConfig,
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
