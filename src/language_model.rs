use safetensors::SafeTensors;

use crate::{
    embedding_lookup::EmbeddingLookupCpu,
    layer_loop::{
        LayerConfig, LayerStack, LayerStackConfig, LinearAttentionConfig, SelfAttentionConfig,
    },
    lm_head::LmHeadCpu,
    log_tensor,
    norm::RmsNormInplaceWebgpu,
    sampler::ArgmaxSamplerCpu,
};

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

pub struct Qwen35Model<'data> {
    pub hidden_size: usize,

    pub embedding_lookup: EmbeddingLookupCpu<'data>,
    pub layer_stack: LayerStack,
    pub final_norm: RmsNormInplaceWebgpu,
    pub last_hidden_readback_buffer: wgpu::Buffer,
    pub lm_head: LmHeadCpu<'data>,
    pub sampler: ArgmaxSamplerCpu,
}

impl<'data> Qwen35Model<'data> {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tensors: &SafeTensors<'data>,
        config: &Qwen35Config,
    ) -> Self {
        let embed_tokens = tensors
            .tensor("model.language_model.embed_tokens.weight")
            .expect("Failed to get tensor: model.language_model.embed_tokens.weight");
        let embedding_lookup = EmbeddingLookupCpu::new(embed_tokens.clone());
        let self_attention_config = SelfAttentionConfig {
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            rope_theta: config.rope_theta,
            partial_rotary_factor: config.partial_rotary_factor,
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
                LayerType::Linear => LayerConfig::Linear(linear_attention_config.clone()),
                LayerType::Full => LayerConfig::Full(self_attention_config.clone()),
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
        let final_norm =
            RmsNormInplaceWebgpu::new(device, queue, final_norm_weight);
        let last_hidden_readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qwen35_model/last_hidden_readback_buffer"),
            size: (config.hidden_size * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lm_head = LmHeadCpu::new(embed_tokens.clone());
        let sampler = ArgmaxSamplerCpu;
        Self {
            hidden_size: config.hidden_size,
            embedding_lookup,
            layer_stack,
            final_norm,
            lm_head,
            last_hidden_readback_buffer,
            sampler,
        }
    }

    pub async fn compute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_ids: &[u32],
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        let token_embeddings = self.embedding_lookup.compute(input_ids);
        let hidden_states_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qwen35_model/hidden_states_buffer"),
            size: (token_embeddings.len() * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &hidden_states_buffer,
            0,
            bytemuck::cast_slice(&token_embeddings),
        );
        self.layer_stack
            .compute(device, queue, &hidden_states_buffer, input_ids.len());
        self.final_norm
            .compute(device, queue, &hidden_states_buffer, input_ids.len());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("qwen35_model/last_hidden_readback_encoder"),
        });
        let last_hidden_byte_offset =
            ((input_ids.len() - 1) * self.hidden_size * std::mem::size_of::<f32>())
                as wgpu::BufferAddress;
        let last_hidden_byte_size =
            (self.hidden_size * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
        encoder.copy_buffer_to_buffer(
            &hidden_states_buffer,
            last_hidden_byte_offset,
            &self.last_hidden_readback_buffer,
            0,
            last_hidden_byte_size,
        );
        let readback_submission_index = queue.submit(Some(encoder.finish()));
        let slice = self.last_hidden_readback_buffer.slice(..);
        let (tx, rx) = tokio::sync::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: Some(readback_submission_index),
            timeout: None,
        });
        rx.await
            .expect("Failed to map buffer")
            .expect("Failed to map buffer");
        let last_hidden_state = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec();
        self.last_hidden_readback_buffer.unmap();
        let logits = self.lm_head.compute(&last_hidden_state);
        self.sampler.sample(&logits, top_k)
    }
}
