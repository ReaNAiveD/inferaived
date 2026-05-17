use safetensors::SafeTensors;

use crate::{
    buffer_view::BufferView,
    embedding_lookup::EmbeddingLookupCpu,
    layer_loop::{
        LayerConfig, LayerSession, LayerStack, LayerStackConfig, LayerStackSession,
        LinearAttentionConfig, SelfAttentionConfig,
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
        let final_norm = RmsNormInplaceWebgpu::new(device, queue, final_norm_weight);
        let lm_head = LmHeadCpu::new(embed_tokens.clone());
        let sampler = ArgmaxSamplerCpu;
        Self {
            hidden_size: config.hidden_size,
            embedding_lookup,
            layer_stack,
            final_norm,
            lm_head,
            sampler,
        }
    }
}

pub struct Qwen35Session<'m, 'data> {
    model: &'m Qwen35Model<'data>,
    layer_session: LayerStackSession<'m>,
    hidden_states_buffer: wgpu::Buffer,
    last_hidden_readback_buffer: wgpu::Buffer,
    position: usize,
    max_seq_len: usize,
}

impl<'m, 'data> Qwen35Session<'m, 'data> {
    pub fn new(model: &'m Qwen35Model<'data>, device: &wgpu::Device, max_seq_len: usize) -> Self {
        debug_assert!(max_seq_len >= 1, "max_seq_len must be >= 1");
        let layer_session = LayerStackSession::new(&model.layer_stack, device, max_seq_len);
        let hidden_states_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qwen35_session/hidden_states_buffer"),
            size: (max_seq_len * model.hidden_size * std::mem::size_of::<f32>())
                as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let last_hidden_readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qwen35_session/last_hidden_readback_buffer"),
            size: (model.hidden_size * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            model,
            layer_session,
            hidden_states_buffer,
            last_hidden_readback_buffer,
            position: 0,
            max_seq_len,
        }
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Append `input_ids` to the session, run the model on just those new
    /// tokens, and return the top-`top_k` candidates for the immediately
    /// following token (i.e. the token at absolute position
    /// `position() + input_ids.len()`).
    pub async fn forward(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_ids: &[u32],
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        let num_new = input_ids.len();
        debug_assert!(num_new >= 1, "forward requires non-empty input_ids");
        debug_assert!(
            self.position + num_new <= self.max_seq_len,
            "session overflow: position {} + {} new tokens exceeds max_seq_len {}",
            self.position,
            num_new,
            self.max_seq_len,
        );
        let prev_position = self.position;
        let hidden_size = self.model.hidden_size;

        let token_embeddings = self.model.embedding_lookup.compute(input_ids);
        let dst_byte_offset =
            (prev_position * hidden_size * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
        queue.write_buffer(
            &self.hidden_states_buffer,
            dst_byte_offset,
            bytemuck::cast_slice(&token_embeddings),
        );

        let new_token_rows = BufferView::new_2d_tight(
            &self.hidden_states_buffer,
            self.max_seq_len as u32,
            hidden_size as u32,
            std::mem::size_of::<f32>() as u32,
        )
        .narrow(0, prev_position as u32, num_new as u32);

        self.layer_session
            .forward(device, queue, new_token_rows, prev_position);
        self.model.final_norm.forward(device, queue, new_token_rows);

        let last_row = prev_position + num_new - 1;
        let last_hidden_byte_offset =
            (last_row * hidden_size * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
        let last_hidden_byte_size =
            (hidden_size * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("qwen35_session/last_hidden_readback_encoder"),
        });
        encoder.copy_buffer_to_buffer(
            &self.hidden_states_buffer,
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

        let logits = self.model.lm_head.compute(&last_hidden_state);
        let top_candidates = self.model.sampler.sample(&logits, top_k);

        self.position += num_new;
        top_candidates
    }
}
