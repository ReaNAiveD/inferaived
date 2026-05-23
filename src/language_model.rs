use safetensors::SafeTensors;

use crate::{
    buffer_view::BufferView,
    embedding_lookup::EmbeddingLookupCpu,
    kernels::norm::{RmsNormInplaceWebgpu, RmsNormInplaceWebgpuRunner},
    layers::{
        layer_stack::{
            LayerConfig, LayerStack, LayerStackConfig, LayerStackRunner, LayerStackSession,
        },
        linear_attention::LinearAttentionConfig,
        self_attention::SelfAttentionConfig,
    },
    lm_head::LmHeadCpu,
    log_tensor,
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

/// A pre-baked bundle of bind groups + pipelines for one full
/// model-level forward pass.
pub struct Qwen35ModelRunner {
    stack_runner: LayerStackRunner,
    final_norm_runner: RmsNormInplaceWebgpuRunner,
}

impl Qwen35ModelRunner {
    pub fn forward(&self, cpass: &mut wgpu::ComputePass<'_>) {
        self.stack_runner.forward(cpass);
        self.final_norm_runner.forward(cpass);
    }
}

/// GPU buffers shared by the prefill and decode paths of a session.
struct Workspace {
    prefill_hidden: wgpu::Buffer,
    position: wgpu::Buffer,
    readback: wgpu::Buffer,
}

impl Workspace {
    fn new(device: &wgpu::Device, hidden_size: usize, max_seq_len: usize) -> Self {
        let prefill_hidden = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qwen35_session/workspace/prefill_hidden"),
            size: (max_seq_len * hidden_size * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let position = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qwen35_session/workspace/position"),
            size: std::mem::size_of::<u32>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qwen35_session/workspace/readback"),
            size: (hidden_size * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            prefill_hidden,
            position,
            readback,
        }
    }
}

/// Decode-path state: a fixed-address single-row residual buffer plus a runner
/// whose bind groups reference it and the session's `position` uniform.
struct DecodeRig {
    hidden: wgpu::Buffer,
    runner: Qwen35ModelRunner,
}

impl DecodeRig {
    fn build<'m, 'data>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &'m Qwen35Model<'data>,
        layer_session: &LayerStackSession<'m>,
        position_buffer: &wgpu::Buffer,
    ) -> Self {
        let hidden_size = model.hidden_size;
        let hidden = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qwen35_session/decode/hidden"),
            size: (hidden_size * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let residual_slot = BufferView::new_2d_tight(
            &hidden,
            1,
            hidden_size as u32,
            std::mem::size_of::<f32>() as u32,
        );
        let stack_runner = layer_session.plan(device, queue, residual_slot, position_buffer);
        let final_norm_runner = model.final_norm.plan(device, queue, residual_slot);
        Self {
            hidden,
            runner: Qwen35ModelRunner {
                stack_runner,
                final_norm_runner,
            },
        }
    }
}

/// Per-conversation mutable state for a [`Qwen35Model`].
pub struct Qwen35Session<'m, 'data> {
    model: &'m Qwen35Model<'data>,
    layer_session: LayerStackSession<'m>,
    workspace: Workspace,
    decode: DecodeRig,
    position: usize,
    max_seq_len: usize,
}

impl<'m, 'data> Qwen35Session<'m, 'data> {
    pub fn new(
        model: &'m Qwen35Model<'data>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        max_seq_len: usize,
    ) -> Self {
        debug_assert!(max_seq_len >= 1, "max_seq_len must be >= 1");
        let layer_session = LayerStackSession::new(&model.layer_stack, device, max_seq_len);
        let workspace = Workspace::new(device, model.hidden_size, max_seq_len);
        let decode = DecodeRig::build(device, queue, model, &layer_session, &workspace.position);
        Self {
            model,
            layer_session,
            workspace,
            decode,
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

    /// Append `input_ids` to the session and run the model over them,
    /// returning the last-token hidden state.
    pub async fn advance(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_ids: &[u32],
    ) -> Vec<f32> {
        let num_new = input_ids.len();
        debug_assert!(num_new >= 1, "advance requires non-empty input_ids");
        debug_assert!(
            self.position + num_new <= self.max_seq_len,
            "session overflow: position {} + {} new tokens exceeds max_seq_len {}",
            self.position,
            num_new,
            self.max_seq_len,
        );
        let prev_position = self.position;
        let token_embeddings = self.model.embedding_lookup.compute(input_ids);

        // Push the absolute position uniform once. Every runner's bind
        // groups already reference `self.workspace.position`; kernels
        // pick up the live value at dispatch time.
        let pos = prev_position as u32;
        queue.write_buffer(&self.workspace.position, 0, bytemuck::bytes_of(&pos));

        let last_hidden = if num_new == 1 {
            self.decode_step(device, queue, &token_embeddings).await
        } else {
            self.prefill_step(device, queue, prev_position, num_new, &token_embeddings)
                .await
        };

        self.position += num_new;
        last_hidden
    }

    /// Convenience: [`advance`](Self::advance) + LM head + top-k sampler.
    pub async fn forward(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_ids: &[u32],
        top_k: usize,
    ) -> Vec<(usize, f32)> {
        let last_hidden = self.advance(device, queue, input_ids).await;
        let logits = self.model.lm_head.compute(&last_hidden);
        self.model.sampler.sample(&logits, top_k)
    }

    /// One decode step. Reuses the eagerly-built `self.decode.runner`.
    async fn decode_step(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        token_embedding: &[f32],
    ) -> Vec<f32> {
        queue.write_buffer(
            &self.decode.hidden,
            0,
            bytemuck::cast_slice(token_embedding),
        );
        self.run_and_read_back(device, queue, &self.decode.runner, &self.decode.hidden, 0)
            .await
    }

    /// One prefill step. Builds a one-shot runner sized to the slot
    /// `prev_position..prev_position+num_new` (see
    /// [`build_prefill_runner`](Self::build_prefill_runner) for why
    /// prefill runners cannot be cached).
    async fn prefill_step(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        prev_position: usize,
        num_new: usize,
        token_embeddings: &[f32],
    ) -> Vec<f32> {
        let hidden_size = self.model.hidden_size;
        let f32_size = std::mem::size_of::<f32>() as wgpu::BufferAddress;
        let dst_byte_offset = (prev_position * hidden_size) as wgpu::BufferAddress * f32_size;
        queue.write_buffer(
            &self.workspace.prefill_hidden,
            dst_byte_offset,
            bytemuck::cast_slice(token_embeddings),
        );
        let runner = self.build_prefill_runner(device, queue, prev_position, num_new);
        let last_row = prev_position + num_new - 1;
        let src_byte_offset = (last_row * hidden_size) as wgpu::BufferAddress * f32_size;
        self.run_and_read_back(
            device,
            queue,
            &runner,
            &self.workspace.prefill_hidden,
            src_byte_offset,
        )
        .await
    }

    /// The session's only encoder / submit / map call site.
    ///
    /// Splitting submit from await here is the natural future-work hook
    /// for overlapped readback (sample token N while GPU computes step
    /// N+1).
    async fn run_and_read_back(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        runner: &Qwen35ModelRunner,
        src_buffer: &wgpu::Buffer,
        src_byte_offset: wgpu::BufferAddress,
    ) -> Vec<f32> {
        let hidden_size = self.model.hidden_size;
        let f32_size = std::mem::size_of::<f32>() as wgpu::BufferAddress;
        let row_bytes = hidden_size as wgpu::BufferAddress * f32_size;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("qwen35_session/step_encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("qwen35_session/step_compute_pass"),
                timestamp_writes: None,
            });
            runner.forward(&mut cpass);
        }
        encoder.copy_buffer_to_buffer(
            src_buffer,
            src_byte_offset,
            &self.workspace.readback,
            0,
            row_bytes,
        );
        let submission = queue.submit(Some(encoder.finish()));

        let slice = self.workspace.readback.slice(..);
        let (tx, rx) = tokio::sync::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        });
        rx.await
            .expect("Failed to map buffer")
            .expect("Failed to map buffer");
        let last_hidden = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec();
        self.workspace.readback.unmap();
        last_hidden
    }

    /// Build a one-shot runner over the prefill slot.
    fn build_prefill_runner(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        prev_position: usize,
        num_new: usize,
    ) -> Qwen35ModelRunner {
        let hidden_size = self.model.hidden_size;
        let new_token_rows = BufferView::new_2d_tight(
            &self.workspace.prefill_hidden,
            self.max_seq_len as u32,
            hidden_size as u32,
            std::mem::size_of::<f32>() as u32,
        )
        .narrow(0, prev_position as u32, num_new as u32);
        let stack_runner =
            self.layer_session
                .plan(device, queue, new_token_rows, &self.workspace.position);
        let final_norm_runner = self.model.final_norm.plan(device, queue, new_token_rows);
        Qwen35ModelRunner {
            stack_runner,
            final_norm_runner,
        }
    }
}
