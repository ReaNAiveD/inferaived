use safetensors::SafeTensors;

use crate::{
    buffer_view::BufferView,
    embedding_lookup::{EmbeddingLookupWebgpu, EmbeddingLookupWebgpuRunner},
    gpu_sampler::{GpuSampler, GpuSamplerRunner},
    kernels::norm::RmsNormInplaceWebgpuRunner,
    layers::layer_stack::{LayerStackRunner, LayerStackSession},
    lm_head::LmHeadWebgpuRunner,
    sampling::{SampledToken, SamplingParams, StoppingCriteria},
};

use super::{Qwen35Config, Qwen35ModelCore};

/// GPU-backend model: shared mid-stack + GPU embed lookup + GPU sampler
/// kernel. Per-request configuration is just `SamplingParams`.
pub struct Qwen35GpuModel {
    pub core: Qwen35ModelCore,
    pub embed: EmbeddingLookupWebgpu,
    pub sampler: GpuSampler,
}

impl Qwen35GpuModel {
    pub fn new<'data>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tensors: &SafeTensors<'data>,
        config: &Qwen35Config,
    ) -> Self {
        let embed_tokens = tensors
            .tensor("model.language_model.embed_tokens.weight")
            .expect("Failed to get tensor: model.language_model.embed_tokens.weight");
        let embed = EmbeddingLookupWebgpu::new(device, queue, embed_tokens.clone());
        let core = Qwen35ModelCore::new(device, queue, tensors, config, embed_tokens);
        let sampler = GpuSampler::new(device);
        Self {
            core,
            embed,
            sampler,
        }
    }
}

/// GPU-session runner: embed + layers + final_norm + lm_head + sampler
/// chained into one compute pass.
struct GpuModelRunner {
    embed_runner: EmbeddingLookupWebgpuRunner,
    stack_runner: LayerStackRunner,
    final_norm_runner: RmsNormInplaceWebgpuRunner,
    lm_head_runner: LmHeadWebgpuRunner,
    sampler_runner: GpuSamplerRunner,
}

impl GpuModelRunner {
    fn forward(&self, cpass: &mut wgpu::ComputePass<'_>) {
        self.embed_runner.forward(cpass);
        self.stack_runner.forward(cpass);
        self.final_norm_runner.forward(cpass);
        self.lm_head_runner.forward(cpass);
        self.sampler_runner.forward(cpass);
    }
}

/// Per-conversation mutable state for a [`Qwen35GpuModel`].
pub struct Qwen35GpuSession<'m> {
    model: &'m Qwen35GpuModel,
    layer_session: LayerStackSession<'m>,
    position_buffer: wgpu::Buffer,
    /// 1 × u32. Sampler writes; decode embed reads. Persistent across
    /// the entire conversation.
    current_token: wgpu::Buffer,
    /// max_seq_len × u32. Caller writes prompt tokens here at
    /// `prev_position` byte offset per prefill call.
    prefill_tokens: wgpu::Buffer,
    /// 1 × u32, mappable. Copied from `current_token` each step.
    token_readback: wgpu::Buffer,
    /// max_seq_len × hidden × f32; written by the GPU embed kernel.
    prefill_hidden: wgpu::Buffer,
    /// 1 × hidden × f32. Owned to keep the underlying GPU buffer alive
    /// for the bind group that references it inside `decode_runner`.
    #[allow(dead_code)]
    decode_hidden: wgpu::Buffer,
    /// 1 × vocab × f32. Sampler kernel reads this (no readback).
    logits: wgpu::Buffer,
    decode_runner: GpuModelRunner,
    position: usize,
    max_seq_len: usize,
    tokens: Vec<u32>,
}

impl<'m> Qwen35GpuSession<'m> {
    pub fn new(
        model: &'m Qwen35GpuModel,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        max_seq_len: usize,
    ) -> Self {
        debug_assert!(max_seq_len >= 1, "max_seq_len must be >= 1");
        let layer_session = LayerStackSession::new(&model.core.layer_stack, device, max_seq_len);
        let position_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qwen35_gpu_session/position"),
            size: std::mem::size_of::<u32>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let hidden_size = model.core.hidden_size;
        let vocab_size = model.core.vocab_size;
        let f32_size_u64 = std::mem::size_of::<f32>() as wgpu::BufferAddress;
        let u32_size_u64 = std::mem::size_of::<u32>() as wgpu::BufferAddress;

        let current_token = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qwen35_gpu_session/current_token"),
            size: u32_size_u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let prefill_tokens = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qwen35_gpu_session/prefill_tokens"),
            size: max_seq_len as wgpu::BufferAddress * u32_size_u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let token_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qwen35_gpu_session/token_readback"),
            size: u32_size_u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let prefill_hidden = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qwen35_gpu_session/prefill_hidden"),
            size: (max_seq_len * hidden_size) as wgpu::BufferAddress * f32_size_u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let decode_hidden = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qwen35_gpu_session/decode_hidden"),
            size: hidden_size as wgpu::BufferAddress * f32_size_u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let logits = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qwen35_gpu_session/logits"),
            size: vocab_size as wgpu::BufferAddress * f32_size_u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let u32_size = std::mem::size_of::<u32>() as u32;
        let f32_size = std::mem::size_of::<f32>() as u32;
        // Decode-time runner: embed reads `current_token`, writes `decode_hidden`;
        // sampler reads `logits`, writes `current_token`. Read-after-write on
        // `current_token` across dispatches in one encoder is fine under wgpu's
        // per-buffer usage tracker (same STORAGE flags, separated by dispatch
        // barriers).
        let current_token_view = BufferView::new_1d(&current_token, u32_size, 1);
        let decode_hidden_view =
            BufferView::new_2d_tight(&decode_hidden, 1, hidden_size as u32, f32_size);
        let logits_view = BufferView::new_2d_tight(&logits, 1, vocab_size as u32, f32_size);
        let embed_runner = model
            .embed
            .plan(device, queue, current_token_view, decode_hidden_view);
        let stack_runner = layer_session.plan(device, queue, decode_hidden_view, &position_buffer);
        let final_norm_runner = model
            .core
            .final_norm
            .plan(device, queue, decode_hidden_view);
        let lm_head_runner =
            model
                .core
                .lm_head
                .plan(device, queue, decode_hidden_view, logits_view);
        let logits_1d = BufferView::new_1d(&logits, f32_size, vocab_size as u32);
        let sampler_runner = model
            .sampler
            .plan(device, queue, logits_1d, current_token_view);
        let decode_runner = GpuModelRunner {
            embed_runner,
            stack_runner,
            final_norm_runner,
            lm_head_runner,
            sampler_runner,
        };

        Self {
            model,
            layer_session,
            position_buffer,
            current_token,
            prefill_tokens,
            token_readback,
            prefill_hidden,
            decode_hidden,
            logits,
            decode_runner,
            position: 0,
            max_seq_len,
            tokens: Vec::with_capacity(max_seq_len),
        }
    }

    pub fn position(&self) -> usize {
        self.position
    }
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Append `input_ids` to the session, dispatch the GPU pipeline,
    /// and return the token id the GPU sampler wrote to `current_token`.
    ///
    /// `_params` is reserved for the temperature / top-k / top-p stages
    /// landing in migration step 7; the current greedy kernel ignores it.
    pub async fn step(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_ids: &[u32],
        _params: &SamplingParams,
    ) -> SampledToken {
        let num_new = input_ids.len();
        debug_assert!(num_new >= 1, "step requires non-empty input_ids");
        debug_assert!(
            self.position + num_new <= self.max_seq_len,
            "session overflow: position {} + {} new tokens exceeds max_seq_len {}",
            self.position,
            num_new,
            self.max_seq_len,
        );
        let prev_position = self.position;
        queue.write_buffer(
            &self.position_buffer,
            0,
            bytemuck::bytes_of(&(prev_position as u32)),
        );

        let token_id = if num_new == 1 {
            // Decode: write the new token into `current_token`. The previous
            // step's sampler may have already written the same value, but
            // honoring `input_ids` lets the caller force an arbitrary token
            // (e.g. speculative-decoding rejection path).
            queue.write_buffer(&self.current_token, 0, bytemuck::bytes_of(&input_ids[0]));
            self.run_and_read_back_token(device, queue, &self.decode_runner)
                .await
        } else {
            // Prefill: write input_ids into `prefill_tokens` at the right
            // offset, build a one-shot runner that embeds num_new tokens.
            let u32_size = std::mem::size_of::<u32>() as wgpu::BufferAddress;
            let dst_byte_offset = prev_position as wgpu::BufferAddress * u32_size;
            queue.write_buffer(
                &self.prefill_tokens,
                dst_byte_offset,
                bytemuck::cast_slice(input_ids),
            );
            let runner = self.build_prefill_runner(device, queue, prev_position, num_new);
            self.run_and_read_back_token(device, queue, &runner).await
        };

        self.tokens.extend_from_slice(input_ids);
        self.position += num_new;
        SampledToken {
            id: token_id,
            // GPU sampler doesn't compute logprob yet (step 7 may add it).
            // Consumers that need it should use Qwen35CpuSession.
            logprob: f32::NAN,
        }
    }

    /// Repeatedly [`step`](Self::step) the session up to `max_tokens`
    /// times, stopping early if any element of `stopping` fires. See
    /// [`crate::language_model::Qwen35CpuSession::generate`] for the
    /// shared rationale.
    pub async fn generate(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_ids: &[u32],
        params: &SamplingParams,
        max_tokens: usize,
        stopping: &[StoppingCriteria],
    ) -> Vec<SampledToken> {
        debug_assert!(max_tokens >= 1, "generate: max_tokens must be >= 1");
        let mut out = Vec::with_capacity(max_tokens);
        let mut input: Vec<u32> = input_ids.to_vec();
        for _ in 0..max_tokens {
            let tok = self.step(device, queue, &input, params).await;
            out.push(tok);
            if stopping.iter().any(|s| s.is_done(&self.tokens, tok)) {
                break;
            }
            input.clear();
            input.push(tok.id);
        }
        out
    }

    async fn run_and_read_back_token(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        runner: &GpuModelRunner,
    ) -> u32 {
        let u32_size = std::mem::size_of::<u32>() as wgpu::BufferAddress;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("qwen35_gpu_session/step_encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("qwen35_gpu_session/step_compute_pass"),
                timestamp_writes: None,
            });
            runner.forward(&mut cpass);
        }
        encoder.copy_buffer_to_buffer(&self.current_token, 0, &self.token_readback, 0, u32_size);
        let submission = queue.submit(Some(encoder.finish()));

        let slice = self.token_readback.slice(..);
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
        let bytes = slice.get_mapped_range();
        let token = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        drop(bytes);
        self.token_readback.unmap();
        token
    }

    fn build_prefill_runner(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        prev_position: usize,
        num_new: usize,
    ) -> GpuModelRunner {
        let hidden_size = self.model.core.hidden_size;
        let vocab_size = self.model.core.vocab_size;
        let f32_size = std::mem::size_of::<f32>() as u32;
        let u32_size = std::mem::size_of::<u32>() as u32;

        // Token slice: prefill_tokens[prev_position .. prev_position+num_new]
        let new_tokens = BufferView::new_1d(
            &self.prefill_tokens,
            u32_size,
            self.max_seq_len as u32,
        )
        .narrow(0, prev_position as u32, num_new as u32);
        // Hidden slice: prefill_hidden[prev_position .. prev_position+num_new, :]
        let new_token_rows = BufferView::new_2d_tight(
            &self.prefill_hidden,
            self.max_seq_len as u32,
            hidden_size as u32,
            f32_size,
        )
        .narrow(0, prev_position as u32, num_new as u32);
        let last_row = new_token_rows.narrow(0, num_new as u32 - 1, 1);
        let logits_view = BufferView::new_2d_tight(&self.logits, 1, vocab_size as u32, f32_size);
        let logits_1d = BufferView::new_1d(&self.logits, f32_size, vocab_size as u32);
        let current_token_view = BufferView::new_1d(&self.current_token, u32_size, 1);

        let embed_runner = self
            .model
            .embed
            .plan(device, queue, new_tokens, new_token_rows);
        let stack_runner =
            self.layer_session
                .plan(device, queue, new_token_rows, &self.position_buffer);
        let final_norm_runner = self
            .model
            .core
            .final_norm
            .plan(device, queue, new_token_rows);
        let lm_head_runner = self
            .model
            .core
            .lm_head
            .plan(device, queue, last_row, logits_view);
        let sampler_runner = self
            .model
            .sampler
            .plan(device, queue, logits_1d, current_token_view);
        GpuModelRunner {
            embed_runner,
            stack_runner,
            final_norm_runner,
            lm_head_runner,
            sampler_runner,
        }
    }
}
