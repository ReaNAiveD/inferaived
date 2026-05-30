use safetensors::SafeTensors;

use crate::{
    buffer_view::BufferView,
    embedding_lookup::EmbeddingLookupCpu,
    kernels::norm::RmsNormInplaceWebgpuRunner,
    layers::layer_stack::{LayerStackRunner, LayerStackSession},
    lm_head::LmHeadWebgpuRunner,
    sampling::{LogitsProcessor, SampledToken, Sampler, StoppingCriteria},
};

use super::{Qwen35ModelCore, Qwen35TextConfig};

/// CPU-backend model: shared mid-stack + CPU embed lookup. Sampler /
/// processor chain is per-request (passed to [`Qwen35CpuSession::step`]).
pub struct Qwen35CpuModel<'data> {
    pub core: Qwen35ModelCore,
    pub embed: EmbeddingLookupCpu<'data>,
}

impl<'data> Qwen35CpuModel<'data> {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tensors: &SafeTensors<'data>,
        config: &Qwen35TextConfig,
    ) -> Self {
        let embed_tokens = tensors
            .tensor("model.language_model.embed_tokens.weight")
            .expect("Failed to get tensor: model.language_model.embed_tokens.weight");
        let embed = EmbeddingLookupCpu::new(embed_tokens.clone());
        let core = Qwen35ModelCore::new(device, queue, tensors, config, embed_tokens);
        Self { core, embed }
    }
}

/// CPU-session runner: layers + final_norm + lm_head, all GPU. Embed
/// runs on CPU before the pass; sampler runs on CPU after a vocab-f32
/// readback.
struct CpuModelRunner {
    stack_runner: LayerStackRunner,
    final_norm_runner: RmsNormInplaceWebgpuRunner,
    lm_head_runner: LmHeadWebgpuRunner,
}

impl CpuModelRunner {
    fn forward(&self, cpass: &mut wgpu::ComputePass<'_>) {
        self.stack_runner.forward(cpass);
        self.final_norm_runner.forward(cpass);
        self.lm_head_runner.forward(cpass);
    }
}

/// Per-conversation mutable state for a [`Qwen35CpuModel`].
pub struct Qwen35CpuSession<'m, 'data> {
    model: &'m Qwen35CpuModel<'data>,
    layer_session: LayerStackSession<'m>,
    position_buffer: wgpu::Buffer,
    decode_hidden: wgpu::Buffer, // 1 × hidden × f32
    logits: wgpu::Buffer,        // 1 × vocab × f32 (lm_head output)
    readback: wgpu::Buffer,      // 1 × vocab × f32 (mappable)
    decode_runner: CpuModelRunner,
    position: usize,
    max_seq_len: usize,
    tokens: Vec<u32>,
}

impl<'m, 'data> Qwen35CpuSession<'m, 'data> {
    pub fn new(
        model: &'m Qwen35CpuModel<'data>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        max_seq_len: usize,
    ) -> Self {
        debug_assert!(max_seq_len >= 1, "max_seq_len must be >= 1");
        let layer_session = LayerStackSession::new(&model.core.layer_stack, device, max_seq_len);
        let position_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qwen35_cpu_session/position"),
            size: std::mem::size_of::<u32>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let hidden_size = model.core.hidden_size;
        let vocab_size = model.core.vocab_size;
        let f32_size_u64 = std::mem::size_of::<f32>() as wgpu::BufferAddress;
        let decode_hidden = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qwen35_cpu_session/decode_hidden"),
            size: hidden_size as wgpu::BufferAddress * f32_size_u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let logits_bytes = vocab_size as wgpu::BufferAddress * f32_size_u64;
        let logits = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qwen35_cpu_session/logits"),
            size: logits_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qwen35_cpu_session/readback"),
            size: logits_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let f32_size = std::mem::size_of::<f32>() as u32;
        let residual = BufferView::new_2d_tight(&decode_hidden, 1, hidden_size as u32, f32_size);
        let logits_view = BufferView::new_2d_tight(&logits, 1, vocab_size as u32, f32_size);
        let stack_runner = layer_session.plan(device, queue, residual, &position_buffer);
        let final_norm_runner = model.core.final_norm.plan(device, queue, residual);
        let lm_head_runner = model
            .core
            .lm_head
            .plan(device, queue, residual, logits_view);
        let decode_runner = CpuModelRunner {
            stack_runner,
            final_norm_runner,
            lm_head_runner,
        };

        Self {
            model,
            layer_session,
            position_buffer,
            decode_hidden,
            logits,
            readback,
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

    /// Erase per-conversation state so the session is indistinguishable
    /// from one returned by [`Self::new`].
    pub fn reset(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("qwen35_cpu_session/reset_encoder"),
        });
        self.layer_session.reset(&mut encoder);
        queue.submit(Some(encoder.finish()));
        self.position = 0;
        self.tokens.clear();
    }

    /// Append `input_ids` to the session, run the model, apply
    /// `processors` to the resulting logits, then sample one token.
    /// The returned token id is **not** appended to the session's
    /// history; the caller decides whether to feed it back via the
    /// next [`step`](Self::step) call.
    pub async fn step(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_ids: &[u32],
        processors: &[LogitsProcessor],
        sampler: &mut Sampler,
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

        let mut logits = if num_new == 1 {
            let token_embedding = self.model.embed.compute(input_ids);
            queue.write_buffer(
                &self.decode_hidden,
                0,
                bytemuck::cast_slice(&token_embedding),
            );
            self.run_and_read_back(device, queue, &self.decode_runner)
                .await
        } else {
            let runner = self.build_prefill_runner(device, queue, input_ids);
            self.run_and_read_back(device, queue, &runner).await
        };

        self.tokens.extend_from_slice(input_ids);
        self.position += num_new;
        for p in processors {
            p.process(&self.tokens, &mut logits);
        }
        sampler.sample(&self.tokens, &logits)
    }

    /// Repeatedly [`step`](Self::step) the session up to `max_tokens`
    /// times, stopping early if any element of `stopping` fires.
    ///
    /// The first iteration consumes `input_ids` (typically the prompt);
    /// subsequent iterations feed the previously-sampled token back as
    /// the sole input. Returns every sampled token in order, including
    /// the one that triggered a stopping criterion.
    ///
    /// `max_tokens` is required (not a [`StoppingCriteria`] variant) to
    /// make the loop bound an explicit argument; passing an empty
    /// `stopping` slice with a large `max_tokens` is the supported
    /// "generate exactly N tokens" idiom.
    pub async fn generate(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_ids: &[u32],
        processors: &[LogitsProcessor],
        sampler: &mut Sampler,
        max_tokens: usize,
        stopping: &[StoppingCriteria],
    ) -> Vec<SampledToken> {
        debug_assert!(max_tokens >= 1, "generate: max_tokens must be >= 1");
        let mut out = Vec::with_capacity(max_tokens);
        let mut input: Vec<u32> = input_ids.to_vec();
        for _ in 0..max_tokens {
            let tok = self.step(device, queue, &input, processors, sampler).await;
            out.push(tok);
            if stopping.iter().any(|s| s.is_done(&self.tokens, tok)) {
                break;
            }
            input.clear();
            input.push(tok.id);
        }
        out
    }

    async fn run_and_read_back(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        runner: &CpuModelRunner,
    ) -> Vec<f32> {
        let vocab_size = self.model.core.vocab_size;
        let f32_size = std::mem::size_of::<f32>() as wgpu::BufferAddress;
        let logits_bytes = vocab_size as wgpu::BufferAddress * f32_size;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("qwen35_cpu_session/step_encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("qwen35_cpu_session/step_compute_pass"),
                timestamp_writes: None,
            });
            runner.forward(&mut cpass);
        }
        encoder.copy_buffer_to_buffer(&self.logits, 0, &self.readback, 0, logits_bytes);
        let submission = queue.submit(Some(encoder.finish()));

        let slice = self.readback.slice(..);
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
        let logits = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec();
        self.readback.unmap();
        logits
    }

    fn build_prefill_runner(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_ids: &[u32],
    ) -> CpuModelRunner {
        let num_new = input_ids.len();
        let hidden_size = self.model.core.hidden_size;
        let vocab_size = self.model.core.vocab_size;
        let f32_size = std::mem::size_of::<f32>() as u32;

        // Per-prefill scratch.
        let token_embeddings = self.model.embed.compute(input_ids);
        let prefill_hidden = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qwen35_cpu_session/prefill_hidden"),
            size: (num_new * hidden_size) as wgpu::BufferAddress * f32_size as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        queue.write_buffer(&prefill_hidden, 0, bytemuck::cast_slice(&token_embeddings));

        let new_token_rows = BufferView::new_2d_tight(
            &prefill_hidden,
            num_new as u32,
            hidden_size as u32,
            f32_size,
        );
        let last_row = new_token_rows.narrow(0, num_new as u32 - 1, 1);
        let logits_view = BufferView::new_2d_tight(&self.logits, 1, vocab_size as u32, f32_size);
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
        CpuModelRunner {
            stack_runner,
            final_norm_runner,
            lm_head_runner,
        }
    }
}
