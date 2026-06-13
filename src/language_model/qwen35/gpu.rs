use async_stream::stream;
use futures_core::Stream;
use safetensors::SafeTensors;

use crate::{
    buffer_view::BufferView,
    embedding_lookup::{EmbeddingLookupWebgpu, EmbeddingLookupWebgpuRunner},
    gpu_sampler::{GpuSampler, GpuSamplerRunner},
    kernels::norm::RmsNormInplaceWebgpuRunner,
    layers::qwen35_layer_stack::{LayerStackRunner, LayerStackSession},
    lm_head::LmHeadWebgpuRunner,
    sampling::{SampledToken, SamplingParams, StoppingCriteria},
};

use super::{Qwen35ModelCore, Qwen35TextConfig};

/// GPU-backend model: shared mid-stack + GPU embed lookup + GPU sampler
/// kernel. Per-request configuration is `SamplingParams`.
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
        config: &Qwen35TextConfig,
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

/// Persistent GPU resources a [`Qwen35GpuSession`] runs on top of.
struct Qwen35GpuWorkspace<'m> {
    model: &'m Qwen35GpuModel,
    layer_session: LayerStackSession<'m>,
    /// 1 × u32, RoPE base position for the current dispatch.
    position_buffer: wgpu::Buffer,
    /// 1 × vocab × f32. Sampler kernel reads this (no readback).
    logits: wgpu::Buffer,
}

impl<'m> Qwen35GpuWorkspace<'m> {
    fn new(model: &'m Qwen35GpuModel, device: &wgpu::Device, max_seq_len: usize) -> Self {
        let layer_session = LayerStackSession::new(&model.core.layer_stack, device, max_seq_len);
        let position_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qwen35_gpu_session/position"),
            size: std::mem::size_of::<u32>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let f32_size_u64 = std::mem::size_of::<f32>() as wgpu::BufferAddress;
        let vocab_size = model.core.vocab_size;
        let logits = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qwen35_gpu_session/logits"),
            size: vocab_size as wgpu::BufferAddress * f32_size_u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self {
            model,
            layer_session,
            position_buffer,
            logits,
        }
    }

    /// Plan a sampling forward through the persistent context buffers, reading
    /// `num_new` tokens from `input_token` into `input_hidden`, with the GPU-
    /// sampled id written to `sampled_token`.
    fn plan(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_token: &wgpu::Buffer,
        input_hidden: &wgpu::Buffer,
        num_new: u32,
        sampled_token: &wgpu::Buffer,
    ) -> GpuModelRunner {
        debug_assert!(num_new >= 1, "plan requires num_new >= 1");
        let hidden_size = self.model.core.hidden_size as u32;
        let vocab_size = self.model.core.vocab_size as u32;
        let u32_size = std::mem::size_of::<u32>() as u32;
        let f32_size = std::mem::size_of::<f32>() as u32;

        let input_token_view = BufferView::new_1d(input_token, u32_size, num_new);
        let input_hidden_view =
            BufferView::new_2d_tight(input_hidden, num_new, hidden_size, f32_size);
        let last_row = input_hidden_view.narrow(0, num_new - 1, 1);
        let logits_view = BufferView::new_2d_tight(&self.logits, 1, vocab_size, f32_size);
        let logits_1d = BufferView::new_1d(&self.logits, f32_size, vocab_size);
        let sampled_token_view = BufferView::new_1d(sampled_token, u32_size, 1);

        let embed_runner =
            self.model
                .embed
                .plan(device, queue, input_token_view, input_hidden_view);
        let stack_runner =
            self.layer_session
                .plan(device, queue, input_hidden_view, &self.position_buffer);
        let final_norm_runner = self
            .model
            .core
            .final_norm
            .plan(device, queue, input_hidden_view);
        let lm_head_runner = self
            .model
            .core
            .lm_head
            .plan(device, queue, last_row, logits_view);
        let sampler_runner = self
            .model
            .sampler
            .plan(device, queue, logits_1d, sampled_token_view);

        GpuModelRunner {
            embed_runner,
            stack_runner,
            final_norm_runner,
            lm_head_runner,
            sampler_runner,
        }
    }

    /// Encode a layer-cache reset into `encoder`. Called by
    /// [`Qwen35GpuSession::reset`].
    fn reset_layers(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.layer_session.reset(encoder);
    }

    fn plan_prefill(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_ids: &[u32],
        sampled_token: &wgpu::Buffer,
    ) -> GpuModelRunner {
        let u32_size = std::mem::size_of::<u32>() as wgpu::BufferAddress;
        let f32_size = std::mem::size_of::<f32>() as wgpu::BufferAddress;
        let hidden_size = self.model.core.hidden_size;
        let prefill_tokens = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qwen35_gpu_session/prefill_tokens"),
            size: input_ids.len() as wgpu::BufferAddress * u32_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&prefill_tokens, 0, bytemuck::cast_slice(input_ids));
        let prefill_hidden = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qwen35_gpu_session/prefill_hidden"),
            size: (input_ids.len() * hidden_size) as wgpu::BufferAddress * f32_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.plan(
            device,
            queue,
            &prefill_tokens,
            &prefill_hidden,
            input_ids.len() as u32,
            sampled_token,
        )
    }
}

/// Per-conversation streaming generator over a [`Qwen35GpuWorkspace`].
pub struct Qwen35GpuSession<'m> {
    workspace: Qwen35GpuWorkspace<'m>,
    /// 1 × u32. Sampler writes; decode embed reads on the next step.
    current_token: wgpu::Buffer,
    /// 1 × u32, mappable. Copied from `current_token` each step.
    token_readback: wgpu::Buffer,
    decode_runner: GpuModelRunner,
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
        let workspace = Qwen35GpuWorkspace::new(model, device, max_seq_len);
        let u32_size = std::mem::size_of::<u32>() as wgpu::BufferAddress;
        let f32_size = std::mem::size_of::<f32>() as wgpu::BufferAddress;
        let current_token = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qwen35_gpu_session/current_token"),
            size: u32_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Single-token decode scratch: kept alive by `decode_runner`'s bind
        // groups (wgpu buffers are Arc-backed internally), so no field needed.
        let decode_hidden = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qwen35_gpu_session/decode_hidden"),
            size: model.core.hidden_size as wgpu::BufferAddress * f32_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let decode_runner = workspace.plan(
            device,
            queue,
            &current_token,
            &decode_hidden,
            1,
            &current_token,
        );
        let token_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("qwen35_gpu_session/token_readback"),
            size: u32_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            workspace,
            current_token,
            token_readback,
            decode_runner,
            max_seq_len,
            tokens: Vec::with_capacity(max_seq_len),
        }
    }

    pub fn position(&self) -> usize {
        self.tokens.len()
    }
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Erase per-conversation state so the session is indistinguishable
    /// from one returned by [`Self::new`].
    pub fn reset(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("qwen35_gpu_session/reset_encoder"),
        });
        self.workspace.reset_layers(&mut encoder);
        queue.submit(Some(encoder.finish()));
        self.tokens.clear();
    }

    /// Append `input_ids` to the session, dispatch the GPU pipeline,
    /// and return the token id the sampler picked from the last
    /// position's logits. `input_ids` is always honored verbatim.
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
            self.tokens.len() + num_new <= self.max_seq_len,
            "session overflow: position {} + {} new tokens exceeds max_seq_len {}",
            self.tokens.len(),
            num_new,
            self.max_seq_len,
        );
        let prev_position = self.tokens.len();
        queue.write_buffer(
            &self.workspace.position_buffer,
            0,
            bytemuck::bytes_of(&(prev_position as u32)),
        );

        let token_id = if num_new == 1 {
            queue.write_buffer(&self.current_token, 0, bytemuck::bytes_of(&input_ids[0]));
            self.run_and_read_back_token(device, queue, &self.decode_runner)
                .await
        } else {
            let runner = self
                .workspace
                .plan_prefill(device, queue, input_ids, &self.current_token);
            self.run_and_read_back_token(device, queue, &runner).await
        };

        self.tokens.extend_from_slice(input_ids);
        SampledToken {
            id: token_id,
            logprob: f32::NAN,
        }
    }

    /// Encode `input_ids` and continue sampling up to `max_tokens`
    /// tokens (inclusive of the first), stopping early when any
    /// element of `stopping` fires.
    ///
    /// Returns a lazy `impl Stream`; tokens are produced as the
    /// caller polls. Dropping the stream (or breaking the consumer
    /// loop) aborts generation without launching further GPU work.
    pub fn generate<'a>(
        &'a mut self,
        device: &'a wgpu::Device,
        queue: &'a wgpu::Queue,
        input_ids: &'a [u32],
        params: &SamplingParams,
        max_tokens: usize,
        stopping: &'a [StoppingCriteria],
    ) -> impl Stream<Item = SampledToken> + Send + 'a {
        debug_assert!(max_tokens >= 1, "generate: max_tokens must be >= 1");
        debug_assert!(
            !input_ids.is_empty(),
            "generate: input_ids must be non-empty",
        );
        let params = *params;
        stream! {
            let mut tok = self.step(device, queue, input_ids, &params).await;
            yield tok;
            if stopping.iter().any(|s| s.is_done(&self.tokens, tok)) {
                return;
            }
            for _ in 1..max_tokens {
                tok = self.decode_next(device, queue, tok.id, &params).await;
                yield tok;
                if stopping.iter().any(|s| s.is_done(&self.tokens, tok)) {
                    break;
                }
            }
        }
    }

    /// In-place decode step. Assumes `current_token` already holds
    /// `prev_sampled_id`. Private because the precondition is invisible
    /// at the API boundary; only [`generate`](Self::generate) can satisfy it.
    async fn decode_next(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        prev_sampled_id: u32,
        _params: &SamplingParams,
    ) -> SampledToken {
        let prev_position = self.tokens.len();
        debug_assert!(
            prev_position + 1 <= self.max_seq_len,
            "session overflow: position {} + 1 exceeds max_seq_len {}",
            prev_position,
            self.max_seq_len,
        );
        queue.write_buffer(
            &self.workspace.position_buffer,
            0,
            bytemuck::bytes_of(&(prev_position as u32)),
        );
        let token_id = self
            .run_and_read_back_token(device, queue, &self.decode_runner)
            .await;

        self.tokens.push(prev_sampled_id);
        SampledToken {
            id: token_id,
            logprob: f32::NAN,
        }
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
}
