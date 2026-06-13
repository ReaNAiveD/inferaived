use async_stream::stream;
use futures_core::Stream;
use safetensors::SafeTensors;

use crate::{
    buffer_view::{BufferView, MAX_DIMS},
    embedding_lookup::{EmbeddingLookupWebgpu, EmbeddingLookupWebgpuRunner},
    gpu_sampler::{GpuSampler, GpuSamplerRunner},
    kernels::{
        elementwise_add::{ElementwiseAddWebgpu, ElementwiseAddWebgpuRunner},
        norm::RmsNormWebgpuRunner,
        scalar_mul::{ScalarMulInplaceWebgpu, ScalarMulInplaceWebgpuRunner},
    },
    layers::hrm_layer_stack::{HrmLayerStackRunner, HrmLayerStackSession},
    lm_head::LmHeadWebgpuRunner,
    log_tensor,
    sampling::{SampledToken, SamplingParams, StoppingCriteria},
};

use super::{HrmTextConfig, HrmTextModelCore};

/// GPU-backend model for HRM-Text: the shared recurrent core + GPU embed
/// lookup + embedding-scale + injection-add + GPU argmax sampler.
pub struct HrmTextGpuModel {
    pub core: HrmTextModelCore,
    pub embed: EmbeddingLookupWebgpu,
    pub embed_scale: ScalarMulInplaceWebgpu,
    pub inject_add: ElementwiseAddWebgpu,
    pub sampler: GpuSampler,
}

impl HrmTextGpuModel {
    pub fn new<'data>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tensors: &SafeTensors<'data>,
        config: &HrmTextConfig,
    ) -> Self {
        let embed_name = "model.embed_tokens.weight";
        let embed_tokens = tensors
            .tensor(embed_name)
            .expect(&format!("Failed to get tensor: {}", embed_name));
        log_tensor(embed_name, &embed_tokens);
        let embed = EmbeddingLookupWebgpu::new(device, queue, embed_tokens);
        let core = HrmTextModelCore::new(device, queue, tensors, config);
        let embed_scale =
            ScalarMulInplaceWebgpu::new(device, core.hidden_size, core.embedding_scale);
        let inject_add = ElementwiseAddWebgpu::new(device, core.hidden_size);
        let sampler = GpuSampler::new(device);
        Self {
            core,
            embed,
            embed_scale,
            inject_add,
            sampler,
        }
    }

    pub fn new_session<'m>(
        &'m self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        max_seq_len: usize,
    ) -> HrmTextGpuSession<'m> {
        HrmTextGpuSession::new(self, device, queue, max_seq_len)
    }
}

/// One stack invocation in the recurrent schedule: inject the slow/fast state
/// into the working buffer, run the 16-layer stack, then apply the
/// module-final parameterless RMSNorm.
struct CycleRunner {
    inject_runner: ElementwiseAddWebgpuRunner,
    stack_runner: HrmLayerStackRunner,
    norm_runner: RmsNormWebgpuRunner,
}

impl CycleRunner {
    fn forward(&self, cpass: &mut wgpu::ComputePass<'_>) {
        self.inject_runner.forward(cpass);
        self.stack_runner.forward(cpass);
        self.norm_runner.forward(cpass);
    }
}

/// Full recurrent forward chained into one compute pass: embed → scale →
/// `H_cycles × (L_cycles + 1)` stack invocations → LM head → sampler.
struct GpuModelRunner {
    embed_runner: EmbeddingLookupWebgpuRunner,
    embed_scale_runner: ScalarMulInplaceWebgpuRunner,
    cycle_runners: Vec<CycleRunner>,
    lm_head_runner: LmHeadWebgpuRunner,
    sampler_runner: GpuSamplerRunner,
}

impl GpuModelRunner {
    fn forward(&self, cpass: &mut wgpu::ComputePass<'_>) {
        self.embed_runner.forward(cpass);
        self.embed_scale_runner.forward(cpass);
        for cycle in &self.cycle_runners {
            cycle.forward(cpass);
        }
        self.lm_head_runner.forward(cpass);
        self.sampler_runner.forward(cpass);
    }
}

/// Persistent GPU resources a [`HrmTextGpuSession`] runs on top of: one KV-cache
/// stack-session per recurrent cycle (`H_cycles` over the H stack,
/// `H_cycles × L_cycles` over the L stack), plus the RoPE position scalar and
/// the logits buffer the sampler reads.
struct HrmTextGpuWorkspace<'m> {
    model: &'m HrmTextGpuModel,
    h_sessions: Vec<HrmLayerStackSession<'m>>,
    l_sessions: Vec<HrmLayerStackSession<'m>>,
    position_buffer: wgpu::Buffer,
    logits: wgpu::Buffer,
}

impl<'m> HrmTextGpuWorkspace<'m> {
    fn new(model: &'m HrmTextGpuModel, device: &wgpu::Device, max_seq_len: usize) -> Self {
        let h_cycles = model.core.h_cycles;
        let l_cycles = model.core.l_cycles;
        let h_sessions = (0..h_cycles)
            .map(|_| HrmLayerStackSession::new(&model.core.h_stack, device, max_seq_len))
            .collect();
        let l_sessions = (0..h_cycles * l_cycles)
            .map(|_| HrmLayerStackSession::new(&model.core.l_stack, device, max_seq_len))
            .collect();
        let position_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hrm_text_gpu_session/position"),
            size: std::mem::size_of::<u32>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let f32_size = std::mem::size_of::<f32>() as wgpu::BufferAddress;
        let logits = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hrm_text_gpu_session/logits"),
            size: model.core.vocab_size as wgpu::BufferAddress * f32_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        Self {
            model,
            h_sessions,
            l_sessions,
            position_buffer,
            logits,
        }
    }

    /// Plan one recurrent forward over the supplied working buffers.
    /// `z_h_buf`, `z_l_buf`, `work_buf` are each `[num_new, hidden]` f32 scratch
    /// owned by the caller; the GPU-sampled id is written to `sampled_token`.
    fn plan(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_token: &wgpu::Buffer,
        z_h_buf: &wgpu::Buffer,
        z_l_buf: &wgpu::Buffer,
        work_buf: &wgpu::Buffer,
        num_new: u32,
        sampled_token: &wgpu::Buffer,
    ) -> GpuModelRunner {
        debug_assert!(num_new >= 1, "plan requires num_new >= 1");
        let core = &self.model.core;
        let hidden = core.hidden_size as u32;
        let vocab = core.vocab_size as u32;
        let u32_size = std::mem::size_of::<u32>() as u32;
        let f32_size = std::mem::size_of::<f32>() as u32;

        let input_token_view = BufferView::new_1d(input_token, u32_size, num_new);
        let z_h = BufferView::new_2d_tight(z_h_buf, num_new, hidden, f32_size);
        let z_l = BufferView::new_2d_tight(z_l_buf, num_new, hidden, f32_size);
        let work = BufferView::new_2d_tight(work_buf, num_new, hidden, f32_size);

        // `z_L_init` is `[hidden]`; broadcast it across all `num_new` rows by
        // pinning the outer stride to zero (every token reads the same seed).
        let mut zli_shape = [1u32; MAX_DIMS];
        let mut zli_stride = [0u32; MAX_DIMS];
        zli_shape[0] = num_new;
        zli_shape[1] = hidden;
        zli_stride[0] = 0;
        zli_stride[1] = 1;
        let z_l_init = BufferView::from_raw(&core.z_l_init, 0, 2, zli_shape, zli_stride, f32_size);

        // z_H = embed(input_ids) * embedding_scale
        let embed_runner = self
            .model
            .embed
            .plan(device, queue, input_token_view, z_h);
        let embed_scale_runner = self.model.embed_scale.plan(device, queue, z_h);

        // Recurrent schedule (reference `HierarchicalReasoningModel.forward`):
        //   for h in 0..H_cycles:
        //       for l in 0..L_cycles: z_L = norm_f(L_stack(z_L + z_H))
        //       z_H = norm_f(H_stack(z_H + z_L))
        // The first L injection reads `z_L_init` (broadcast); later cycles read
        // the running `z_L`.
        let mut cycle_runners = Vec::with_capacity(core.h_cycles * (core.l_cycles + 1));
        for h in 0..core.h_cycles {
            for l in 0..core.l_cycles {
                let slow_inject = if h == 0 && l == 0 { z_l_init } else { z_l };
                let inject_runner = self
                    .model
                    .inject_add
                    .plan(device, queue, work, slow_inject, z_h);
                let stack_runner =
                    self.l_sessions[h * core.l_cycles + l]
                        .plan(device, queue, work, &self.position_buffer);
                let norm_runner = core.norm_f.plan(device, queue, work, z_l);
                cycle_runners.push(CycleRunner {
                    inject_runner,
                    stack_runner,
                    norm_runner,
                });
            }
            let inject_runner = self
                .model
                .inject_add
                .plan(device, queue, work, z_h, z_l);
            let stack_runner =
                self.h_sessions[h].plan(device, queue, work, &self.position_buffer);
            let norm_runner = core.norm_f.plan(device, queue, work, z_h);
            cycle_runners.push(CycleRunner {
                inject_runner,
                stack_runner,
                norm_runner,
            });
        }

        // Final z_H (already norm_f'd by the last H pass) → logits → sample.
        let last_row = z_h.narrow(0, num_new - 1, 1);
        let logits_view = BufferView::new_2d_tight(&self.logits, 1, vocab, f32_size);
        let logits_1d = BufferView::new_1d(&self.logits, f32_size, vocab);
        let sampled_token_view = BufferView::new_1d(sampled_token, u32_size, 1);
        let lm_head_runner = core
            .lm_head
            .plan(device, queue, last_row, logits_view);
        let sampler_runner = self
            .model
            .sampler
            .plan(device, queue, logits_1d, sampled_token_view);

        GpuModelRunner {
            embed_runner,
            embed_scale_runner,
            cycle_runners,
            lm_head_runner,
            sampler_runner,
        }
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
        let hidden = self.model.core.hidden_size;
        let n = input_ids.len();
        let prefill_tokens = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hrm_text_gpu_session/prefill_tokens"),
            size: n as wgpu::BufferAddress * u32_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&prefill_tokens, 0, bytemuck::cast_slice(input_ids));
        let make_hidden = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (n * hidden) as wgpu::BufferAddress * f32_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let z_h = make_hidden("hrm_text_gpu_session/prefill_z_h");
        let z_l = make_hidden("hrm_text_gpu_session/prefill_z_l");
        let work = make_hidden("hrm_text_gpu_session/prefill_work");
        self.plan(
            device,
            queue,
            &prefill_tokens,
            &z_h,
            &z_l,
            &work,
            n as u32,
            sampled_token,
        )
    }
}

/// Per-conversation streaming generator over a [`HrmTextGpuWorkspace`].
///
/// NOTE: attention is currently pure causal. HRM-Text was pre-trained with a
/// PrefixLM mask (bidirectional prompt, causal completion); matching it is a
/// follow-up. Causal decoding is the documented fallback and produces coherent
/// but slightly off-distribution logits versus the reference.
pub struct HrmTextGpuSession<'m> {
    workspace: HrmTextGpuWorkspace<'m>,
    /// 1 × u32. Sampler writes; decode embed reads on the next step.
    current_token: wgpu::Buffer,
    /// 1 × u32, mappable. Copied from `current_token` each step.
    token_readback: wgpu::Buffer,
    decode_runner: GpuModelRunner,
    max_seq_len: usize,
    tokens: Vec<u32>,
}

impl<'m> HrmTextGpuSession<'m> {
    pub fn new(
        model: &'m HrmTextGpuModel,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        max_seq_len: usize,
    ) -> Self {
        debug_assert!(max_seq_len >= 1, "max_seq_len must be >= 1");
        let workspace = HrmTextGpuWorkspace::new(model, device, max_seq_len);
        let u32_size = std::mem::size_of::<u32>() as wgpu::BufferAddress;
        let f32_size = std::mem::size_of::<f32>() as wgpu::BufferAddress;
        let current_token = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hrm_text_gpu_session/current_token"),
            size: u32_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Single-token decode working buffers. Kept alive by the decode
        // runner's bind groups after these handles drop.
        let make_decode_hidden = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: model.core.hidden_size as wgpu::BufferAddress * f32_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let decode_z_h = make_decode_hidden("hrm_text_gpu_session/decode_z_h");
        let decode_z_l = make_decode_hidden("hrm_text_gpu_session/decode_z_l");
        let decode_work = make_decode_hidden("hrm_text_gpu_session/decode_work");
        let decode_runner = workspace.plan(
            device,
            queue,
            &current_token,
            &decode_z_h,
            &decode_z_l,
            &decode_work,
            1,
            &current_token,
        );
        let token_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hrm_text_gpu_session/token_readback"),
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

    /// Append `input_ids`, dispatch the recurrent pipeline, and return the
    /// token id the sampler picked from the last position's logits.
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

    /// Encode `input_ids` and continue greedily up to `max_tokens` tokens
    /// (inclusive of the first), stopping early when any `stopping` fires.
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
        debug_assert!(!input_ids.is_empty(), "generate: input_ids must be non-empty");
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
    /// `prev_sampled_id` (the sampler wrote it on the previous dispatch).
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
            label: Some("hrm_text_gpu_session/step_encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hrm_text_gpu_session/step_compute_pass"),
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
