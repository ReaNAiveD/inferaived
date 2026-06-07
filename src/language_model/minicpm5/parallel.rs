use crate::buffer_view::BufferView;
use crate::embedding_lookup::EmbeddingLookupWebgpuRunner;
use crate::gpu_sampler::GpuSamplerRunner;
use crate::kernels::masked_block_attention::encode_visibility;
use crate::kernels::norm::RmsNormInplaceWebgpuRunner;
use crate::language_model::MiniCPM5GpuModel;
use crate::layers::minicpm5_masked_layer_stack::{
    MiniCPM5MaskedLayerStackRunner, MiniCPM5MaskedLayerStackSession,
};
use crate::lm_head::LmHeadWebgpuRunner;
use crate::parallel_window::{ContextWindow, ContextWindowId, PositionBand, WindowTable};
use crate::sampling::{SampledToken, SamplingParams, StoppingCriteria};
use async_stream::stream;
use futures_core::Stream;
use std::ops::Range;

impl MiniCPM5GpuModel {
    /// Open a [`MiniCPM5ContextNamespace`] builder over this model with the always-visible `prefix_tokens`.
    pub fn new_context_namespace<'m>(
        &'m self,
        max_seq_len: usize,
        prefix_tokens: &[u32],
    ) -> MiniCPM5ContextNamespace<'m> {
        MiniCPM5ContextNamespace::new(self, max_seq_len, prefix_tokens)
    }
}

/// A **mutable builder** for a reusable context over a [`MiniCPM5GpuModel`].
pub struct MiniCPM5ContextNamespace<'m> {
    model: &'m MiniCPM5GpuModel,
    /// Total rows in each per-layer KV pool, bounded by the hardware/model sequence limit.
    max_seq_len: usize,
    prefix_tokens: Vec<u32>,
    /// Staged windows, in id order (window id = index into this vec).
    windows: Vec<Vec<u32>>,
}

impl<'m> MiniCPM5ContextNamespace<'m> {
    /// Open an empty builder over `model` with the always-visible `prefix_tokens`.
    pub fn new(model: &'m MiniCPM5GpuModel, max_seq_len: usize, prefix_tokens: &[u32]) -> Self {
        assert!(max_seq_len >= 1, "max_seq_len must be >= 1");
        Self {
            model,
            max_seq_len,
            prefix_tokens: prefix_tokens.to_vec(),
            windows: Vec::new(),
        }
    }

    /// Stage a window of `tokens` and return its id.
    pub fn add_context(&mut self, tokens: &[u32]) -> ContextWindowId {
        let id = ContextWindowId(self.windows.len());
        self.windows.push(tokens.to_vec());
        id
    }

    pub fn band_width(&self) -> usize {
        self.windows
            .iter()
            .map(|w| w.len())
            .max()
            .unwrap_or(1)
            .max(1)
    }

    /// Encode the prefix and every staged window exactly once into freshly
    /// allocated per-layer KV pools, then freeze them into a [`MiniCPM5Context`].
    pub fn compile(self, device: &wgpu::Device, queue: &wgpu::Queue) -> MiniCPM5Context<'m> {
        // Fit the band width to the longest staged window.
        let band_width = self.band_width();
        let band = PositionBand::new(self.prefix_tokens.len(), band_width);
        let mut window_table = WindowTable::new(band);
        let staged: Vec<(ContextWindow, Vec<u32>)> = self
            .windows
            .into_iter()
            .map(|tokens| {
                let window = window_table.register_window(tokens.len());
                (window, tokens)
            })
            .collect();
        let context_kv_slots = window_table.total_slots();
        assert!(
            self.max_seq_len > context_kv_slots,
            "max_seq_len ({}) must exceed the {} context slots (prefix + windows) to leave room for decoding",
            self.max_seq_len,
            context_kv_slots,
        );
        let kv_capacity = self.max_seq_len;
        let session_capacity = self.max_seq_len - context_kv_slots;

        let layer_stack =
            MiniCPM5MaskedLayerStackSession::new(&self.model.core.layer_stack, device, kv_capacity);
        let max_visibility_ranges = staged.len() + 2;
        let encoder =
            MaskedContextEncoder::new(device, self.model, layer_stack, max_visibility_ranges);

        encoder.encode_prefix(device, queue, &self.prefix_tokens);
        for (window, tokens) in &staged {
            encoder.encode_window(device, queue, band, window, tokens);
        }

        MiniCPM5Context {
            workspace: encoder.finish(),
            session_capacity,
            window_table,
            kv_offset: context_kv_slots,
        }
    }
}

/// Embed + masked layer stack, planned to **populate the KV cache only**
/// (no head, no sampler). Returned by
/// [`MiniCPM5MaskedGpuWorkspace::plan_kv`] for prefix / window encoding.
struct MiniCPM5MaskedKvOnlyRunner {
    embed_runner: EmbeddingLookupWebgpuRunner,
    stack_runner: MiniCPM5MaskedLayerStackRunner,
}

impl MiniCPM5MaskedKvOnlyRunner {
    fn forward(&self, cpass: &mut wgpu::ComputePass<'_>) {
        self.embed_runner.forward(cpass);
        self.stack_runner.forward(cpass);
    }
}

/// Embed + masked layer stack + final_norm + lm_head + sampler chained into
/// one compute pass. Returned by [`MiniCPM5MaskedGpuWorkspace::plan`] for
/// inference forwards.
struct MiniCPM5MaskedRunner {
    embed_runner: EmbeddingLookupWebgpuRunner,
    stack_runner: MiniCPM5MaskedLayerStackRunner,
    final_norm_runner: RmsNormInplaceWebgpuRunner,
    lm_head_runner: LmHeadWebgpuRunner,
    sampler_runner: GpuSamplerRunner,
}

impl MiniCPM5MaskedRunner {
    fn forward(&self, cpass: &mut wgpu::ComputePass<'_>) {
        self.embed_runner.forward(cpass);
        self.stack_runner.forward(cpass);
        self.final_norm_runner.forward(cpass);
        self.lm_head_runner.forward(cpass);
        self.sampler_runner.forward(cpass);
    }
}

/// Everything a planned masked batch forward reads, supplied per batch by the caller.
struct MiniCPM5MaskedGpuWorkspace<'m> {
    model: &'m MiniCPM5GpuModel,
    layer_session: MiniCPM5MaskedLayerStackSession<'m>,
    /// `u32` RoPE base position, token `i` rotates at base `+ i`.
    rope_position_buffer: wgpu::Buffer,
    /// `u32` destination KV cache slot for the first token of the in-flight
    /// batch (token `i` scatters to slot `+ i`).
    scatter_position_buffer: wgpu::Buffer,
    /// `[1, vocab]` logits for the batch's last token.
    logits: wgpu::Buffer,
    /// Persistent flat `visibility` range list.
    visibility: wgpu::Buffer,
}

impl<'m> MiniCPM5MaskedGpuWorkspace<'m> {
    /// Allocate the shared, context-lifetime GPU resources.
    fn new(
        device: &wgpu::Device,
        model: &'m MiniCPM5GpuModel,
        layer_session: MiniCPM5MaskedLayerStackSession<'m>,
        max_visibility_ranges: usize,
    ) -> Self {
        let vocab_size = model.core.vocab_size;
        let f32_sz = std::mem::size_of::<f32>() as wgpu::BufferAddress;
        let u32_sz = std::mem::size_of::<u32>() as wgpu::BufferAddress;
        let rope_position_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("minicpm5_context/rope_position"),
            size: u32_sz,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scatter_position_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("minicpm5_context/scatter_position"),
            size: u32_sz,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let logits = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("minicpm5_context/logits"),
            size: vocab_size as wgpu::BufferAddress * f32_sz,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let visibility = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("minicpm5_context/visibility"),
            size: (1 + 2 * max_visibility_ranges) as wgpu::BufferAddress * u32_sz,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            model,
            layer_session,
            rope_position_buffer,
            scatter_position_buffer,
            logits,
            visibility,
        }
    }

    fn plan_kv(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_token: &wgpu::Buffer,
        input_hidden: &wgpu::Buffer,
        num_new: u32,
    ) -> MiniCPM5MaskedKvOnlyRunner {
        let f32_sz = std::mem::size_of::<f32>() as u32;
        let u32_sz = std::mem::size_of::<u32>() as u32;
        let hidden_size = self.model.core.hidden_size as u32;
        let input_token_view = BufferView::new_1d(input_token, u32_sz, num_new);
        let input_hidden_view =
            BufferView::new_2d_tight(input_hidden, num_new, hidden_size, f32_sz);

        let embed_runner =
            self.model
                .embed
                .plan(device, queue, input_token_view, input_hidden_view);

        let stack_runner = self.layer_session.plan(
            device,
            queue,
            input_hidden_view,
            &self.rope_position_buffer,
            &self.scatter_position_buffer,
            &self.visibility,
        );

        MiniCPM5MaskedKvOnlyRunner {
            embed_runner,
            stack_runner,
        }
    }

    fn plan(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_token: &wgpu::Buffer,
        input_hidden: &wgpu::Buffer,
        num_new: u32,
        sampled_token: &wgpu::Buffer,
    ) -> MiniCPM5MaskedRunner {
        let MiniCPM5MaskedKvOnlyRunner {
            embed_runner,
            stack_runner,
        } = self.plan_kv(device, queue, input_token, input_hidden, num_new);

        let f32_sz = std::mem::size_of::<f32>() as u32;
        let u32_sz = std::mem::size_of::<u32>() as u32;
        let hidden_size = self.model.core.hidden_size as u32;
        let vocab_size = self.model.core.vocab_size as u32;
        let input_hidden_view =
            BufferView::new_2d_tight(input_hidden, num_new, hidden_size, f32_sz);
        let last_row = input_hidden_view.narrow(0, num_new - 1, 1);
        let logits_view = BufferView::new_2d_tight(&self.logits, 1, vocab_size, f32_sz);
        let final_norm_runner = self.model.core.final_norm.plan(device, queue, last_row);
        let lm_head_runner = self
            .model
            .core
            .lm_head
            .plan(device, queue, last_row, logits_view);
        let logits_1d = BufferView::new_1d(&self.logits, f32_sz, vocab_size);
        let sampled_token_view = BufferView::new_1d(sampled_token, u32_sz, 1);
        let sampler_runner = self
            .model
            .sampler
            .plan(device, queue, logits_1d, sampled_token_view);

        MiniCPM5MaskedRunner {
            embed_runner,
            stack_runner,
            final_norm_runner,
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
    ) -> MiniCPM5MaskedRunner {
        let u32_sz = std::mem::size_of::<u32>() as wgpu::BufferAddress;
        let f32_sz = std::mem::size_of::<f32>() as wgpu::BufferAddress;
        let hidden_size = self.model.core.hidden_size;
        let prefill_tokens = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("minicpm5_context/prefill_tokens"),
            size: input_ids.len() as wgpu::BufferAddress * u32_sz,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&prefill_tokens, 0, bytemuck::cast_slice(input_ids));
        let prefill_hidden = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("minicpm5_context/prefill_hidden"),
            size: (input_ids.len() * hidden_size) as wgpu::BufferAddress * f32_sz,
            usage: wgpu::BufferUsages::STORAGE,
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

    fn plan_kv_prefill(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_ids: &[u32],
    ) -> MiniCPM5MaskedKvOnlyRunner {
        let u32_sz = std::mem::size_of::<u32>() as wgpu::BufferAddress;
        let f32_sz = std::mem::size_of::<f32>() as wgpu::BufferAddress;
        let hidden_size = self.model.core.hidden_size;
        let prefill_tokens = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("minicpm5_context/prefill_tokens"),
            size: input_ids.len() as wgpu::BufferAddress * u32_sz,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&prefill_tokens, 0, bytemuck::cast_slice(input_ids));
        let prefill_hidden = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("minicpm5_context/prefill_hidden"),
            size: (input_ids.len() * hidden_size) as wgpu::BufferAddress * f32_sz,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        self.plan_kv(
            device,
            queue,
            &prefill_tokens,
            &prefill_hidden,
            input_ids.len() as u32,
        )
    }

    /// Rewrite the two 1 × u32 position scalars in place.
    fn write_positions(&self, queue: &wgpu::Queue, rope_position_start: u32, kv_slot: u32) {
        queue.write_buffer(
            &self.rope_position_buffer,
            0,
            bytemuck::bytes_of(&rope_position_start),
        );
        queue.write_buffer(
            &self.scatter_position_buffer,
            0,
            bytemuck::bytes_of(&kv_slot),
        );
    }

    /// Rewrite the persistent `visibility` range list in place.
    fn write_visibility(
        &self,
        queue: &wgpu::Queue,
        shared_ranges: &[Range<u32>],
        causal_anchor_slot: u32,
    ) {
        let data = encode_visibility(shared_ranges, causal_anchor_slot);
        queue.write_buffer(&self.visibility, 0, bytemuck::cast_slice(&data));
    }
}

/// **Build-only** handle that owns a fresh [`MiniCPM5MaskedGpuWorkspace`] while
/// the context's prefix and windows are being encoded into the KV cache.
struct MaskedContextEncoder<'m> {
    workspace: MiniCPM5MaskedGpuWorkspace<'m>,
}

impl<'m> MaskedContextEncoder<'m> {
    /// Allocate the underlying workspace, ready to receive prefix + window encodes.
    fn new(
        device: &wgpu::Device,
        model: &'m MiniCPM5GpuModel,
        layer_session: MiniCPM5MaskedLayerStackSession<'m>,
        max_visibility_ranges: usize,
    ) -> Self {
        Self {
            workspace: MiniCPM5MaskedGpuWorkspace::new(
                device,
                model,
                layer_session,
                max_visibility_ranges,
            ),
        }
    }

    fn finish(self) -> MiniCPM5MaskedGpuWorkspace<'m> {
        self.workspace
    }

    /// Forward `tokens` through `embed → masked layer stack` into the KV
    /// cache (no logits, no sampling) using the given RoPE / scatter / visibility scalars.
    fn encode_forward(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tokens: &[u32],
        rope_position_start: u32,
        kv_slot_start: u32,
        shared_ranges: &[Range<u32>],
        causal_anchor_slot: u32,
    ) {
        debug_assert!(!tokens.is_empty(), "batch requires at least one token");
        self.workspace
            .write_visibility(queue, shared_ranges, causal_anchor_slot);
        self.workspace
            .write_positions(queue, rope_position_start, kv_slot_start);
        let runner = self.workspace.plan_kv_prefill(device, queue, tokens);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("minicpm5_context/encode_encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("minicpm5_context/encode_pass"),
                timestamp_writes: None,
            });
            runner.forward(&mut cpass);
        }
        queue.submit(Some(encoder.finish()));
    }

    /// Encode the shared prefix into KV cache slots `[0, P)` of every layer
    /// as a single causal-triangle batched forward.
    fn encode_prefix(&self, device: &wgpu::Device, queue: &wgpu::Queue, tokens: &[u32]) {
        if tokens.is_empty() {
            return;
        }
        self.encode_forward(device, queue, tokens, 0, 0, &[], 0);
    }

    /// Encode `window` into its assigned KV cache slots, with RoPE offset
    /// `P + B − L` so the window ends at the shared anchor.
    fn encode_window(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        band: PositionBand,
        window: &ContextWindow,
        tokens: &[u32],
    ) {
        if tokens.is_empty() {
            return;
        }
        let prefix_len = band.prefix_len();
        let rope_offset = band.window_start_position(window.seq_len);
        let slot_start = window.slot_start;
        let prefix_range = 0..prefix_len as u32;
        self.encode_forward(
            device,
            queue,
            tokens,
            rope_offset as u32,
            slot_start as u32,
            std::slice::from_ref(&prefix_range),
            slot_start as u32,
        );
    }
}

/// A **compiled context** over a frozen set of parallel windows.
pub struct MiniCPM5Context<'m> {
    /// Persistent shared GPU resources that every [`MiniCPM5MaskedSession`] runs on top of.
    workspace: MiniCPM5MaskedGpuWorkspace<'m>,
    /// Slots reserved for a session's own prompt + decoded tokens.
    session_capacity: usize,
    /// Band geometry + window→slot assignment for the context region.
    window_table: WindowTable,
    /// First KV cache slot of the session region.
    kv_offset: usize,
}

impl<'m> MiniCPM5Context<'m> {
    /// Open an empty generation [`MiniCPM5MaskedSession`] over this context that sees
    /// the shared prefix plus the `visible` windows.
    ///
    /// The returned session borrows the context for its whole lifetime, so only
    /// one session is active at a time.
    pub fn begin<'ctx>(
        &'ctx mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        visible: &[ContextWindowId],
    ) -> MiniCPM5MaskedSession<'ctx, 'm> {
        // Write the session's visibility range list into the workspace's persistent
        // `visibility` buffer; prefill and decode reuse it as-is.
        let shared: Vec<Range<u32>> = self
            .window_table
            .visible_ranges(visible)
            .unwrap_or_else(|e| panic!("invalid session visibility: {e}"))
            .into_iter()
            .map(|r| r.start as u32..r.end as u32)
            .collect();
        self.workspace
            .write_visibility(queue, &shared, self.kv_offset as u32);

        MiniCPM5MaskedSession::new(
            device,
            queue,
            &self.workspace,
            self.window_table.band(),
            self.session_capacity,
            self.kv_offset,
        )
    }
}

/// A **rewindable generation session** over a [`MiniCPM5Context`].
pub struct MiniCPM5MaskedSession<'ctx, 'm> {
    workspace: &'ctx MiniCPM5MaskedGpuWorkspace<'m>,
    /// Band geometry.
    band: PositionBand,
    /// Slots reserved for this session's prompt + decoded tokens.
    session_capacity: usize,
    /// First KV cache slot of the session region.
    kv_offset: usize,
    /// 1 × u32. Sampler writes; decode embed reads on the next step.
    current_token: wgpu::Buffer,
    /// 1 × u32, mappable. Copied from `current_token` each step.
    token_readback: wgpu::Buffer,
    decode_runner: MiniCPM5MaskedRunner,
    /// Every token fed so far.
    tokens: Vec<u32>,
}

impl<'ctx, 'm> MiniCPM5MaskedSession<'ctx, 'm> {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        workspace: &'ctx MiniCPM5MaskedGpuWorkspace<'m>,
        band: PositionBand,
        session_capacity: usize,
        kv_offset: usize,
    ) -> Self {
        let u32_sz = std::mem::size_of::<u32>() as wgpu::BufferAddress;
        let f32_sz = std::mem::size_of::<f32>() as wgpu::BufferAddress;
        let current_token = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("minicpm5_context/current_token"),
            size: u32_sz,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let token_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("minicpm5_context/token_readback"),
            size: u32_sz,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let decode_hidden = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("minicpm5_context/decode_hidden"),
            size: workspace.model.core.hidden_size as wgpu::BufferAddress * f32_sz,
            usage: wgpu::BufferUsages::STORAGE,
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

        Self {
            workspace,
            band,
            session_capacity,
            kv_offset,
            current_token,
            token_readback,
            decode_runner,
            tokens: Vec::new(),
        }
    }

    /// Rewind the session to `token_count` tokens, discarding everything after
    /// it; the next [`Self::step`] / [`Self::generate`] continues from there.
    ///
    /// Panics if `token_count` is past the current tail.
    pub fn rewind(&mut self, token_count: usize) {
        assert!(
            token_count <= self.tokens.len(),
            "cannot rewind past the current tail ({} > {})",
            token_count,
            self.tokens.len(),
        );
        self.tokens.truncate(token_count);
    }

    /// Feed `input_ids` into the session, dispatch the GPU pipeline, and
    /// return the token id the sampler picked from the last position's logits.
    /// `input_ids` is always honored verbatim.
    pub async fn step(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_ids: &[u32],
        _params: &SamplingParams,
    ) -> SampledToken {
        let num_new = input_ids.len();
        debug_assert!(num_new >= 1, "step requires non-empty input_ids");
        let n_past = self.tokens.len();
        debug_assert!(
            n_past + num_new <= self.session_capacity,
            "session overflow: position {} + {} new tokens exceeds session capacity {}",
            n_past,
            num_new,
            self.session_capacity,
        );
        let band = self.band;
        let kv_slot = self.kv_offset + n_past;
        self.workspace.write_positions(
            queue,
            band.generation_position(n_past) as u32,
            kv_slot as u32,
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

    /// Encode `input_ids` and continue sampling up to `max_tokens` tokens
    /// (inclusive of the first), stopping early when any element of `stopping`
    /// fires.
    ///
    /// Returns a lazy `impl Stream`; tokens are produced as the caller polls.
    /// Dropping the stream (or breaking the consumer loop) aborts generation
    /// without launching further GPU work.
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
    /// `prev_sampled_id`. Private because the precondition is invisible at the
    /// API boundary; only [`generate`](Self::generate) can satisfy it.
    async fn decode_next(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        prev_sampled_id: u32,
        _params: &SamplingParams,
    ) -> SampledToken {
        let n_past = self.tokens.len();
        debug_assert!(
            n_past < self.session_capacity,
            "session overflow: position {} + 1 exceeds session capacity {}",
            n_past,
            self.session_capacity,
        );
        let band = self.band;
        let kv_slot = self.kv_offset + n_past;
        self.workspace.write_positions(
            queue,
            band.generation_position(n_past) as u32,
            kv_slot as u32,
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
        runner: &MiniCPM5MaskedRunner,
    ) -> u32 {
        let u32_sz = std::mem::size_of::<u32>() as wgpu::BufferAddress;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("minicpm5_context/decode_encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("minicpm5_context/decode_pass"),
                timestamp_writes: None,
            });
            runner.forward(&mut cpass);
        }
        encoder.copy_buffer_to_buffer(&self.current_token, 0, &self.token_readback, 0, u32_sz);
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
