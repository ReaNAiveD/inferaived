//! End-to-end decode-throughput benchmark.
//!
//! Run with:
//!     cargo run --release --example bench_decode
//!
//! Tuning knobs (all optional environment variables):
//!     BENCH_WARMUP_TOKENS   default 8    tokens decoded before the timer starts
//!                                        (covers pipeline compilation + first-touch costs)
//!     BENCH_MEASURE_TOKENS  default 64   tokens decoded under the timer per run
//!     BENCH_RUNS            default 3    measured runs; reports min / median
//!     BENCH_PROMPT          default ""   prompt text (defaults to a fixed sentence)
//!     BENCH_CSV             default ""   if set, append one CSV row to this file
//!
//! Output is human-readable per-run plus a final single-line CSV summary on stdout
//! (always) and optionally appended to `$BENCH_CSV`. Schema:
//!
//!     git_sha,timestamp_unix,prompt_tokens,warmup,measure,runs,
//!     ttft_ms_min,ttft_ms_median,decode_tok_s_min,decode_tok_s_median
//!
//! `ttft_*` is total prefill latency for the entire prompt (encode + first forward
//! call, including its CPU readback). `decode_tok_s_*` is steady-state throughput
//! measured only over the `BENCH_MEASURE_TOKENS` post-warmup tokens, so it excludes
//! both prefill and the warmup decode steps.

use inferaived::language_model::{Qwen35Config, Qwen35GpuModel, Qwen35GpuSession};
use inferaived::sampling::SamplingParams;
use safetensors::SafeTensors;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;
use tokio_stream::StreamExt;
use tracing::info;
use wgpu::{
    BackendOptions, Backends, DeviceDescriptor, ExperimentalFeatures, Features, Instance,
    InstanceDescriptor, InstanceFlags, MemoryBudgetThresholds, MemoryHints, PowerPreference,
    RequestAdapterOptions, Trace,
};

const DEFAULT_PROMPT: &str = "Inferaived is a Rust library for running transformer-based language models on consumer GPUs using WebGPU.";

const MODEL_SAFETENSORS: &str = "model/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors";
const MODEL_TOKENIZER: &str = "model/Qwen3.5-0.8B/tokenizer.json";
const MODEL_CONFIG: &str = "model/Qwen3.5-0.8B/config.json";

fn features(supported: Features) -> Features {
    let mut required = Features::empty();
    if supported.contains(Features::SHADER_F16) {
        required |= Features::SHADER_F16;
    }
    if supported.contains(Features::TIMESTAMP_QUERY) {
        required |= Features::TIMESTAMP_QUERY;
    }
    if supported.contains(Features::SUBGROUP) {
        required |= Features::SUBGROUP;
    }
    if supported.contains(Features::SUBGROUP_BARRIER) {
        required |= Features::SUBGROUP_BARRIER;
    }
    if supported.contains(Features::SHADER_FLOAT32_ATOMIC) {
        required |= Features::SHADER_FLOAT32_ATOMIC;
    }
    required
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_string(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| default.to_string())
}

fn median(sorted: &[f64]) -> f64 {
    debug_assert!(!sorted.is_empty());
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

/// Best-effort `git rev-parse --short HEAD`. Returns `"unknown"` if git is
/// unavailable or the call fails (e.g. running outside a checkout).
fn git_sha() -> String {
    let out = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output();
    match out {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        _ => "unknown".to_string(),
    }
}

#[derive(Debug, Clone, Copy)]
struct RunResult {
    ttft_ms: f64,
    decode_tok_s: f64,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // -------- config --------
    let warmup_tokens = env_usize("BENCH_WARMUP_TOKENS", 8);
    let measure_tokens = env_usize("BENCH_MEASURE_TOKENS", 64);
    let runs = env_usize("BENCH_RUNS", 3);
    let prompt = env_string("BENCH_PROMPT", DEFAULT_PROMPT);
    let csv_path = std::env::var("BENCH_CSV").ok().filter(|s| !s.is_empty());

    assert!(runs >= 1, "BENCH_RUNS must be >= 1");
    assert!(measure_tokens >= 1, "BENCH_MEASURE_TOKENS must be >= 1");

    // -------- load model + tokenizer (once, shared across runs) --------
    let buffer = std::fs::read(MODEL_SAFETENSORS).expect("Failed to read safetensors");
    let tensors = SafeTensors::deserialize(&buffer[..]).expect("Failed to deserialize tensors");
    let tokenizer = Tokenizer::from_file(MODEL_TOKENIZER).expect("Failed to load tokenizer");

    let encoded = tokenizer
        .encode(prompt.as_str(), false)
        .expect("Failed to encode prompt");
    let prompt_ids: Vec<u32> = encoded.get_ids().to_vec();
    let prompt_tokens = prompt_ids.len();

    // Session must fit: prompt + warmup + measure tokens.
    let max_seq_len = prompt_tokens + warmup_tokens + measure_tokens;

    println!(
        "bench_decode: prompt_tokens={prompt_tokens} warmup={warmup_tokens} \
         measure={measure_tokens} runs={runs} max_seq_len={max_seq_len}"
    );

    // -------- wgpu init (once, shared across runs) --------
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::PRIMARY,
        flags: InstanceFlags::default(),
        memory_budget_thresholds: MemoryBudgetThresholds::default(),
        backend_options: BackendOptions::default(),
        display: None,
    });
    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .expect("Failed to request adapter");
    let (device, queue) = adapter
        .request_device(&DeviceDescriptor {
            label: None,
            required_features: features(adapter.features()),
            required_limits: adapter.limits(),
            experimental_features: ExperimentalFeatures::default(),
            memory_hints: MemoryHints::Performance,
            trace: Trace::default(),
        })
        .await
        .expect("Failed to request device");
    info!(
        "Device ready: backend={:?} name={:?}",
        adapter.get_info().backend,
        adapter.get_info().name
    );

    let config =
        Qwen35Config::from_json_file(MODEL_CONFIG).expect("Failed to load model config");
    let model = Qwen35GpuModel::new(&device, &queue, &tensors, &config.text_config);

    // -------- runs --------
    // Each run builds a fresh session so KV-cache state doesn't leak across
    // runs and the warmup-vs-measure split is honest (warmup tokens always
    // hit a fresh cache, just like the measure phase).
    let mut results: Vec<RunResult> = Vec::with_capacity(runs);
    for run_idx in 0..runs {
        let mut session = Qwen35GpuSession::new(&model, &device, &queue, max_seq_len);
        let params = SamplingParams::default();

        // Prefill: encode prompt + run first forward. We treat the entire
        // first call as TTFT, including its CPU readback (which is part of
        // the user-observable latency anyway).
        let t0 = Instant::now();
        let tok = session.step(&device, &queue, &prompt_ids, &params).await;
        let ttft_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let mut next = tok.id;

        // Warmup decode (untimed): covers pipeline compilation and
        // first-touch caching not exercised by prefill.
        if warmup_tokens > 0 {
            let warm: Vec<_> = session
                .generate(&device, &queue, &[next], &params, warmup_tokens, &[])
                .collect()
                .await;
            next = warm.last().expect("non-empty warmup").id;
        }

        // Measured decode. Drive the stream by hand so we don't allocate
        // a Vec we'll just throw away.
        let t1 = Instant::now();
        {
            let seed = [next];
            let stream = session.generate(&device, &queue, &seed, &params, measure_tokens, &[]);
            tokio::pin!(stream);
            while stream.next().await.is_some() {}
        }
        let decode_secs = t1.elapsed().as_secs_f64();
        let decode_tok_s = measure_tokens as f64 / decode_secs;

        println!(
            "run {}/{}: ttft={:.1} ms  decode={:.2} tok/s  ({} tok in {:.3} s)",
            run_idx + 1,
            runs,
            ttft_ms,
            decode_tok_s,
            measure_tokens,
            decode_secs,
        );

        results.push(RunResult {
            ttft_ms,
            decode_tok_s,
        });
    }

    // -------- aggregate --------
    let mut ttft: Vec<f64> = results.iter().map(|r| r.ttft_ms).collect();
    let mut decode: Vec<f64> = results.iter().map(|r| r.decode_tok_s).collect();
    ttft.sort_by(|a, b| a.partial_cmp(b).unwrap());
    decode.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let ttft_min = ttft[0];
    let ttft_median = median(&ttft);
    // For throughput, MIN is the pessimistic (worst-case) tok/s; we instead
    // report MIN as the *floor* and median as the typical value. The
    // headline number to track in regressions is `decode_tok_s_median`.
    let decode_min = decode[0];
    let decode_median = median(&decode);

    println!();
    println!(
        "summary: ttft_min={:.1} ms  ttft_med={:.1} ms  decode_min={:.2} tok/s  decode_med={:.2} tok/s",
        ttft_min, ttft_median, decode_min, decode_median
    );

    // -------- CSV line --------
    let sha = git_sha();
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let csv_line = format!(
        "{sha},{ts},{prompt_tokens},{warmup_tokens},{measure_tokens},{runs},\
         {ttft_min:.3},{ttft_median:.3},{decode_min:.4},{decode_median:.4}"
    );
    println!("csv: {csv_line}");

    if let Some(path) = csv_path {
        use std::io::Write;
        let write_header = !std::path::Path::new(&path).exists();
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .expect("Failed to open BENCH_CSV file");
        if write_header {
            writeln!(
                f,
                "git_sha,timestamp_unix,prompt_tokens,warmup,measure,runs,ttft_ms_min,ttft_ms_median,decode_tok_s_min,decode_tok_s_median"
            )
            .expect("Failed to write CSV header");
        }
        writeln!(f, "{csv_line}").expect("Failed to append CSV line");
        println!("appended to {path}");
    }
}
