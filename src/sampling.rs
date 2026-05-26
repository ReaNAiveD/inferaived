use rand::SeedableRng;
use rand::distr::{Distribution, weighted::WeightedIndex};
use rand::rngs::SmallRng;

/// Numeric knobs for the standard logit transforms + terminal samplers.
#[derive(Clone, Copy, Debug)]
pub struct SamplingParams {
    /// `<= 0` ⇒ argmax (no random draw); `1.0` ⇒ identity scaling.
    pub temperature: f32,
    /// Keep top-k logits; `0` ⇒ disabled.
    pub top_k: usize,
    /// Nucleus probability mass; `1.0` ⇒ disabled.
    pub top_p: f32,
    /// Minimum probability relative to max; `0.0` ⇒ disabled.
    pub min_p: f32,
    /// HF-style repetition penalty; `1.0` ⇒ disabled.
    pub repetition_penalty: f32,
    /// Subtract `presence` per unique repeated token; `0.0` ⇒ disabled.
    pub presence_penalty: f32,
    /// Subtract `freq * count` per repeated token; `0.0` ⇒ disabled.
    pub frequency_penalty: f32,
    /// RNG seed for [`Sampler::Multinomial`]; unused for [`Sampler::Greedy`].
    pub rng_seed: u64,
}

impl Default for SamplingParams {
    /// Pure greedy decoding: `temperature = 0`, every other knob at its
    /// disabled value.
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            rng_seed: 0,
        }
    }
}

/// One sampled token plus its post-softmax log-probability under the
/// distribution the sampler drew from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampledToken {
    pub id: u32,
    pub logprob: f32,
}

/// Pure logit transform. CPU only. Composable into a chain.
pub enum LogitsProcessor {
    Temperature(f32),
    TopK(usize),
    TopP(f32),
    MinP(f32),
    RepetitionPenalty(f32),
    FreqPresencePenalty { freq: f32, presence: f32 },
    LogitBias(Vec<(u32, f32)>),
}

impl LogitsProcessor {
    pub fn process(&self, generated: &[u32], logits: &mut [f32]) {
        match self {
            LogitsProcessor::Temperature(t) => apply_temperature(*t, logits),
            LogitsProcessor::TopK(k) => apply_top_k(*k, logits),
            LogitsProcessor::TopP(p) => apply_top_p(*p, logits),
            LogitsProcessor::MinP(p) => apply_min_p(*p, logits),
            LogitsProcessor::RepetitionPenalty(penalty) => {
                apply_repetition_penalty(*penalty, generated, logits);
            }
            LogitsProcessor::FreqPresencePenalty { freq, presence } => {
                apply_freq_presence_penalty(*freq, *presence, generated, logits);
            }
            LogitsProcessor::LogitBias(biases) => {
                for (id, bias) in biases {
                    if let Some(l) = logits.get_mut(*id as usize) {
                        *l += *bias;
                    }
                }
            }
        }
    }
}

/// Terminal token selector. CPU only.
pub enum Sampler {
    Greedy,
    Multinomial { rng: SmallRng },
}

impl Sampler {
    /// `logits` is the post-processor vector. `generated` is the running
    /// list of previously-sampled token ids (unused by the built-in
    /// samplers, exposed for symmetry with [`LogitsProcessor::process`]).
    pub fn sample(&mut self, _generated: &[u32], logits: &[f32]) -> SampledToken {
        match self {
            Sampler::Greedy => sample_greedy(logits),
            Sampler::Multinomial { rng } => sample_multinomial(rng, logits),
        }
    }
}

/// Session-level orthogonal generation-stopping rule.
pub enum StoppingCriteria {
    Eos(Vec<u32>),
    // Future: StopStrings { ... } once we wire in a tokenizer.
}

impl StoppingCriteria {
    pub fn is_done(&self, _generated: &[u32], last: SampledToken) -> bool {
        match self {
            StoppingCriteria::Eos(eos) => eos.contains(&last.id),
        }
    }
}

/// Build the standard processor chain from `params`.
///
/// Order: repetition / freq-presence penalties first, then truncation
/// (top-k, top-p, min-p), then temperature. Empty vec when every knob is
/// at its disabled value.
pub fn default_processors(p: &SamplingParams) -> Vec<LogitsProcessor> {
    let mut out = Vec::new();
    if p.repetition_penalty != 1.0 {
        out.push(LogitsProcessor::RepetitionPenalty(p.repetition_penalty));
    }
    if p.frequency_penalty != 0.0 || p.presence_penalty != 0.0 {
        out.push(LogitsProcessor::FreqPresencePenalty {
            freq: p.frequency_penalty,
            presence: p.presence_penalty,
        });
    }
    if p.top_k != 0 {
        out.push(LogitsProcessor::TopK(p.top_k));
    }
    if p.top_p < 1.0 {
        out.push(LogitsProcessor::TopP(p.top_p));
    }
    if p.min_p > 0.0 {
        out.push(LogitsProcessor::MinP(p.min_p));
    }
    if p.temperature > 0.0 && p.temperature != 1.0 {
        out.push(LogitsProcessor::Temperature(p.temperature));
    }
    out
}

/// [`Sampler::Greedy`] when `temperature <= 0`, else
/// [`Sampler::Multinomial`] seeded from `params.rng_seed`.
pub fn default_sampler(p: &SamplingParams) -> Sampler {
    if p.temperature <= 0.0 {
        Sampler::Greedy
    } else {
        Sampler::Multinomial {
            rng: SmallRng::seed_from_u64(p.rng_seed),
        }
    }
}

// -----------------------------------------------------------------------------
// Processor implementations
// -----------------------------------------------------------------------------

fn apply_temperature(t: f32, logits: &mut [f32]) {
    if t <= 0.0 || t == 1.0 {
        return;
    }
    for l in logits.iter_mut() {
        *l /= t;
    }
}

fn apply_top_k(k: usize, logits: &mut [f32]) {
    if k == 0 || k >= logits.len() {
        return;
    }
    // O(n) partition to find the k-th largest logit value.
    let mut sorted: Vec<f32> = logits.to_vec();
    sorted.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
    let threshold = sorted[k - 1];
    for l in logits.iter_mut() {
        if l.total_cmp(&threshold).is_lt() {
            *l = f32::NEG_INFINITY;
        }
    }
}

fn apply_top_p(p: f32, logits: &mut [f32]) {
    if p >= 1.0 {
        return;
    }
    let probs = softmax(logits);
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_unstable_by(|&i, &j| probs[j].total_cmp(&probs[i]));
    // Smallest prefix whose cumulative prob is >= p.
    let mut cumsum = 0.0f32;
    let mut keep = 0usize;
    for &idx in &indices {
        cumsum += probs[idx];
        keep += 1;
        if cumsum >= p {
            break;
        }
    }
    for &idx in &indices[keep..] {
        logits[idx] = f32::NEG_INFINITY;
    }
}

fn apply_min_p(p: f32, logits: &mut [f32]) {
    if p <= 0.0 {
        return;
    }
    let probs = softmax(logits);
    let max_p = probs.iter().fold(0.0f32, |a, &b| a.max(b));
    let threshold = p * max_p;
    for (i, &prob) in probs.iter().enumerate() {
        if prob < threshold {
            logits[i] = f32::NEG_INFINITY;
        }
    }
}

fn apply_repetition_penalty(penalty: f32, generated: &[u32], logits: &mut [f32]) {
    if penalty == 1.0 {
        return;
    }
    // HF convention: positive logits divide, negative logits multiply.
    // Apply at most once per unique token in history.
    let mut seen = std::collections::HashSet::new();
    for &id in generated {
        if !seen.insert(id) {
            continue;
        }
        if let Some(l) = logits.get_mut(id as usize) {
            if *l > 0.0 {
                *l /= penalty;
            } else {
                *l *= penalty;
            }
        }
    }
}

fn apply_freq_presence_penalty(freq: f32, presence: f32, generated: &[u32], logits: &mut [f32]) {
    if freq == 0.0 && presence == 0.0 {
        return;
    }
    let mut counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for &id in generated {
        *counts.entry(id).or_insert(0) += 1;
    }
    for (id, count) in counts {
        if let Some(l) = logits.get_mut(id as usize) {
            *l -= freq * count as f32 + presence;
        }
    }
}

// -----------------------------------------------------------------------------
// Sampler implementations
// -----------------------------------------------------------------------------

fn sample_greedy(logits: &[f32]) -> SampledToken {
    let (idx, &val) = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .expect("sample_greedy: empty logits");
    SampledToken {
        id: idx as u32,
        logprob: logprob_of(logits, val),
    }
}

fn sample_multinomial(rng: &mut SmallRng, logits: &[f32]) -> SampledToken {
    let probs = softmax(logits);
    let dist = WeightedIndex::new(&probs).expect("sample_multinomial: invalid weights");
    let idx = dist.sample(rng);
    SampledToken {
        id: idx as u32,
        logprob: probs[idx].max(f32::MIN_POSITIVE).ln(),
    }
}

// -----------------------------------------------------------------------------
// Numerics
// -----------------------------------------------------------------------------

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// `ln(softmax(logits)[idx])`, given the logit value at `idx`. Numerically
/// stable via the log-sum-exp trick.
fn logprob_of(logits: &[f32], logit_at_idx: f32) -> f32 {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let log_sum_exp = max + logits.iter().map(|&l| (l - max).exp()).sum::<f32>().ln();
    logit_at_idx - log_sum_exp
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn default_params_is_greedy() {
        let p = SamplingParams::default();
        assert!(default_processors(&p).is_empty());
        assert!(matches!(default_sampler(&p), Sampler::Greedy));
    }

    #[test]
    fn temperature_disabled_is_noop() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let before = logits.clone();
        apply_temperature(1.0, &mut logits);
        assert_eq!(logits, before);
        apply_temperature(0.0, &mut logits);
        assert_eq!(logits, before);
    }

    #[test]
    fn temperature_scales_logits() {
        let mut logits = vec![1.0, 2.0, 4.0];
        apply_temperature(2.0, &mut logits);
        assert_eq!(logits, vec![0.5, 1.0, 2.0]);
    }

    #[test]
    fn top_k_masks_below_threshold() {
        let mut logits = vec![1.0, 5.0, 3.0, 4.0, 2.0];
        apply_top_k(2, &mut logits);
        // Top-2 are 5.0 (idx 1) and 4.0 (idx 3); others become -inf.
        assert_eq!(logits[1], 5.0);
        assert_eq!(logits[3], 4.0);
        assert!(logits[0].is_infinite() && logits[0].is_sign_negative());
        assert!(logits[2].is_infinite() && logits[2].is_sign_negative());
        assert!(logits[4].is_infinite() && logits[4].is_sign_negative());
    }

    #[test]
    fn top_k_disabled_is_noop() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let before = logits.clone();
        apply_top_k(0, &mut logits);
        assert_eq!(logits, before);
        apply_top_k(100, &mut logits);
        assert_eq!(logits, before);
    }

    #[test]
    fn top_p_keeps_smallest_prefix_above_p() {
        // Probabilities after softmax of [4, 3, 2, 1] are roughly
        // [0.643, 0.236, 0.087, 0.032]. With p=0.8, the smallest prefix is
        // {0.643, 0.236} summing to 0.879. Indices 2 and 3 should be masked.
        let mut logits = vec![4.0, 3.0, 2.0, 1.0];
        apply_top_p(0.8, &mut logits);
        assert_eq!(logits[0], 4.0);
        assert_eq!(logits[1], 3.0);
        assert!(logits[2].is_infinite() && logits[2].is_sign_negative());
        assert!(logits[3].is_infinite() && logits[3].is_sign_negative());
    }

    #[test]
    fn min_p_masks_low_probability_tokens() {
        // Probs ~ [0.643, 0.236, 0.087, 0.032]. min_p=0.5 means threshold =
        // 0.5 * 0.643 = 0.322. Only idx 0 survives.
        let mut logits = vec![4.0, 3.0, 2.0, 1.0];
        apply_min_p(0.5, &mut logits);
        assert_eq!(logits[0], 4.0);
        for l in &logits[1..] {
            assert!(l.is_infinite() && l.is_sign_negative());
        }
    }

    #[test]
    fn repetition_penalty_positive_logit_divides() {
        let mut logits = vec![2.0, -1.0, 3.0];
        apply_repetition_penalty(2.0, &[0, 1], &mut logits);
        assert_eq!(logits[0], 1.0); // 2.0 / 2.0 (positive divides)
        assert_eq!(logits[1], -2.0); // -1.0 * 2.0 (negative multiplies)
        assert_eq!(logits[2], 3.0); // untouched
    }

    #[test]
    fn freq_presence_penalty_counts_occurrences() {
        let mut logits = vec![5.0, 5.0, 5.0];
        apply_freq_presence_penalty(0.5, 1.0, &[0, 0, 1], &mut logits);
        // token 0: count=2, subtract 0.5*2 + 1.0 = 2.0 → 3.0
        // token 1: count=1, subtract 0.5*1 + 1.0 = 1.5 → 3.5
        // token 2: untouched
        assert_eq!(logits[0], 3.0);
        assert_eq!(logits[1], 3.5);
        assert_eq!(logits[2], 5.0);
    }

    #[test]
    fn logit_bias_adds_per_token() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let proc = LogitsProcessor::LogitBias(vec![(0, 0.5), (2, -1.0), (99, 100.0)]);
        proc.process(&[], &mut logits);
        assert_eq!(logits, vec![1.5, 2.0, 2.0]);
    }

    #[test]
    fn greedy_picks_argmax() {
        let logits = vec![1.0, 3.0, 2.0];
        let tok = sample_greedy(&logits);
        assert_eq!(tok.id, 1);
        // logprob = 3.0 - log_sum_exp([1,3,2]).
        let expected = 3.0f32 - (1.0f32.exp() + 3.0f32.exp() + 2.0f32.exp()).ln();
        assert!(approx_eq(tok.logprob, expected, 1e-5));
    }

    #[test]
    fn multinomial_is_reproducible_with_same_seed() {
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let mut s1 = Sampler::Multinomial {
            rng: SmallRng::seed_from_u64(42),
        };
        let mut s2 = Sampler::Multinomial {
            rng: SmallRng::seed_from_u64(42),
        };
        let draws_a: Vec<_> = (0..20).map(|_| s1.sample(&[], &logits).id).collect();
        let draws_b: Vec<_> = (0..20).map(|_| s2.sample(&[], &logits).id).collect();
        assert_eq!(draws_a, draws_b);
    }

    #[test]
    fn multinomial_distribution_approximates_softmax() {
        // Coarse check: with a sharp distribution, multinomial draws should
        // pick the high-probability token most of the time.
        let logits = vec![0.0, 5.0];
        let mut s = Sampler::Multinomial {
            rng: SmallRng::seed_from_u64(1234),
        };
        let n = 1000;
        let hits = (0..n).filter(|_| s.sample(&[], &logits).id == 1).count();
        // P(idx=1) ≈ e^5 / (1 + e^5) ≈ 0.993. Expect >950 hits / 1000.
        assert!(
            hits > 950,
            "expected idx=1 most of the time, got {hits}/{n}"
        );
    }

    #[test]
    fn stopping_eos_matches_token() {
        let crit = StoppingCriteria::Eos(vec![7, 13]);
        assert!(!crit.is_done(
            &[1, 2],
            SampledToken {
                id: 3,
                logprob: 0.0
            }
        ));
        assert!(crit.is_done(
            &[1, 2],
            SampledToken {
                id: 7,
                logprob: 0.0
            }
        ));
        assert!(crit.is_done(
            &[1, 2],
            SampledToken {
                id: 13,
                logprob: 0.0
            }
        ));
    }

    #[test]
    fn default_processors_includes_only_enabled_knobs() {
        let params = SamplingParams {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.1,
            ..SamplingParams::default()
        };
        let procs = default_processors(&params);
        let kinds: Vec<&'static str> = procs
            .iter()
            .map(|p| match p {
                LogitsProcessor::Temperature(_) => "T",
                LogitsProcessor::TopK(_) => "K",
                LogitsProcessor::TopP(_) => "P",
                LogitsProcessor::MinP(_) => "M",
                LogitsProcessor::RepetitionPenalty(_) => "R",
                LogitsProcessor::FreqPresencePenalty { .. } => "F",
                LogitsProcessor::LogitBias(_) => "B",
            })
            .collect();
        // Order: penalties → top-k → top-p → min-p → temperature.
        assert_eq!(kinds, vec!["R", "K", "P", "T"]);
    }
}
