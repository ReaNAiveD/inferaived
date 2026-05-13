pub struct ArgmaxSamplerCpu;

impl ArgmaxSamplerCpu {
    pub fn new() -> Self {
        Self
    }

    /// Returns up to `top_k` `(token_id, logit)` pairs sorted by descending
    /// logit. `top_k` is clamped to `logits.len()`. Returns an empty vec when
    /// `top_k == 0` or `logits` is empty.
    pub fn sample(&self, logits: &[f32], top_k: usize) -> Vec<(usize, f32)> {
        let k = top_k.min(logits.len());
        if k == 0 {
            return Vec::new();
        }
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        // O(N) partition: the k largest end up in indexed[..k], unordered within.
        // `total_cmp` is NaN-safe, unlike `partial_cmp`.
        indexed.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
        indexed.truncate(k);
        // O(k log k) sort just the survivors, descending by logit.
        indexed.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        indexed
    }
}