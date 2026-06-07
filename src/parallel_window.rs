#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ContextWindowId(pub usize);

/// Fixed geometry of the shared position band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PositionBand {
    /// Shared prefix length
    prefix_len: usize,
    /// Fixed band width (maximum supported window length)
    band_width: usize,
}

impl PositionBand {
    pub fn new(prefix_len: usize, band_width: usize) -> Self {
        assert!(band_width > 0, "band_width must be non-zero");
        Self {
            prefix_len,
            band_width,
        }
    }

    pub fn prefix_len(&self) -> usize {
        self.prefix_len
    }

    pub fn band_width(&self) -> usize {
        self.band_width
    }

    /// RoPE position of the first token of a window of length `window_len`.
    pub fn window_start_position(&self, window_len: usize) -> usize {
        assert!(
            window_len <= self.band_width,
            "window_len ({}) exceeds band_width ({})",
            window_len,
            self.band_width,
        );
        self.prefix_len + self.band_width - window_len
    }

    pub fn window_end_position(&self) -> usize {
        self.prefix_len + self.band_width - 1
    }

    pub fn generation_position(&self, step: usize) -> usize {
        self.prefix_len + self.band_width + step
    }

    pub fn window_positions(&self, window_len: usize) -> Vec<usize> {
        let start = self.window_start_position(window_len);
        (start..start + window_len).collect()
    }
}

/// Metadata for one registered window: its length and where its slots live in
/// the shared KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextWindow {
    pub id: ContextWindowId,
    pub seq_len: usize,
    pub slot_start: usize,
}

impl ContextWindow {
    /// Half-open slot range `[slot_start, slot_start + seq_len)` in the cache.
    pub fn slot_range(&self) -> std::ops::Range<usize> {
        self.slot_start..self.slot_start + self.seq_len
    }
}

/// CPU-side registry of context windows plus the shared band geometry.
///
/// The cache is laid out as:
///
/// ```text
///     [ prefix slots: 0 .. P ) [ window 0 slots ] [ window 1 slots ] ...
/// ```
#[derive(Debug, Clone)]
pub struct WindowTable {
    band: PositionBand,
    windows: Vec<ContextWindow>,
    /// Next free slot in the cache (starts past the prefix).
    next_slot: usize,
}

impl WindowTable {
    pub fn new(band: PositionBand) -> Self {
        let prefix_len = band.prefix_len();
        Self {
            band,
            windows: Vec::new(),
            next_slot: prefix_len,
        }
    }

    pub fn band(&self) -> PositionBand {
        self.band
    }

    pub fn windows(&self) -> &[ContextWindow] {
        &self.windows
    }

    /// Look up a registered window by id.
    pub fn window(&self, id: ContextWindowId) -> Option<&ContextWindow> {
        self.windows.iter().find(|w| w.id == id)
    }

    /// Total number of slots currently used by prefix + all windows.
    pub fn total_slots(&self) -> usize {
        self.next_slot
    }

    /// Register a new window of length `seq_len`, assigning it the next id and
    /// the next contiguous slot range in the cache. Returns the registered
    /// [`ContextWindow`] (it is `Copy`, so the caller gets its id and row range
    /// without a follow-up lookup).
    ///
    /// Panics if `seq_len` exceeds the band width.
    pub fn register_window(&mut self, seq_len: usize) -> ContextWindow {
        assert!(
            seq_len <= self.band.band_width(),
            "window seq_len ({}) exceeds band_width ({})",
            seq_len,
            self.band.band_width(),
        );
        let id = ContextWindowId(self.windows.len());
        let entry = ContextWindow {
            id,
            seq_len,
            slot_start: self.next_slot,
        };
        self.next_slot += seq_len;
        self.windows.push(entry);
        entry
    }

    /// RoPE offset (first-token position) for a registered window.
    ///
    /// Returns [`UnknownWindow`] if `id` was never registered.
    pub fn window_rope_offset(&self, id: ContextWindowId) -> Result<usize, UnknownWindow> {
        let entry = self.window(id).ok_or(UnknownWindow(id))?;
        Ok(self.band.window_start_position(entry.seq_len))
    }

    /// Validate a visibility list: every id must be registered and appear at
    /// most once.
    fn validate_visibility(&self, visible: &[ContextWindowId]) -> Result<(), InvalidVisibility> {
        // Count occurrences while preserving first-appearance order.
        let mut counts: Vec<(ContextWindowId, usize)> = Vec::new();
        for &id in visible {
            match counts.iter_mut().find(|(cid, _)| *cid == id) {
                Some(slot) => slot.1 += 1,
                None => counts.push((id, 1)),
            }
        }
        let duplicates: Vec<(ContextWindowId, usize)> =
            counts.iter().copied().filter(|(_, n)| *n > 1).collect();
        let unknown: Vec<ContextWindowId> = counts
            .iter()
            .map(|(id, _)| *id)
            .filter(|id| self.window(*id).is_none())
            .collect();
        if duplicates.is_empty() && unknown.is_empty() {
            Ok(())
        } else {
            Err(InvalidVisibility {
                duplicates,
                unknown,
            })
        }
    }

    /// Build the ordered list of KV cache slots a generation attends to, given
    /// the windows it can see. The order is:
    ///
    /// 1. the shared prefix slots `[0, P)` (always visible), then
    /// 2. the slots of each visible window, in the order listed.
    pub fn visible_slots(
        &self,
        visible: &[ContextWindowId],
    ) -> Result<Vec<usize>, InvalidVisibility> {
        self.validate_visibility(visible)?;
        let mut slots: Vec<usize> = (0..self.band.prefix_len()).collect();
        for &id in visible {
            let entry = self.window(id).expect("validated by validate_visibility");
            slots.extend(entry.slot_range());
        }
        Ok(slots)
    }

    /// Build the ordered list of contiguous KV cache slot *ranges* a generation
    /// attends to, given the windows it can see. Like [`Self::visible_slots`]
    /// but keeps each contiguous block intact instead of expanding to individual
    /// slots — the form the masked-attention range list consumes:
    ///
    /// 1. the shared prefix range `0..P` (always visible), then
    /// 2. the slot range of each visible window, in the order listed.
    pub fn visible_ranges(
        &self,
        visible: &[ContextWindowId],
    ) -> Result<Vec<std::ops::Range<usize>>, InvalidVisibility> {
        self.validate_visibility(visible)?;
        let mut ranges: Vec<std::ops::Range<usize>> = Vec::with_capacity(1 + visible.len());
        ranges.push(0..self.band.prefix_len());
        for &id in visible {
            let entry = self.window(id).expect("validated by validate_visibility");
            ranges.push(entry.slot_range());
        }
        Ok(ranges)
    }

    /// RoPE position for each slot returned by [`Self::visible_slots`] with the
    /// same `visible` argument. Prefix slot `i` has position `i`; a window slot
    /// has its end-aligned band position. Useful for gather-based decode paths
    /// and for cross-checking the encoder's per-window offsets.
    pub fn visible_slot_positions(
        &self,
        visible: &[ContextWindowId],
    ) -> Result<Vec<usize>, InvalidVisibility> {
        self.validate_visibility(visible)?;
        let mut positions: Vec<usize> = (0..self.band.prefix_len()).collect();
        for &id in visible {
            let entry = self.window(id).expect("validated by validate_visibility");
            positions.extend(self.band.window_positions(entry.seq_len));
        }
        Ok(positions)
    }
}

/// A window id was not registered in the table.
///
/// Returned by [`WindowTable::window_rope_offset`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnknownWindow(pub ContextWindowId);

impl std::fmt::Display for UnknownWindow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "unknown window id {:?}", self.0)
    }
}

impl std::error::Error for UnknownWindow {}

/// A visibility list was malformed.
///
/// Returned by [`WindowTable::visible_slots`] and
/// [`WindowTable::visible_slot_positions`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidVisibility {
    /// Ids that appeared more than once, each paired with its occurrence count,
    /// in first-appearance order.
    pub duplicates: Vec<(ContextWindowId, usize)>,
    /// Ids that are not registered in the table, in first-appearance order.
    pub unknown: Vec<ContextWindowId>,
}

impl std::fmt::Display for InvalidVisibility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid window visibility:")?;
        if !self.duplicates.is_empty() {
            write!(f, " duplicated ids [")?;
            for (i, (id, count)) in self.duplicates.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{id:?} x{count}")?;
            }
            write!(f, "]")?;
        }
        if !self.unknown.is_empty() {
            write!(f, " unknown ids {:?}", self.unknown)?;
        }
        Ok(())
    }
}

impl std::error::Error for InvalidVisibility {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_are_end_aligned_to_shared_anchor() {
        let band = PositionBand::new(/* prefix */ 4, /* band */ 10);
        // Every window, whatever its length, ends at P + B - 1 = 13.
        assert_eq!(band.window_end_position(), 13);
        for len in 1..=10 {
            let positions = band.window_positions(len);
            assert_eq!(*positions.last().unwrap(), 13, "len {len} must end at 13");
            assert_eq!(positions.len(), len);
            assert_eq!(positions[0], band.window_start_position(len));
            // contiguous, ascending
            for w in positions.windows(2) {
                assert_eq!(w[1], w[0] + 1);
            }
        }
    }

    #[test]
    fn generation_sees_every_window_tail_at_equal_offset() {
        let band = PositionBand::new(4, 10);
        let end = band.window_end_position(); // 13
        for step in 0..5 {
            let gen_pos = band.generation_position(step);
            assert_eq!(gen_pos, 14 + step);
            // Relative distance from generation token to any window's last token
            // is the same regardless of window length: t + 1.
            assert_eq!(gen_pos - end, step + 1);
        }
    }

    #[test]
    fn short_and_long_windows_share_tail_distance() {
        let band = PositionBand::new(0, 8);
        let short = band.window_positions(2); // [6, 7]
        let long = band.window_positions(8); // [0..8)
        assert_eq!(*short.last().unwrap(), *long.last().unwrap());
        let gen_pos = band.generation_position(0);
        assert_eq!(
            gen_pos - short.last().unwrap(),
            gen_pos - long.last().unwrap()
        );
    }

    #[test]
    fn slot_layout_is_prefix_then_windows() {
        let mut table = WindowTable::new(PositionBand::new(3, 16));
        assert_eq!(table.total_slots(), 3); // prefix occupies first slots
        let a = table.register_window(5);
        let b = table.register_window(2);
        assert_eq!(a.id, ContextWindowId(0));
        assert_eq!(b.id, ContextWindowId(1));
        assert_eq!(a.slot_range(), 3..8);
        assert_eq!(b.slot_range(), 8..10);
        assert_eq!(table.total_slots(), 10);
    }

    #[test]
    fn window_rope_offset_is_end_aligned_start() {
        let mut table = WindowTable::new(PositionBand::new(4, 10));
        let a = table.register_window(10); // full-width window
        let b = table.register_window(3); // short window
        assert_eq!(table.window_rope_offset(a.id).unwrap(), 4); // P + B - L = 4 + 10 - 10
        assert_eq!(table.window_rope_offset(b.id).unwrap(), 11); // 4 + 10 - 3
    }

    #[test]
    fn visible_slots_are_prefix_plus_selected_windows() {
        let mut table = WindowTable::new(PositionBand::new(2, 16));
        let a = table.register_window(3); // slots 2..5
        let _b = table.register_window(4); // slots 5..9
        let c = table.register_window(1); // slots 9..10

        // Generation that can see A and C only.
        let slots = table.visible_slots(&[a.id, c.id]).unwrap();
        assert_eq!(slots, vec![0, 1, /*A*/ 2, 3, 4, /*C*/ 9]);
    }

    #[test]
    fn visible_slot_positions_match_band_positions() {
        let mut table = WindowTable::new(PositionBand::new(2, 8));
        let a = table.register_window(3); // positions [P+B-3 .. ] = [7,8,9]
        let c = table.register_window(1); // position [9]
        let positions = table.visible_slot_positions(&[a.id, c.id]).unwrap();
        // prefix positions 0,1 then window A [7,8,9] then window C [9]
        assert_eq!(positions, vec![0, 1, 7, 8, 9, 9]);
    }
    #[test]
    fn duplicate_visibility_reports_id_and_count() {
        let mut table = WindowTable::new(PositionBand::new(0, 8));
        let a = table.register_window(2);
        let b = table.register_window(2);
        let err = table
            .visible_slots(&[a.id, b.id, a.id, a.id])
            .expect_err("duplicate ids must error");
        assert_eq!(
            err,
            InvalidVisibility {
                duplicates: vec![(a.id, 3)],
                unknown: vec![],
            },
        );
    }

    #[test]
    fn unknown_visibility_reports_missing_ids() {
        let mut table = WindowTable::new(PositionBand::new(0, 8));
        let a = table.register_window(2);
        let missing = ContextWindowId(7);
        let err = table
            .visible_slots(&[a.id, missing])
            .expect_err("unknown id must error");
        assert_eq!(
            err,
            InvalidVisibility {
                duplicates: vec![],
                unknown: vec![missing],
            },
        );
    }

    #[test]
    fn visibility_reports_duplicates_and_unknowns_together() {
        let mut table = WindowTable::new(PositionBand::new(0, 8));
        let a = table.register_window(2);
        let missing = ContextWindowId(9);
        let err = table
            .visible_slots(&[a.id, a.id, missing, missing])
            .expect_err("malformed visibility must error");
        assert_eq!(
            err,
            InvalidVisibility {
                duplicates: vec![(a.id, 2), (missing, 2)],
                unknown: vec![missing],
            },
        );
    }

    #[test]
    fn window_rope_offset_rejects_unknown_id() {
        let table = WindowTable::new(PositionBand::new(0, 8));
        let err = table
            .window_rope_offset(ContextWindowId(0))
            .expect_err("unknown id must error");
        assert_eq!(err, UnknownWindow(ContextWindowId(0)));
    }
    #[test]
    #[should_panic(expected = "exceeds band_width")]
    fn oversized_window_panics() {
        let mut table = WindowTable::new(PositionBand::new(0, 4));
        let _ = table.register_window(5);
    }
}
