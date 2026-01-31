use crate::TranscriptSegment;

const MUTABLE_WINDOW_MS: i64 = 15_000;
const SIMILARITY_THRESHOLD: f64 = 0.5;
const MAX_SEGMENTS: usize = 2_000;

/// Ordered ledger of transcript segments with overlap-aware deduplication.
///
/// The audio chunker retains a 1s overlap between consecutive emits, so
/// consecutive transcribe calls may produce segments covering the same audio.
/// `append` merges incoming segments against existing ones using temporal
/// overlap and text similarity to decide whether to replace or keep both.
pub struct TranscriptLedger {
    segments: Vec<TranscriptSegment>,
    highest_end_ms: i64,
}

impl TranscriptLedger {
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
            highest_end_ms: 0,
        }
    }

    /// Merge new transcription output into the ledger, deduplicating overlaps and
    /// finalizing old segments that fall outside the overlap window.
    pub fn append(&mut self, mut incoming: Vec<TranscriptSegment>) {
        incoming.sort_by_key(|s| s.start_ms);

        for seg in incoming {
            if seg.end_ms > self.highest_end_ms {
                self.highest_end_ms = seg.end_ms;
            }

            if self
                .segments
                .iter()
                .any(|existing| existing.finalized && overlaps(existing, &seg))
            {
                continue;
            }

            let mut replaced = false;
            for existing in self.segments.iter_mut() {
                if existing.finalized {
                    continue;
                }
                if overlaps(existing, &seg)
                    && text_similarity(&existing.text, &seg.text) >= SIMILARITY_THRESHOLD
                {
                    // Newer segment has more context -- replace the old one.
                    *existing = seg.clone();
                    replaced = true;
                    break;
                }
            }

            if !replaced {
                // Insert maintaining sort order by start_ms.
                let pos = self
                    .segments
                    .partition_point(|s| s.start_ms <= seg.start_ms);
                self.segments.insert(pos, seg);
            }
        }

        // Finalize segments that are safely behind the overlap window.
        let cutoff = self.highest_end_ms - MUTABLE_WINDOW_MS;
        for seg in &mut self.segments {
            if seg.end_ms < cutoff {
                seg.finalized = true;
            }
        }

        self.prune_finalized(MAX_SEGMENTS);
    }

    /// Full transcript read.
    pub fn segments(&self) -> &[TranscriptSegment] {
        &self.segments
    }

    /// Latest end timestamp across all segments.
    pub fn highest_end_ms(&self) -> i64 {
        self.highest_end_ms
    }

    /// Incremental read: segments with `id > since_id`.
    pub fn segments_since(&self, since_id: u64) -> &[TranscriptSegment] {
        match self.segments.iter().position(|s| s.id > since_id) {
            Some(pos) => &self.segments[pos..],
            None => &[],
        }
    }

    /// Tail slice for summarize context window.
    pub fn last_n_segments(&self, n: usize) -> &[TranscriptSegment] {
        let start = self.segments.len().saturating_sub(n);
        &self.segments[start..]
    }

    /// Segment count.
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// Whether the ledger is empty.
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    fn prune_finalized(&mut self, max_segments: usize) {
        if self.segments.len() <= max_segments {
            return;
        }

        let mut keep = Vec::with_capacity(max_segments);
        for seg in &self.segments {
            if !seg.finalized {
                keep.push(seg.clone());
            }
        }

        let remaining = max_segments.saturating_sub(keep.len());
        if remaining == 0 {
            self.segments = keep;
            return;
        }

        let finalized: Vec<_> = self.segments.iter().filter(|seg| seg.finalized).collect();
        let start = finalized.len().saturating_sub(remaining);
        for seg in finalized[start..].iter() {
            keep.push((*seg).clone());
        }

        keep.sort_by_key(|seg| seg.start_ms);
        self.segments = keep;
    }
}

impl Default for TranscriptLedger {
    fn default() -> Self {
        Self::new()
    }
}

/// Two segments overlap if their time ranges intersect.
fn overlaps(a: &TranscriptSegment, b: &TranscriptSegment) -> bool {
    a.start_ms <= b.end_ms && b.start_ms <= a.end_ms
}

/// Fast text similarity based on containment and longest common prefix/suffix.
fn text_similarity(a: &str, b: &str) -> f64 {
    let na = a.to_lowercase();
    let nb = b.to_lowercase();
    let shorter = na.len().min(nb.len());
    if shorter == 0 {
        return 0.0;
    }
    if na.contains(&nb) || nb.contains(&na) {
        return 1.0;
    }
    let prefix = longest_common_prefix(na.as_bytes(), nb.as_bytes());
    let suffix = longest_common_suffix(na.as_bytes(), nb.as_bytes());
    prefix.max(suffix) as f64 / shorter as f64
}

fn longest_common_prefix(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

fn longest_common_suffix(a: &[u8], b: &[u8]) -> usize {
    a.iter()
        .rev()
        .zip(b.iter().rev())
        .take_while(|(x, y)| x == y)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(id: u64, start: i64, end: i64, text: &str) -> TranscriptSegment {
        TranscriptSegment {
            id,
            start_ms: start,
            end_ms: end,
            speaker: None,
            text: text.to_string(),
            finalized: false,
        }
    }

    #[test]
    fn append_to_empty() {
        let mut ledger = TranscriptLedger::new();
        ledger.append(vec![seg(2, 200, 400, "second"), seg(1, 0, 200, "first")]);
        assert_eq!(ledger.len(), 2);
        assert_eq!(ledger.segments()[0].text, "first");
        assert_eq!(ledger.segments()[1].text, "second");
    }

    #[test]
    fn non_overlapping_kept() {
        let mut ledger = TranscriptLedger::new();
        ledger.append(vec![seg(1, 0, 100, "hello")]);
        ledger.append(vec![seg(2, 200, 300, "world")]);
        assert_eq!(ledger.len(), 2);
    }

    #[test]
    fn exact_duplicate_replaced() {
        let mut ledger = TranscriptLedger::new();
        ledger.append(vec![seg(1, 0, 100, "hello world")]);
        ledger.append(vec![seg(2, 0, 100, "hello world")]);
        assert_eq!(ledger.len(), 1);
        assert_eq!(ledger.segments()[0].id, 2);
    }

    #[test]
    fn prefix_overlap_merged() {
        let mut ledger = TranscriptLedger::new();
        ledger.append(vec![seg(1, 0, 100, "the quick brown")]);
        // Overlapping time range with text that shares a long prefix.
        ledger.append(vec![seg(2, 50, 200, "the quick brown fox")]);
        assert_eq!(ledger.len(), 1);
        assert_eq!(ledger.segments()[0].text, "the quick brown fox");
    }

    #[test]
    fn dissimilar_overlap_kept() {
        let mut ledger = TranscriptLedger::new();
        ledger.append(vec![seg(1, 0, 100, "hello world")]);
        // Same time range but completely different text.
        ledger.append(vec![seg(2, 50, 150, "goodbye moon")]);
        assert_eq!(ledger.len(), 2);
    }

    #[test]
    fn finalization_after_window() {
        let mut ledger = TranscriptLedger::new();
        ledger.append(vec![seg(1, 0, 100, "old segment")]);
        // Push highest_end_ms far enough that the first segment is finalized.
        ledger.append(vec![seg(2, 20_000, 21_000, "new segment")]);
        assert!(ledger.segments()[0].finalized);
        assert!(!ledger.segments()[1].finalized);
    }

    #[test]
    fn finalized_segments_ignore_overlaps() {
        let mut ledger = TranscriptLedger::new();
        ledger.append(vec![seg(1, 0, 100, "old segment")]);
        ledger.append(vec![seg(2, 20_000, 21_000, "new segment")]);
        assert!(ledger.segments()[0].finalized);

        ledger.append(vec![seg(3, 50, 150, "old segment")]);
        assert_eq!(ledger.len(), 2);
        assert_eq!(ledger.segments()[0].id, 1);
    }

    #[test]
    fn segments_since_filters() {
        let mut ledger = TranscriptLedger::new();
        ledger.append(vec![
            seg(1, 0, 100, "a"),
            seg(2, 100, 200, "b"),
            seg(3, 200, 300, "c"),
        ]);
        let since = ledger.segments_since(1);
        assert_eq!(since.len(), 2);
        assert_eq!(since[0].id, 2);
        assert_eq!(since[1].id, 3);
    }

    #[test]
    fn last_n_clamps() {
        let mut ledger = TranscriptLedger::new();
        ledger.append(vec![seg(1, 0, 100, "a"), seg(2, 100, 200, "b")]);
        assert_eq!(ledger.last_n_segments(5).len(), 2);
        assert_eq!(ledger.last_n_segments(1).len(), 1);
        assert_eq!(ledger.last_n_segments(1)[0].id, 2);
    }

    #[test]
    fn text_similarity_cases() {
        assert_eq!(text_similarity("", "hello"), 0.0);
        assert_eq!(text_similarity("hello", "hello"), 1.0);
        assert_eq!(text_similarity("hello world", "Hello"), 1.0); // containment
        assert!(text_similarity("abcdef", "abcxyz") > 0.4); // prefix 3/6 = 0.5
        assert!(text_similarity("xyzabc", "qqqabc") > 0.4); // suffix 3/6 = 0.5
        assert!(text_similarity("hello", "goodbye") < SIMILARITY_THRESHOLD);
    }

    #[test]
    fn prunes_old_finalized_segments() {
        let mut ledger = TranscriptLedger::new();
        let mut segments = Vec::new();
        for i in 0..(MAX_SEGMENTS as u64 + 50) {
            segments.push(seg(i, i as i64 * 10, i as i64 * 10 + 10, "hello"));
        }
        ledger.append(segments);
        ledger.append(vec![seg(9999, 1_000_000, 1_000_010, "new")]);
        assert!(ledger.len() <= MAX_SEGMENTS + 1);
    }
}
