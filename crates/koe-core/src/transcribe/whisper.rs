use std::sync::atomic::{AtomicU64, Ordering};

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::{AudioChunk, TranscribeError, TranscriptSegment};

use super::TranscribeProvider;

/// Local transcribe provider using whisper.cpp via whisper-rs with Metal acceleration.
pub struct WhisperProvider {
    ctx: WhisperContext,
    segment_id: AtomicU64,
}

impl WhisperProvider {
    pub fn new(model_path: &str) -> Result<Self, TranscribeError> {
        let ctx = WhisperContext::new_with_params(model_path, WhisperContextParameters::new())
            .map_err(|e| TranscribeError::ModelLoad(format!("{e}")))?;
        Ok(Self {
            ctx,
            segment_id: AtomicU64::new(0),
        })
    }
}

impl TranscribeProvider for WhisperProvider {
    fn name(&self) -> &'static str {
        "whisper"
    }

    fn transcribe(
        &mut self,
        chunk: &AudioChunk,
    ) -> Result<Vec<TranscriptSegment>, TranscribeError> {
        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| TranscribeError::TranscribeFailed(format!("{e}")))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 5 });
        params.set_language(Some("en"));
        params.set_n_threads(4);
        params.set_print_progress(false);
        params.set_print_realtime(false);

        state
            .full(params, &chunk.pcm_mono_f32)
            .map_err(|e| TranscribeError::TranscribeFailed(format!("{e}")))?;

        let base_ms = (chunk.start_pts_ns / 1_000_000) as i64;
        let n_segments = state.full_n_segments();
        let mut segments = Vec::with_capacity(n_segments as usize);

        for i in 0..n_segments {
            let Some(seg) = state.get_segment(i) else {
                continue;
            };
            let text = match seg.to_str() {
                Ok(t) => t.trim().to_owned(),
                Err(e) => {
                    eprintln!("whisper: failed to decode segment {i} text: {e}");
                    continue;
                }
            };
            if text.is_empty() {
                continue;
            }
            // whisper timestamps are in centiseconds (10ms units)
            let start_ms = base_ms + seg.start_timestamp() * 10;
            let end_ms = base_ms + seg.end_timestamp() * 10;
            let id = self.segment_id.fetch_add(1, Ordering::Relaxed);
            segments.push(TranscriptSegment {
                id,
                start_ms,
                end_ms,
                speaker: None,
                text,
                finalized: false,
            });
        }

        Ok(segments)
    }
}
