use crate::types::{AudioChunk, AudioSource};

const SAMPLE_RATE: u32 = 16_000;
const MIN_SAMPLES: usize = 32_000; // 2 s
const TARGET_SAMPLES: usize = 64_000; // 4 s
const MAX_SAMPLES: usize = 96_000; // 6 s
const OVERLAP_SAMPLES: usize = 16_000; // 1 s

/// Accumulates 16 kHz mono samples and emits speech-gated chunks with overlap.
pub struct Chunker {
    buffer: Vec<f32>,
    start_pts_ns: i128,
    was_speech: bool,
    source: AudioSource,
}

impl Chunker {
    pub fn new(source: AudioSource) -> Self {
        Self {
            buffer: Vec::with_capacity(MAX_SAMPLES),
            start_pts_ns: 0,
            was_speech: false,
            source,
        }
    }

    /// Push resampled 16 kHz samples with current VAD speech state.
    /// Returns a chunk when emission criteria are met.
    pub fn push(&mut self, samples: &[f32], pts_ns: i128, speech: bool) -> Option<AudioChunk> {
        if self.buffer.is_empty() {
            self.start_pts_ns = pts_ns;
        }
        self.buffer.extend_from_slice(samples);

        let speech_to_silence = self.was_speech && !speech;
        self.was_speech = speech;

        // Emit when: buffer >= target AND speech->silence, OR buffer >= max
        let should_emit = (self.buffer.len() >= TARGET_SAMPLES && speech_to_silence)
            || self.buffer.len() >= MAX_SAMPLES;

        if should_emit && self.buffer.len() >= MIN_SAMPLES {
            Some(self.emit())
        } else {
            None
        }
    }

    /// Flush remaining samples (e.g. on stop). Emits if buffer >= min.
    pub fn flush(&mut self) -> Option<AudioChunk> {
        if self.buffer.len() >= MIN_SAMPLES {
            Some(self.emit())
        } else if !self.buffer.is_empty() {
            // Emit even short buffers on flush to avoid data loss
            Some(self.emit())
        } else {
            None
        }
    }

    fn emit(&mut self) -> AudioChunk {
        let chunk = AudioChunk {
            source: self.source,
            start_pts_ns: self.start_pts_ns,
            sample_rate_hz: SAMPLE_RATE,
            pcm_mono_f32: self.buffer.clone(),
        };

        // Retain overlap
        if self.buffer.len() > OVERLAP_SAMPLES {
            let retain_start = self.buffer.len() - OVERLAP_SAMPLES;
            self.buffer.drain(..retain_start);
            // Advance PTS past the emitted (non-overlap) portion
            self.start_pts_ns += ((chunk.pcm_mono_f32.len() as i128 - OVERLAP_SAMPLES as i128)
                * 1_000_000_000)
                / SAMPLE_RATE as i128;
        } else {
            self.buffer.clear();
        }

        chunk
    }

    pub fn buffered_samples(&self) -> usize {
        self.buffer.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_emit_below_min() {
        let mut chunker = Chunker::new(AudioSource::System);
        let samples = vec![0.0f32; MIN_SAMPLES - 1];
        assert!(chunker.push(&samples, 0, true).is_none());
    }

    #[test]
    fn emit_at_target_on_speech_to_silence() {
        let mut chunker = Chunker::new(AudioSource::System);

        // Fill to target with speech
        let samples = vec![0.1f32; TARGET_SAMPLES];
        assert!(chunker.push(&samples, 0, true).is_none());

        // Speech -> silence triggers emit
        let more = vec![0.0f32; 512];
        let chunk = chunker.push(&more, 1_000_000, false);
        assert!(chunk.is_some());
        let chunk = chunk.unwrap();
        assert_eq!(chunk.pcm_mono_f32.len(), TARGET_SAMPLES + 512);
    }

    #[test]
    fn force_emit_at_max() {
        let mut chunker = Chunker::new(AudioSource::Microphone);

        // Fill to max while in speech (no speech->silence transition)
        let samples = vec![0.1f32; MAX_SAMPLES];
        let chunk = chunker.push(&samples, 0, true);
        assert!(chunk.is_some());
        assert_eq!(chunk.unwrap().pcm_mono_f32.len(), MAX_SAMPLES);
    }

    #[test]
    fn overlap_retained_after_emit() {
        let mut chunker = Chunker::new(AudioSource::System);

        let samples = vec![0.1f32; MAX_SAMPLES];
        chunker.push(&samples, 0, true);

        // After emit, buffer should contain overlap
        assert_eq!(chunker.buffered_samples(), OVERLAP_SAMPLES);
    }

    #[test]
    fn flush_emits_remaining() {
        let mut chunker = Chunker::new(AudioSource::System);
        let samples = vec![0.1f32; 5000];
        chunker.push(&samples, 0, true);
        assert!(chunker.flush().is_some());
        assert_eq!(chunker.buffered_samples(), 0);
    }

    #[test]
    fn flush_empty_returns_none() {
        let mut chunker = Chunker::new(AudioSource::System);
        assert!(chunker.flush().is_none());
    }

    #[test]
    fn no_emit_on_silence_to_silence() {
        let mut chunker = Chunker::new(AudioSource::System);
        let samples = vec![0.0f32; TARGET_SAMPLES + 100];
        // Continuous silence, no speech->silence transition, below max
        assert!(chunker.push(&samples, 0, false).is_none());
    }

    #[test]
    fn chunk_source_preserved() {
        let mut chunker = Chunker::new(AudioSource::Microphone);
        let samples = vec![0.1f32; MAX_SAMPLES];
        let chunk = chunker.push(&samples, 0, true).unwrap();
        assert_eq!(chunk.source, AudioSource::Microphone);
        assert_eq!(chunk.sample_rate_hz, 16_000);
    }
}
