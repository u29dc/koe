use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// A single audio frame from a capture source.
pub struct AudioFrame {
    pub pts_ns: i128,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub samples_f32: Vec<f32>,
}

/// Identifies the origin of an audio stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioSource {
    System,
    Microphone,
    Mixed,
}

/// A speech-gated chunk of resampled 16 kHz mono PCM ready for transcription.
pub struct AudioChunk {
    pub source: AudioSource,
    pub start_pts_ns: i128,
    pub sample_rate_hz: u32,
    pub pcm_mono_f32: Vec<f32>,
}

/// A single transcript segment produced by transcription.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptSegment {
    pub id: u64,
    pub start_ms: i64,
    pub end_ms: i64,
    pub speaker: Option<String>,
    pub text: String,
    pub finalized: bool,
}

/// Rolling meeting notes as a flat bullet stream.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MeetingNotes {
    pub bullets: Vec<NoteBullet>,
}

/// A single bullet note.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteBullet {
    pub id: String,
    pub text: String,
    pub evidence: Vec<u64>,
}

/// A patch operation on the meeting notes state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotesOp {
    Add {
        id: String,
        text: String,
        evidence: Vec<u64>,
    },
}

/// A batch of note operations to apply atomically.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotesPatch {
    pub ops: Vec<NotesOp>,
}

/// Events emitted by a summarize provider during streaming.
pub enum SummarizeEvent {
    DraftToken(String),
    PatchReady(NotesPatch),
}

/// Atomic counters for capture pipeline statistics.
#[derive(Debug, Clone)]
pub struct CaptureStats {
    pub frames_captured: Arc<AtomicU64>,
    pub frames_dropped: Arc<AtomicU64>,
    pub chunks_emitted: Arc<AtomicU64>,
    pub chunks_dropped: Arc<AtomicU64>,
    pub raw_frames_dropped: Arc<AtomicU64>,
}

impl CaptureStats {
    pub fn new() -> Self {
        Self {
            frames_captured: Arc::new(AtomicU64::new(0)),
            frames_dropped: Arc::new(AtomicU64::new(0)),
            chunks_emitted: Arc::new(AtomicU64::new(0)),
            chunks_dropped: Arc::new(AtomicU64::new(0)),
            raw_frames_dropped: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn inc_frames_captured(&self) {
        self.frames_captured.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_frames_dropped(&self) {
        self.frames_dropped.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_chunks_emitted(&self) {
        self.chunks_emitted.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_chunks_dropped(&self) {
        self.chunks_dropped.fetch_add(1, Ordering::Relaxed);
    }

    pub fn inc_raw_frames_dropped(&self) {
        self.raw_frames_dropped.fetch_add(1, Ordering::Relaxed);
    }

    pub fn frames_captured(&self) -> u64 {
        self.frames_captured.load(Ordering::Relaxed)
    }

    pub fn frames_dropped(&self) -> u64 {
        self.frames_dropped.load(Ordering::Relaxed)
    }

    pub fn chunks_emitted(&self) -> u64 {
        self.chunks_emitted.load(Ordering::Relaxed)
    }

    pub fn chunks_dropped(&self) -> u64 {
        self.chunks_dropped.load(Ordering::Relaxed)
    }

    pub fn raw_frames_dropped(&self) -> u64 {
        self.raw_frames_dropped.load(Ordering::Relaxed)
    }
}

impl Default for CaptureStats {
    fn default() -> Self {
        Self::new()
    }
}
