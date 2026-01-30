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

/// Structured meeting notes state.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MeetingState {
    pub key_points: Vec<NoteItem>,
    pub actions: Vec<ActionItem>,
    pub decisions: Vec<NoteItem>,
}

/// A single note item (key point or decision).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteItem {
    pub id: String,
    pub text: String,
    pub evidence: Vec<u64>,
}

/// An action item with optional owner and due date.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionItem {
    pub id: String,
    pub text: String,
    pub owner: Option<String>,
    pub due: Option<String>,
    pub evidence: Vec<u64>,
}

/// A patch operation on the meeting notes state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotesOp {
    AddKeyPoint {
        id: String,
        text: String,
        evidence: Vec<u64>,
    },
    AddAction {
        id: String,
        text: String,
        owner: Option<String>,
        due: Option<String>,
        evidence: Vec<u64>,
    },
    AddDecision {
        id: String,
        text: String,
        evidence: Vec<u64>,
    },
    UpdateAction {
        id: String,
        owner: Option<String>,
        due: Option<String>,
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
}

impl CaptureStats {
    pub fn new() -> Self {
        Self {
            frames_captured: Arc::new(AtomicU64::new(0)),
            frames_dropped: Arc::new(AtomicU64::new(0)),
            chunks_emitted: Arc::new(AtomicU64::new(0)),
            chunks_dropped: Arc::new(AtomicU64::new(0)),
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
}

impl Default for CaptureStats {
    fn default() -> Self {
        Self::new()
    }
}
