use thiserror::Error;

/// Errors from audio capture backends.
#[derive(Debug, Error)]
pub enum CaptureError {
    #[error("screen recording permission denied")]
    PermissionDenied,

    #[error("no display available for capture")]
    NoDisplay,

    #[error("capture configuration failed: {0}")]
    ConfigFailed(String),

    #[error("capture start failed: {0}")]
    StartFailed(String),

    #[error("capture stop failed: {0}")]
    StopFailed(String),

    #[error("capture backend error: {0}")]
    Backend(String),
}

/// Errors from audio processing (resampling, VAD, chunking).
#[derive(Debug, Error)]
pub enum ProcessError {
    #[error("resampler initialization failed: {0}")]
    ResamplerInit(String),

    #[error("VAD initialization failed: {0}")]
    VadInit(String),

    #[error("resample failed: {0}")]
    ResampleFailed(String),

    #[error("capture error: {0}")]
    Capture(#[from] CaptureError),
}

/// Errors from transcribe providers.
#[derive(Debug, Error)]
pub enum TranscribeError {
    #[error("model load failed: {0}")]
    ModelLoad(String),

    #[error("transcription failed: {0}")]
    TranscribeFailed(String),

    #[error("network error: {0}")]
    Network(String),

    #[error("invalid response: {0}")]
    InvalidResponse(String),
}

/// Errors from summarize providers.
#[derive(Debug, Error)]
pub enum SummarizeError {
    #[error("summarize failed: {0}")]
    Failed(String),

    #[error("network error: {0}")]
    Network(String),

    #[error("invalid response: {0}")]
    InvalidResponse(String),
}
