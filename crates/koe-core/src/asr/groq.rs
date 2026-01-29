use std::sync::atomic::{AtomicU64, Ordering};

use serde::Deserialize;
use ureq::unversioned::multipart::{Form, Part};

use crate::{AsrError, AudioChunk, TranscriptSegment};

use super::{AsrProvider, encode_wav};

const GROQ_TRANSCRIPTIONS_URL: &str = "https://api.groq.com/openai/v1/audio/transcriptions";

/// Cloud ASR provider using the Groq Whisper API.
pub struct GroqProvider {
    api_key: String,
    segment_id: AtomicU64,
}

#[derive(Deserialize)]
struct GroqResponse {
    segments: Option<Vec<GroqSegment>>,
    text: String,
}

#[derive(Deserialize)]
struct GroqSegment {
    start: f64,
    end: f64,
    text: String,
}

impl GroqProvider {
    pub fn new() -> Result<Self, AsrError> {
        let api_key = std::env::var("GROQ_API_KEY")
            .map_err(|_| AsrError::ModelLoad("GROQ_API_KEY not set".into()))?;
        Ok(Self {
            api_key,
            segment_id: AtomicU64::new(0),
        })
    }
}

impl AsrProvider for GroqProvider {
    fn name(&self) -> &'static str {
        "groq"
    }

    fn transcribe(&mut self, chunk: &AudioChunk) -> Result<Vec<TranscriptSegment>, AsrError> {
        let wav_data = encode_wav(&chunk.pcm_mono_f32, chunk.sample_rate_hz);

        let form = Form::new()
            .text("model", "whisper-large-v3-turbo")
            .text("response_format", "verbose_json")
            .text("language", "en")
            .part(
                "file",
                Part::bytes(&wav_data)
                    .file_name("audio.wav")
                    .mime_str("audio/wav")
                    .map_err(|e| AsrError::Network(format!("{e}")))?,
            );

        let response = ureq::post(GROQ_TRANSCRIPTIONS_URL)
            .header("Authorization", &format!("Bearer {}", self.api_key))
            .send(form)
            .map_err(|e| AsrError::Network(format!("{e}")))?;

        let groq: GroqResponse = response
            .into_body()
            .read_json()
            .map_err(|e| AsrError::InvalidResponse(format!("{e}")))?;

        let base_ms = (chunk.start_pts_ns / 1_000_000) as i64;

        let segments = match groq.segments {
            Some(segs) => segs
                .into_iter()
                .filter_map(|s| {
                    let text = s.text.trim().to_owned();
                    if text.is_empty() {
                        return None;
                    }
                    let id = self.segment_id.fetch_add(1, Ordering::Relaxed);
                    Some(TranscriptSegment {
                        id,
                        start_ms: (s.start * 1000.0) as i64 + base_ms,
                        end_ms: (s.end * 1000.0) as i64 + base_ms,
                        speaker: None,
                        text,
                        finalized: false,
                    })
                })
                .collect(),
            None => {
                let text = groq.text.trim().to_owned();
                if text.is_empty() {
                    vec![]
                } else {
                    let id = self.segment_id.fetch_add(1, Ordering::Relaxed);
                    vec![TranscriptSegment {
                        id,
                        start_ms: base_ms,
                        end_ms: base_ms,
                        speaker: None,
                        text,
                        finalized: false,
                    }]
                }
            }
        };

        Ok(segments)
    }
}
