use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

use serde::Deserialize;
use ureq::unversioned::multipart::{Form, Part};

use crate::http::{default_agent, retry_delay, should_retry};
use crate::{AudioChunk, TranscribeError, TranscriptSegment};

use super::{TranscribeProvider, encode_wav};

const GROQ_TRANSCRIPTIONS_URL: &str = "https://api.groq.com/openai/v1/audio/transcriptions";
const DEFAULT_MODEL: &str = "whisper-large-v3-turbo";
const MAX_RETRIES: usize = 2;

/// Cloud transcribe provider using the Groq Whisper API.
pub struct GroqProvider {
    api_key: String,
    model: String,
    segment_id: AtomicU64,
    agent: ureq::Agent,
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
    pub fn new(model: Option<&str>, api_key: Option<&str>) -> Result<Self, TranscribeError> {
        let api_key = api_key
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| TranscribeError::ModelLoad("cloud API key not set".into()))?
            .to_string();
        Ok(Self {
            api_key,
            model: model.unwrap_or(DEFAULT_MODEL).to_owned(),
            segment_id: AtomicU64::new(0),
            agent: default_agent(),
        })
    }
}

impl TranscribeProvider for GroqProvider {
    fn name(&self) -> &'static str {
        "groq"
    }

    fn transcribe(
        &mut self,
        chunk: &AudioChunk,
    ) -> Result<Vec<TranscriptSegment>, TranscribeError> {
        let wav_data = encode_wav(&chunk.pcm_mono_f32, chunk.sample_rate_hz);

        let mut last_error: Option<ureq::Error> = None;
        let mut groq: Option<GroqResponse> = None;

        for attempt in 0..=MAX_RETRIES {
            let form = Form::new()
                .text("model", self.model.as_str())
                .text("response_format", "verbose_json")
                .text("language", "en")
                .part(
                    "file",
                    Part::bytes(&wav_data)
                        .file_name("audio.wav")
                        .mime_str("audio/wav")
                        .map_err(|e| TranscribeError::Network(format!("{e}")))?,
                );

            let response = self
                .agent
                .post(GROQ_TRANSCRIPTIONS_URL)
                .header("Authorization", &format!("Bearer {}", self.api_key))
                .send(form);

            match response {
                Ok(resp) => {
                    let payload: GroqResponse = resp
                        .into_body()
                        .read_json()
                        .map_err(|e| TranscribeError::InvalidResponse(format!("{e}")))?;
                    groq = Some(payload);
                    break;
                }
                Err(err) => {
                    let retry = should_retry(&err);
                    last_error = Some(err);
                    if retry && attempt < MAX_RETRIES {
                        thread::sleep(retry_delay(attempt));
                        continue;
                    }
                    return Err(TranscribeError::Network(format!("{}", last_error.unwrap())));
                }
            }
        }

        let groq = groq.ok_or_else(|| {
            TranscribeError::Network(
                last_error
                    .map(|err| err.to_string())
                    .unwrap_or_else(|| "groq request failed".to_string()),
            )
        })?;

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
