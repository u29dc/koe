use crate::SummarizeError;
use crate::types::{MeetingState, SummarizeEvent, TranscriptSegment};
use serde::Deserialize;
use serde_json::json;

use super::{SummarizeProvider, patch};

const DEFAULT_BASE_URL: &str = "http://localhost:11434";

pub struct OllamaProvider {
    model: String,
    base_url: String,
}

impl OllamaProvider {
    pub fn new(model: &str) -> Result<Self, SummarizeError> {
        let base_url = std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.into());
        Ok(Self {
            model: model.to_string(),
            base_url,
        })
    }
}

impl SummarizeProvider for OllamaProvider {
    fn name(&self) -> &'static str {
        "ollama"
    }

    fn summarize(
        &mut self,
        recent_segments: &[TranscriptSegment],
        state: &MeetingState,
        context: Option<&str>,
        participants: &[String],
        on_event: &mut dyn FnMut(SummarizeEvent),
    ) -> Result<(), SummarizeError> {
        let prompt = patch::build_prompt(recent_segments, state, context, participants);
        let url = format!("{}/api/generate", self.base_url);
        let body = json!({
            "model": self.model,
            "prompt": prompt,
            "stream": true,
        });

        let response = ureq::post(&url)
            .send_json(body)
            .map_err(|e| SummarizeError::Network(format!("{e}")))?;

        let raw = response
            .into_body()
            .read_to_string()
            .map_err(|e| SummarizeError::Network(format!("{e}")))?;

        let mut full_text = String::new();
        for line in raw.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let chunk: OllamaChunk = serde_json::from_str(line)
                .map_err(|e| SummarizeError::InvalidResponse(e.to_string()))?;
            if let Some(token) = chunk.response {
                on_event(SummarizeEvent::DraftToken(token.clone()));
                full_text.push_str(&token);
            }
            if chunk.done.unwrap_or(false) {
                break;
            }
        }

        let patch = patch::parse_patch(full_text.trim())?;
        on_event(SummarizeEvent::PatchReady(patch));
        Ok(())
    }
}

#[derive(Deserialize)]
struct OllamaChunk {
    response: Option<String>,
    done: Option<bool>,
}
