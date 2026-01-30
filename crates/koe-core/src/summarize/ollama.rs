use crate::SummarizeError;
use crate::http::{default_agent, retry_delay, should_retry};
use crate::types::{MeetingState, SummarizeEvent, TranscriptSegment};
use serde::Deserialize;
use serde_json::json;
use std::thread;

use super::{SummarizeProvider, patch};

const DEFAULT_BASE_URL: &str = "http://localhost:11434";
const MAX_RETRIES: usize = 2;

pub struct OllamaProvider {
    model: String,
    base_url: String,
    agent: ureq::Agent,
}

impl OllamaProvider {
    pub fn new(model: &str) -> Result<Self, SummarizeError> {
        let base_url = std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.into());
        Ok(Self {
            model: model.to_string(),
            base_url,
            agent: default_agent(),
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
        let mut last_error: Option<ureq::Error> = None;
        let mut raw_body: Option<String> = None;

        for attempt in 0..=MAX_RETRIES {
            let body = json!({
                "model": self.model,
                "prompt": prompt,
                "stream": true,
            });

            let response = self.agent.post(&url).send_json(body);

            match response {
                Ok(resp) => {
                    let raw = resp
                        .into_body()
                        .read_to_string()
                        .map_err(|e| SummarizeError::Network(format!("{e}")))?;
                    raw_body = Some(raw);
                    break;
                }
                Err(err) => {
                    let retry = should_retry(&err);
                    last_error = Some(err);
                    if retry && attempt < MAX_RETRIES {
                        thread::sleep(retry_delay(attempt));
                        continue;
                    }
                    return Err(SummarizeError::Network(format!("{}", last_error.unwrap())));
                }
            }
        }

        let raw = raw_body.ok_or_else(|| {
            SummarizeError::Network(
                last_error
                    .map(|err| err.to_string())
                    .unwrap_or_else(|| "ollama request failed".to_string()),
            )
        })?;

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
