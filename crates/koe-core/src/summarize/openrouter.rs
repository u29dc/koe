use crate::SummarizeError;
use crate::http::{default_agent, retry_delay, should_retry};
use crate::types::{MeetingState, SummarizeEvent, TranscriptSegment};
use serde::Deserialize;
use serde_json::json;
use std::thread;

use super::{SummarizeProvider, patch};

const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";
const DEFAULT_MODEL: &str = "google/gemini-2.5-flash";
const SYSTEM_PROMPT: &str =
    "You are a meeting notes engine. Follow the instructions and output only JSON.";
const MAX_RETRIES: usize = 2;

pub struct OpenRouterProvider {
    model: String,
    base_url: String,
    api_key: String,
    agent: ureq::Agent,
}

impl OpenRouterProvider {
    pub fn new(model: Option<&str>, api_key: Option<&str>) -> Result<Self, SummarizeError> {
        let api_key = api_key
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| SummarizeError::Failed("cloud API key not set".into()))?
            .to_string();
        let base_url =
            std::env::var("OPENROUTER_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.into());
        Ok(Self {
            model: model.unwrap_or(DEFAULT_MODEL).to_string(),
            base_url,
            api_key,
            agent: default_agent(),
        })
    }

    fn build_request_body(&self, prompt: &str) -> serde_json::Value {
        json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        })
    }

    fn parse_response(body: &str) -> Result<String, SummarizeError> {
        let response: OpenRouterResponse = serde_json::from_str(body)
            .map_err(|e| SummarizeError::InvalidResponse(e.to_string()))?;
        let choice = response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| SummarizeError::InvalidResponse("no choices".into()))?;
        Ok(choice.message.content)
    }
}

impl SummarizeProvider for OpenRouterProvider {
    fn name(&self) -> &'static str {
        "openrouter"
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
        let url = format!("{}/chat/completions", self.base_url);
        let mut last_error: Option<ureq::Error> = None;
        let mut raw_body: Option<String> = None;

        for attempt in 0..=MAX_RETRIES {
            let body = self.build_request_body(&prompt);
            let response = self
                .agent
                .post(&url)
                .header("Authorization", &format!("Bearer {}", self.api_key))
                .send_json(body);

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
                    .unwrap_or_else(|| "openrouter request failed".to_string()),
            )
        })?;

        let content = Self::parse_response(raw.trim())?;
        if !content.is_empty() {
            on_event(SummarizeEvent::DraftToken(content.clone()));
        }
        let patch = patch::parse_patch(content.trim())?;
        on_event(SummarizeEvent::PatchReady(patch));
        Ok(())
    }
}

#[derive(Deserialize)]
struct OpenRouterResponse {
    choices: Vec<OpenRouterChoice>,
}

#[derive(Deserialize)]
struct OpenRouterChoice {
    message: OpenRouterMessage,
}

#[derive(Deserialize)]
struct OpenRouterMessage {
    content: String,
}

#[cfg(test)]
mod tests {
    use super::OpenRouterProvider;
    use crate::http::default_agent;

    #[test]
    fn parse_response_extracts_content() {
        let body = r#"{"choices":[{"message":{"content":"{\"ops\": []}"}}]}"#;
        let content = OpenRouterProvider::parse_response(body).unwrap();
        assert!(content.contains("ops"));
    }

    #[test]
    fn build_request_body_uses_system_prompt_and_model() {
        let provider = OpenRouterProvider {
            model: "test-model".to_string(),
            base_url: "http://example.com".to_string(),
            api_key: "test-key".to_string(),
            agent: default_agent(),
        };
        let body = provider.build_request_body("prompt");
        let model = body.get("model").and_then(|value| value.as_str());
        let system = body
            .get("messages")
            .and_then(|value| value.as_array())
            .and_then(|messages| messages.first())
            .and_then(|message| message.get("content"))
            .and_then(|value| value.as_str());
        assert_eq!(model, Some("test-model"));
        assert_eq!(
            system,
            Some("You are a meeting notes engine. Follow the instructions and output only JSON.")
        );
    }
}
