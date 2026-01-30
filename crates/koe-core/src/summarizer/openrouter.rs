use crate::SummarizerError;
use crate::types::{MeetingState, SummarizerEvent, TranscriptSegment};
use serde::Deserialize;
use serde_json::json;

use super::{SummarizerProvider, patch};

const DEFAULT_BASE_URL: &str = "https://openrouter.ai/api/v1";
const DEFAULT_MODEL: &str = "google/gemini-2.5-flash";

pub struct OpenRouterProvider {
    model: String,
    base_url: String,
    api_key: String,
}

impl OpenRouterProvider {
    pub fn new(model: Option<&str>) -> Result<Self, SummarizerError> {
        let api_key = std::env::var("OPENROUTER_API_KEY")
            .map_err(|_| SummarizerError::Failed("OPENROUTER_API_KEY not set".into()))?;
        let base_url =
            std::env::var("OPENROUTER_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.into());
        Ok(Self {
            model: model.unwrap_or(DEFAULT_MODEL).to_string(),
            base_url,
            api_key,
        })
    }

    fn parse_response(body: &str) -> Result<String, SummarizerError> {
        let response: OpenRouterResponse = serde_json::from_str(body)
            .map_err(|e| SummarizerError::InvalidResponse(e.to_string()))?;
        let choice = response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| SummarizerError::InvalidResponse("no choices".into()))?;
        Ok(choice.message.content)
    }
}

impl SummarizerProvider for OpenRouterProvider {
    fn name(&self) -> &'static str {
        "openrouter"
    }

    fn summarize(
        &mut self,
        recent_segments: &[TranscriptSegment],
        state: &MeetingState,
        context: Option<&str>,
        on_event: &mut dyn FnMut(SummarizerEvent),
    ) -> Result<(), SummarizerError> {
        let prompt = patch::build_prompt(recent_segments, state, context);
        let url = format!("{}/chat/completions", self.base_url);
        let body = json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a meeting notes engine."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        });

        let response = ureq::post(&url)
            .header("Authorization", &format!("Bearer {}", self.api_key))
            .send_json(body)
            .map_err(|e| SummarizerError::Network(format!("{e}")))?;

        let raw = response
            .into_body()
            .read_to_string()
            .map_err(|e| SummarizerError::Network(format!("{e}")))?;

        let content = Self::parse_response(raw.trim())?;
        if !content.is_empty() {
            on_event(SummarizerEvent::DraftToken(content.clone()));
        }
        let patch = patch::parse_patch(content.trim())?;
        on_event(SummarizerEvent::PatchReady(patch));
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

    #[test]
    fn parse_response_extracts_content() {
        let body = r#"{"choices":[{"message":{"content":"{\"ops\": []}"}}]}"#;
        let content = OpenRouterProvider::parse_response(body).unwrap();
        assert!(content.contains("ops"));
    }
}
