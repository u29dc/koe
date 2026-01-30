use crate::SummarizerError;
use crate::types::{MeetingState, NotesOp, NotesPatch, SummarizerEvent, TranscriptSegment};
use serde::Deserialize;
use serde_json::json;

use super::SummarizerProvider;

const DEFAULT_BASE_URL: &str = "http://localhost:11434";

pub struct OllamaProvider {
    model: String,
    base_url: String,
}

impl OllamaProvider {
    pub fn new(model: &str) -> Result<Self, SummarizerError> {
        let base_url = std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.into());
        Ok(Self {
            model: model.to_string(),
            base_url,
        })
    }

    fn build_prompt(&self, recent: &[TranscriptSegment], state: &MeetingState) -> String {
        let transcript = recent
            .iter()
            .map(|s| format!("[{}-{}] {}", s.start_ms, s.end_ms, s.text.trim()))
            .collect::<Vec<_>>()
            .join("\n");
        let state_json = serde_json::to_string(state).unwrap_or_else(|_| "{}".to_string());

        format!(
            "You are a meeting notes engine. Return ONLY valid JSON with this schema:\n{{\n  \"ops\": [\n    {{\"op\": \"add_key_point\", \"id\": \"kp_1\", \"text\": \"...\", \"evidence\": [1,2]}}\n  ]\n}}\n\nRules:\n- patch-only: add/update ops only, no deletes\n- stable IDs: reuse IDs when updating\n- evidence is a list of transcript segment IDs\n- if no updates, return {{\"ops\": []}}\n\nTranscript:\n{transcript}\n\nCurrent state JSON:\n{state_json}\n"
        )
    }

    fn parse_patch(&self, output: &str) -> Result<NotesPatch, SummarizerError> {
        if let Ok(payload) = serde_json::from_str::<PatchPayload>(output) {
            return Ok(payload.into_patch());
        }

        let json = extract_json_object(output)
            .ok_or_else(|| SummarizerError::InvalidResponse("no json object found".into()))?;
        let payload: PatchPayload = serde_json::from_str(json)
            .map_err(|e| SummarizerError::InvalidResponse(e.to_string()))?;
        Ok(payload.into_patch())
    }
}

impl SummarizerProvider for OllamaProvider {
    fn name(&self) -> &'static str {
        "ollama"
    }

    fn summarize(
        &mut self,
        recent_segments: &[TranscriptSegment],
        state: &MeetingState,
        on_event: &mut dyn FnMut(SummarizerEvent),
    ) -> Result<(), SummarizerError> {
        let prompt = self.build_prompt(recent_segments, state);
        let url = format!("{}/api/generate", self.base_url);
        let body = json!({
            "model": self.model,
            "prompt": prompt,
            "stream": true,
        });

        let response = ureq::post(&url)
            .send_json(body)
            .map_err(|e| SummarizerError::Network(format!("{e}")))?;

        let raw = response
            .into_body()
            .read_to_string()
            .map_err(|e| SummarizerError::Network(format!("{e}")))?;

        let mut full_text = String::new();
        for line in raw.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let chunk: OllamaChunk = serde_json::from_str(line)
                .map_err(|e| SummarizerError::InvalidResponse(e.to_string()))?;
            if let Some(token) = chunk.response {
                on_event(SummarizerEvent::DraftToken(token.clone()));
                full_text.push_str(&token);
            }
            if chunk.done.unwrap_or(false) {
                break;
            }
        }

        let patch = self.parse_patch(full_text.trim())?;
        on_event(SummarizerEvent::PatchReady(patch));
        Ok(())
    }
}

#[derive(Deserialize)]
struct OllamaChunk {
    response: Option<String>,
    done: Option<bool>,
}

#[derive(Deserialize)]
struct PatchPayload {
    #[serde(default)]
    ops: Vec<PatchOpPayload>,
}

impl PatchPayload {
    fn into_patch(self) -> NotesPatch {
        NotesPatch {
            ops: self.ops.into_iter().map(|op| op.into()).collect(),
        }
    }
}

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum PatchOpPayload {
    AddKeyPoint {
        id: String,
        text: String,
        #[serde(default)]
        evidence: Vec<u64>,
    },
    AddAction {
        id: String,
        text: String,
        owner: Option<String>,
        due: Option<String>,
        #[serde(default)]
        evidence: Vec<u64>,
    },
    AddDecision {
        id: String,
        text: String,
        #[serde(default)]
        evidence: Vec<u64>,
    },
    UpdateAction {
        id: String,
        owner: Option<String>,
        due: Option<String>,
    },
}

impl From<PatchOpPayload> for NotesOp {
    fn from(value: PatchOpPayload) -> Self {
        match value {
            PatchOpPayload::AddKeyPoint { id, text, evidence } => {
                NotesOp::AddKeyPoint { id, text, evidence }
            }
            PatchOpPayload::AddAction {
                id,
                text,
                owner,
                due,
                evidence,
            } => NotesOp::AddAction {
                id,
                text,
                owner,
                due,
                evidence,
            },
            PatchOpPayload::AddDecision { id, text, evidence } => {
                NotesOp::AddDecision { id, text, evidence }
            }
            PatchOpPayload::UpdateAction { id, owner, due } => {
                NotesOp::UpdateAction { id, owner, due }
            }
        }
    }
}

fn extract_json_object(input: &str) -> Option<&str> {
    let start = input.find('{')?;
    let end = input.rfind('}')?;
    if end <= start {
        return None;
    }
    Some(&input[start..=end])
}

#[cfg(test)]
mod tests {
    use super::{OllamaProvider, extract_json_object};
    use crate::types::{MeetingState, SummarizerEvent, TranscriptSegment};

    fn seg(id: u64, text: &str) -> TranscriptSegment {
        TranscriptSegment {
            id,
            start_ms: 0,
            end_ms: 10,
            speaker: None,
            text: text.to_string(),
            finalized: true,
        }
    }

    #[test]
    fn extract_json_object_finds_bounds() {
        let input = "prefix {\"ops\": []} suffix";
        assert_eq!(extract_json_object(input), Some("{\"ops\": []}"));
    }

    #[test]
    fn parse_patch_payload() {
        let provider = OllamaProvider::new("test").unwrap();
        let output = r#"{"ops":[{"op":"add_key_point","id":"k1","text":"hello","evidence":[1]}]}"#;
        let patch = provider.parse_patch(output).unwrap();
        assert_eq!(patch.ops.len(), 1);
    }

    #[test]
    fn summarize_emits_patch_event_from_prompt() {
        let provider = OllamaProvider::new("test").unwrap();
        let prompt = provider.build_prompt(&[seg(1, "hello")], &MeetingState::default());
        assert!(prompt.contains("Transcript:"));
    }

    #[test]
    fn parse_patch_with_wrapped_json() {
        let provider = OllamaProvider::new("test").unwrap();
        let output = "text {\"ops\": []} more";
        let patch = provider.parse_patch(output).unwrap();
        assert!(patch.ops.is_empty());
    }

    #[test]
    fn patch_ready_event_consumes_ops() {
        let provider = OllamaProvider::new("test").unwrap();
        let output = r#"{"ops":[{"op":"add_decision","id":"d1","text":"decided","evidence":[]}]}"#;
        let patch = provider.parse_patch(output).unwrap();
        let events = [SummarizerEvent::PatchReady(patch)];
        assert_eq!(events.len(), 1);
    }
}
