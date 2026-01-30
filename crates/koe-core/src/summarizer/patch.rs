use crate::SummarizerError;
use crate::types::{MeetingState, NotesOp, NotesPatch, TranscriptSegment};
use serde::Deserialize;

pub(crate) fn build_prompt(
    recent: &[TranscriptSegment],
    state: &MeetingState,
    context: Option<&str>,
) -> String {
    let transcript = recent
        .iter()
        .filter(|s| s.finalized)
        .map(|s| format!("[{}-{}] {}", s.start_ms, s.end_ms, s.text.trim()))
        .collect::<Vec<_>>()
        .join("\n");
    let state_json = serde_json::to_string(state).unwrap_or_else(|_| "{}".to_string());
    let context_block = context
        .filter(|value| !value.is_empty())
        .map(|value| format!("Context:\n{value}\n\n"))
        .unwrap_or_default();

    format!(
        "You are a meeting notes engine. Return ONLY valid JSON with this schema:\n{{\n  \"ops\": [\n    {{\"op\": \"add_key_point\", \"id\": \"kp_1\", \"text\": \"...\", \"evidence\": [1,2]}}\n  ]\n}}\n\nRules:\n- patch-only: add/update ops only, no deletes\n- stable IDs: reuse IDs when updating\n- evidence is a list of transcript segment IDs\n- if no updates, return {{\"ops\": []}}\n\n{context_block}Transcript:\n{transcript}\n\nCurrent state JSON:\n{state_json}\n"
    )
}

pub(crate) fn parse_patch(output: &str) -> Result<NotesPatch, SummarizerError> {
    if let Ok(payload) = serde_json::from_str::<PatchPayload>(output) {
        return Ok(payload.into_patch());
    }

    let json = extract_json_object(output)
        .ok_or_else(|| SummarizerError::InvalidResponse("no json object found".into()))?;
    let payload: PatchPayload =
        serde_json::from_str(json).map_err(|e| SummarizerError::InvalidResponse(e.to_string()))?;
    Ok(payload.into_patch())
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
    use super::{build_prompt, extract_json_object, parse_patch};
    use crate::types::{MeetingState, TranscriptSegment};

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

    fn seg_unfinalized(id: u64, text: &str) -> TranscriptSegment {
        TranscriptSegment {
            finalized: false,
            ..seg(id, text)
        }
    }

    #[test]
    fn extract_json_object_finds_bounds() {
        let input = "prefix {\"ops\": []} suffix";
        assert_eq!(extract_json_object(input), Some("{\"ops\": []}"));
    }

    #[test]
    fn parse_patch_payload() {
        let output = r#"{"ops":[{"op":"add_key_point","id":"k1","text":"hello","evidence":[1]}]}"#;
        let patch = parse_patch(output).unwrap();
        assert_eq!(patch.ops.len(), 1);
    }

    #[test]
    fn parse_patch_with_wrapped_json() {
        let output = "text {\"ops\": []} more";
        let patch = parse_patch(output).unwrap();
        assert!(patch.ops.is_empty());
    }

    #[test]
    fn build_prompt_includes_transcript() {
        let prompt = build_prompt(&[seg(1, "hello")], &MeetingState::default(), None);
        assert!(prompt.contains("Transcript:"));
    }

    #[test]
    fn build_prompt_includes_context() {
        let prompt = build_prompt(
            &[seg(1, "hello")],
            &MeetingState::default(),
            Some("team sync"),
        );
        assert!(prompt.contains("Context:"));
        assert!(prompt.contains("team sync"));
    }

    #[test]
    fn build_prompt_uses_finalized_only() {
        let prompt = build_prompt(
            &[seg(1, "keep"), seg_unfinalized(2, "drop")],
            &MeetingState::default(),
            None,
        );
        assert!(prompt.contains("keep"));
        assert!(!prompt.contains("drop"));
    }
}
