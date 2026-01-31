use crate::SummarizeError;
use crate::types::{MeetingNotes, NotesOp, NotesPatch, TranscriptSegment};
use serde::Deserialize;

pub(crate) fn build_prompt(
    recent: &[TranscriptSegment],
    notes: &MeetingNotes,
    context: Option<&str>,
    participants: &[String],
) -> String {
    const JSON_SCHEMA_SAMPLE: &str = r#"
{
    "ops": [
        {"op": "add", "id": "n_1", "text": "...", "evidence": [1,2]}
    ]
}
"#;
    const EMPTY_OPS: &str = r#"{"ops": []}"#;
    let transcript = recent
        .iter()
        .map(|s| {
            let text = s.text.trim();
            match s.speaker.as_deref() {
                Some(speaker) if !speaker.is_empty() => {
                    format!("[{}-{}] {speaker}: {text}", s.start_ms, s.end_ms)
                }
                _ => format!("[{}-{}] {text}", s.start_ms, s.end_ms),
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    let context_block = context
        .filter(|value| !value.is_empty())
        .map(|value| format!("Context:\n{value}\n\n"))
        .unwrap_or_default();
    let participants_list = participants
        .iter()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();
    let participants_block = if participants_list.is_empty() {
        String::new()
    } else {
        format!("Participants: {}\n\n", participants_list.join(", "))
    };
    let notes_block = if notes.bullets.is_empty() {
        String::new()
    } else {
        let lines = notes
            .bullets
            .iter()
            .map(|bullet| format!("- {}: {}", bullet.id, bullet.text.trim()))
            .collect::<Vec<_>>()
            .join("\n");
        format!("Existing notes (avoid duplicates):\n{lines}\n\n")
    };

    format!(
        r#"
<task>
You are processing a live meeting transcript in 4-second increments. Your job: capture anything that might be worth remembering. Err on the side of inclusion -- it's easy to ignore a low-value note later, but impossible to recover a missed one.
</task>

<schema>
Output JSON matching this schema:
{schema}
</schema>

<rules>
Only return {empty_ops} if the transcript is truly empty content: pure filler, greetings with no substance, or silence.
</rules>

---

<capture>
WHAT TO CAPTURE:

Tier 1 - Always capture:
- Decisions: anything agreed, chosen, rejected, or finalized
- Action items: who will do what (deadline optional but include if mentioned)
- Commitments: promises, offers, acceptances
- Deadlines or dates mentioned
- Names, titles, or contacts introduced

Tier 2 - Capture liberally:
- Key facts, numbers, metrics, technical details
- Opinions or positions stated clearly ("I think we should...", "My concern is...")
- Questions raised (even if not yet answered)
- Problems or blockers identified
- Topics flagged for follow-up ("we should discuss X later")
- Context that explains why something matters

Capture liberally, but only if it adds new facts. If it rephrases an existing note, skip it.
</capture>

---

<skip>
WHAT TO SKIP:

- Pure filler and backchannels: "yeah", "um", "so", "right", "okay", "thanks", "got it"
- Greetings and small talk with zero content
- Paraphrases of existing notes (check the list; new facts only)
- Sentence fragments that are clearly incomplete mid-thought
- Single-word utterances unless they are a name, number, or date

That's it. Everything else is fair game.
</skip>

---

<format>
FORMAT RULES:

- Max 3 ops per response; 0-2 is normal
- Each bullet: 1 sentence, <=120 characters
- Prefer concrete and specific over vague ("ship Friday" not "ship soon")
- Do not include speaker labels in note text
- ID format: "n_<number>" -- must not collide with existing note IDs
- Evidence field: list start_ms values from supporting transcript segments
</format>

---

<input>
<input_context>
{context_block}
</input_context>

<input_participants>
{participants_block}
</input_participants>

<input_notes>
{notes_block}
</input_notes>

<input_transcript>
{transcript}
</input_transcript>
</input>
"#,
        schema = JSON_SCHEMA_SAMPLE,
        empty_ops = EMPTY_OPS,
        context_block = context_block,
        participants_block = participants_block,
        notes_block = notes_block,
        transcript = transcript
    )
}

pub(crate) fn parse_patch(output: &str) -> Result<NotesPatch, SummarizeError> {
    if let Ok(payload) = serde_json::from_str::<PatchPayload>(output) {
        return Ok(payload.into_patch());
    }

    let json = extract_json_object(output)
        .ok_or_else(|| SummarizeError::InvalidResponse("no json object found".into()))?;
    let payload: PatchPayload =
        serde_json::from_str(json).map_err(|e| SummarizeError::InvalidResponse(e.to_string()))?;
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
    Add {
        id: String,
        text: String,
        #[serde(default)]
        evidence: Vec<u64>,
    },
}

impl From<PatchOpPayload> for NotesOp {
    fn from(value: PatchOpPayload) -> Self {
        match value {
            PatchOpPayload::Add { id, text, evidence } => NotesOp::Add { id, text, evidence },
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
    use crate::types::{MeetingNotes, NoteBullet, TranscriptSegment};

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
        let output = r#"{"ops":[{"op":"add","id":"n1","text":"hello","evidence":[1]}]}"#;
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
        let prompt = build_prompt(&[seg(1, "hello")], &MeetingNotes::default(), None, &[]);
        assert!(prompt.contains("<input_transcript>"));
    }

    #[test]
    fn build_prompt_includes_context() {
        let prompt = build_prompt(
            &[seg(1, "hello")],
            &MeetingNotes::default(),
            Some("team sync"),
            &[],
        );
        assert!(prompt.contains("Context:"));
        assert!(prompt.contains("team sync"));
    }

    #[test]
    fn build_prompt_includes_unfinalized() {
        let prompt = build_prompt(
            &[seg(1, "keep"), seg_unfinalized(2, "drop")],
            &MeetingNotes::default(),
            None,
            &[],
        );
        assert!(prompt.contains("keep"));
        assert!(prompt.contains("drop"));
    }

    #[test]
    fn build_prompt_is_information_dense() {
        let prompt = build_prompt(&[seg(1, "alpha")], &MeetingNotes::default(), None, &[]);
        assert!(prompt.contains("WHAT TO CAPTURE"));
        assert!(prompt.contains("<=120"));
        assert!(prompt.contains("Max 3 ops per response"));
    }

    #[test]
    fn build_prompt_includes_speaker_labels() {
        let mut with_speaker = seg(1, "hello");
        with_speaker.speaker = Some("Me".to_string());
        let prompt = build_prompt(&[with_speaker], &MeetingNotes::default(), None, &[]);
        assert!(prompt.contains("Me: hello"));
    }

    #[test]
    fn build_prompt_includes_participants() {
        let participants = vec!["Han".to_string(), "Sarah".to_string()];
        let prompt = build_prompt(
            &[seg(1, "hello")],
            &MeetingNotes::default(),
            None,
            &participants,
        );
        assert!(prompt.contains("Participants: Han, Sarah"));
    }

    #[test]
    fn build_prompt_includes_existing_notes() {
        let mut notes = MeetingNotes::default();
        notes.bullets.push(NoteBullet {
            id: "n_1".to_string(),
            text: "Decision: ship by Friday".to_string(),
            evidence: vec![1],
        });
        let prompt = build_prompt(&[seg(1, "hello")], &notes, None, &[]);
        assert!(prompt.contains("Existing notes (avoid duplicates):"));
        assert!(prompt.contains("n_1"));
        assert!(prompt.contains("Decision: ship by Friday"));
    }
}
