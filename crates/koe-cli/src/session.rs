use crate::config::ConfigPaths;
use koe_core::types::{MeetingState, TranscriptSegment};
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};
use thiserror::Error;
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;
use uuid::Uuid;

const CONTEXT_PREFIX: &str = "context";
const AUDIO_PREFIX: &str = "audio";
const TRANSCRIPT_PREFIX: &str = "transcript";
const NOTES_PREFIX: &str = "notes";

#[derive(Debug, Error)]
pub enum SessionError {
    #[error("session io error: {0}")]
    Io(#[from] io::Error),
    #[error("session metadata error: {0}")]
    Metadata(#[from] toml::ser::Error),
    #[error("session json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("session time error: {0}")]
    Time(#[from] time::error::Format),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    pub id: String,
    pub start_time: String,
    pub last_update: String,
    pub end_time: Option<String>,
    pub finalized: bool,
    pub context: Option<String>,
    pub participants: Vec<String>,
    pub title: Option<String>,
    pub description: Option<String>,
    pub tags: Vec<String>,
    pub audio_sample_rate_hz: u32,
    pub audio_channels: u16,
    pub audio_sources: Vec<String>,
    pub context_file: String,
    pub audio_raw_file: String,
    pub audio_wav_file: String,
    pub transcript_file: String,
    pub notes_file: String,
    pub asr_provider: String,
    pub asr_model: String,
    pub summarizer_provider: String,
    pub summarizer_model: String,
}

#[derive(Debug, Clone)]
pub struct SessionMetadataInput {
    pub context: Option<String>,
    pub participants: Vec<String>,
    pub audio_sample_rate_hz: u32,
    pub audio_channels: u16,
    pub audio_sources: Vec<String>,
    pub asr_provider: String,
    pub asr_model: String,
    pub summarizer_provider: String,
    pub summarizer_model: String,
}

impl SessionMetadata {
    pub fn new(input: SessionMetadataInput) -> Result<Self, SessionError> {
        let id = Uuid::now_v7().to_string();
        let start_time = OffsetDateTime::now_utc().format(&Rfc3339)?;
        let last_update = start_time.clone();
        let context_file = file_name(CONTEXT_PREFIX, "txt", &id);
        let audio_raw_file = file_name(AUDIO_PREFIX, "raw", &id);
        let audio_wav_file = file_name(AUDIO_PREFIX, "wav", &id);
        let transcript_file = file_name(TRANSCRIPT_PREFIX, "jsonl", &id);
        let notes_file = file_name(NOTES_PREFIX, "json", &id);
        Ok(Self {
            id,
            start_time,
            last_update,
            end_time: None,
            finalized: false,
            context: input.context,
            participants: input.participants,
            title: None,
            description: None,
            tags: Vec::new(),
            audio_sample_rate_hz: input.audio_sample_rate_hz,
            audio_channels: input.audio_channels,
            audio_sources: input.audio_sources,
            context_file,
            audio_raw_file,
            audio_wav_file,
            transcript_file,
            notes_file,
            asr_provider: input.asr_provider,
            asr_model: input.asr_model,
            summarizer_provider: input.summarizer_provider,
            summarizer_model: input.summarizer_model,
        })
    }
}

#[derive(Debug)]
pub struct SessionHandle {
    dir: PathBuf,
    export_dir: Option<PathBuf>,
    metadata_path: PathBuf,
    context_path: PathBuf,
    metadata: SessionMetadata,
}

impl SessionHandle {
    pub fn start(
        paths: &ConfigPaths,
        metadata: SessionMetadata,
        export_dir: Option<PathBuf>,
    ) -> Result<Self, SessionError> {
        fs::create_dir_all(&paths.sessions_dir)?;
        let dir = paths.sessions_dir.join(&metadata.id);
        fs::create_dir_all(&dir)?;
        let metadata_path = dir.join("metadata.toml");
        let context_path = dir.join(&metadata.context_file);
        let audio_raw_path = dir.join(&metadata.audio_raw_file);
        let transcript_path = dir.join(&metadata.transcript_file);
        let notes_path = dir.join(&metadata.notes_file);

        let context_value = metadata.context.clone().unwrap_or_default();
        write_atomic(&context_path, context_value.as_bytes())?;
        write_metadata(&metadata_path, &metadata)?;
        fs::write(audio_raw_path, [])?;
        fs::write(transcript_path, [])?;
        let notes_snapshot = NotesSnapshot {
            updated_at: OffsetDateTime::now_utc().format(&Rfc3339)?,
            state: MeetingState::default(),
        };
        let notes_payload = serde_json::to_string_pretty(&notes_snapshot)?;
        write_atomic(&notes_path, notes_payload.as_bytes())?;

        Ok(Self {
            dir,
            export_dir,
            metadata_path,
            context_path,
            metadata,
        })
    }

    pub fn context(&self) -> Option<&str> {
        self.metadata.context.as_deref()
    }

    pub fn open_audio_raw(&self) -> Result<std::fs::File, SessionError> {
        Ok(OpenOptions::new()
            .append(true)
            .open(self.audio_raw_path())?)
    }

    pub fn append_transcript(
        &mut self,
        segments: &[TranscriptSegment],
    ) -> Result<(), SessionError> {
        if segments.is_empty() {
            return Ok(());
        }
        let mut file = OpenOptions::new()
            .append(true)
            .open(self.transcript_path())?;
        for segment in segments {
            let record = TranscriptRecord::from_segment(segment);
            let line = serde_json::to_string(&record)?;
            file.write_all(line.as_bytes())?;
            file.write_all(b"\n")?;
        }
        self.touch_metadata()?;
        Ok(())
    }

    pub fn write_notes(&mut self, state: &MeetingState) -> Result<(), SessionError> {
        let snapshot = NotesSnapshot {
            updated_at: OffsetDateTime::now_utc().format(&Rfc3339)?,
            state: state.clone(),
        };
        let payload = serde_json::to_string_pretty(&snapshot)?;
        write_atomic(&self.notes_path(), payload.as_bytes())?;
        self.touch_metadata()?;
        Ok(())
    }

    pub fn update_context(&mut self, context: String) -> Result<(), SessionError> {
        self.metadata.context = Some(context.clone());
        self.metadata.last_update = OffsetDateTime::now_utc().format(&Rfc3339)?;
        write_atomic(&self.context_path, context.as_bytes())?;
        write_metadata(&self.metadata_path, &self.metadata)?;
        Ok(())
    }

    pub fn export_transcript_markdown(
        &self,
        segments: &[TranscriptSegment],
    ) -> Result<(), SessionError> {
        let export_root = self.export_root()?;
        let path = export_root.join("transcript.md");
        let mut output = String::from("# Transcript\n");
        if segments.is_empty() {
            output.push_str("- (empty)\n");
        } else {
            for segment in segments {
                let start = format_timestamp(segment.start_ms);
                let end = format_timestamp(segment.end_ms);
                let speaker = segment.speaker.as_deref().unwrap_or("Unknown");
                let text = segment.text.replace('\n', " ").trim().to_string();
                output.push_str(&format!("- [{start}-{end}] {speaker}: {text}\n"));
            }
        }
        write_atomic(&path, output.as_bytes())?;
        Ok(())
    }

    pub fn export_notes_markdown(&self, state: &MeetingState) -> Result<(), SessionError> {
        let export_root = self.export_root()?;
        let path = export_root.join("notes.md");
        let mut output = String::from("# Notes\n\n");

        output.push_str("## Key points\n");
        if state.key_points.is_empty() {
            output.push_str("- (none)\n");
        } else {
            for item in &state.key_points {
                output.push_str(&format!("- {}\n", item.text.trim()));
            }
        }

        output.push_str("\n## Actions\n");
        if state.actions.is_empty() {
            output.push_str("- (none)\n");
        } else {
            for item in &state.actions {
                let mut suffix = Vec::new();
                if let Some(owner) = item.owner.as_deref() {
                    suffix.push(format!("owner: {owner}"));
                }
                if let Some(due) = item.due.as_deref() {
                    suffix.push(format!("due: {due}"));
                }
                if suffix.is_empty() {
                    output.push_str(&format!("- {}\n", item.text.trim()));
                } else {
                    output.push_str(&format!("- {} ({})\n", item.text.trim(), suffix.join(", ")));
                }
            }
        }

        output.push_str("\n## Decisions\n");
        if state.decisions.is_empty() {
            output.push_str("- (none)\n");
        } else {
            for item in &state.decisions {
                output.push_str(&format!("- {}\n", item.text.trim()));
            }
        }

        write_atomic(&path, output.as_bytes())?;
        Ok(())
    }

    pub fn export_on_exit(
        &mut self,
        segments: &[TranscriptSegment],
        state: &MeetingState,
    ) -> Result<(), SessionError> {
        self.write_notes(state)?;
        self.export_transcript_markdown(segments)?;
        self.export_notes_markdown(state)?;
        self.finalize()
    }

    pub fn finalize(&mut self) -> Result<(), SessionError> {
        let end_time = OffsetDateTime::now_utc().format(&Rfc3339)?;
        self.metadata.end_time = Some(end_time.clone());
        self.metadata.last_update = end_time;
        self.metadata.finalized = true;
        write_metadata(&self.metadata_path, &self.metadata)?;
        Ok(())
    }

    fn audio_raw_path(&self) -> PathBuf {
        self.dir.join(&self.metadata.audio_raw_file)
    }

    fn transcript_path(&self) -> PathBuf {
        self.dir.join(&self.metadata.transcript_file)
    }

    fn notes_path(&self) -> PathBuf {
        self.dir.join(&self.metadata.notes_file)
    }

    fn export_root(&self) -> Result<PathBuf, SessionError> {
        let root = match &self.export_dir {
            Some(base) => base.join(&self.metadata.id),
            None => self.dir.clone(),
        };
        fs::create_dir_all(&root)?;
        Ok(root)
    }

    fn touch_metadata(&mut self) -> Result<(), SessionError> {
        self.metadata.last_update = OffsetDateTime::now_utc().format(&Rfc3339)?;
        write_metadata(&self.metadata_path, &self.metadata)?;
        Ok(())
    }
}

fn file_name(prefix: &str, ext: &str, id: &str) -> String {
    format!("{prefix}-{id}.{ext}")
}

#[derive(Serialize)]
struct TranscriptRecord {
    id: u64,
    start_ms: i64,
    end_ms: i64,
    speaker: Option<String>,
    text: String,
    finalized: bool,
    source: String,
}

impl TranscriptRecord {
    fn from_segment(segment: &TranscriptSegment) -> Self {
        let source = match segment.speaker.as_deref() {
            Some("Me") => "microphone",
            Some("Them") => "system",
            _ => "unknown",
        };
        Self {
            id: segment.id,
            start_ms: segment.start_ms,
            end_ms: segment.end_ms,
            speaker: segment.speaker.clone(),
            text: segment.text.clone(),
            finalized: segment.finalized,
            source: source.to_string(),
        }
    }
}

#[derive(Serialize)]
struct NotesSnapshot {
    updated_at: String,
    state: MeetingState,
}

fn write_metadata(path: &Path, metadata: &SessionMetadata) -> Result<(), SessionError> {
    let contents = toml::to_string_pretty(metadata)?;
    write_atomic(path, contents.as_bytes())
}

fn write_atomic(path: &Path, contents: &[u8]) -> Result<(), SessionError> {
    let parent = path
        .parent()
        .ok_or_else(|| io::Error::other("session path missing parent directory"))?;
    let tmp_path = parent.join(".tmp-write");
    fs::write(&tmp_path, contents)?;
    fs::rename(tmp_path, path)?;
    Ok(())
}

fn format_timestamp(ms: i64) -> String {
    let total_seconds = ms.max(0) / 1000;
    let minutes = total_seconds / 60;
    let seconds = total_seconds % 60;
    format!("{minutes:02}:{seconds:02}")
}

#[cfg(test)]
mod tests {
    use super::{SessionHandle, SessionMetadata, SessionMetadataInput};
    use crate::config::ConfigPaths;
    use koe_core::types::{MeetingState, NoteItem, TranscriptSegment};
    use tempfile::tempdir;

    #[test]
    fn export_on_exit_writes_transcript_and_notes() {
        let temp = tempdir().unwrap();
        let paths = ConfigPaths::from_base(temp.path().join("koe"));
        let metadata = SessionMetadata::new(SessionMetadataInput {
            context: None,
            participants: Vec::new(),
            audio_sample_rate_hz: 48_000,
            audio_channels: 1,
            audio_sources: vec!["system".to_string()],
            asr_provider: "whisper".to_string(),
            asr_model: "base.en".to_string(),
            summarizer_provider: "ollama".to_string(),
            summarizer_model: "qwen3:30b-a3b".to_string(),
        })
        .unwrap();
        let session_id = metadata.id.clone();
        let notes_file = metadata.notes_file.clone();

        let mut session = SessionHandle::start(&paths, metadata, None).unwrap();

        let segments = vec![TranscriptSegment {
            id: 1,
            start_ms: 0,
            end_ms: 1_000,
            speaker: Some("Me".to_string()),
            text: "hello".to_string(),
            finalized: true,
        }];
        let mut state = MeetingState::default();
        state.key_points.push(NoteItem {
            id: "k1".to_string(),
            text: "first point".to_string(),
            evidence: vec![1],
        });

        session.export_on_exit(&segments, &state).unwrap();

        let session_dir = paths.sessions_dir.join(session_id);
        let transcript_md = std::fs::read_to_string(session_dir.join("transcript.md")).unwrap();
        assert!(transcript_md.contains("hello"));

        let notes_path = session_dir.join(notes_file);
        let notes_json = std::fs::read_to_string(notes_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&notes_json).unwrap();
        assert!(parsed.get("updated_at").is_some());
        assert!(parsed.get("state").is_some());
    }
}
