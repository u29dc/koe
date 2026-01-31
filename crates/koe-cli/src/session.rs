use crate::config::ConfigPaths;
use koe_core::types::{MeetingNotes, TranscriptSegment};
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
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
    pub transcribe_provider: String,
    pub transcribe_model: String,
    pub summarize_provider: String,
    pub summarize_model: String,
}

#[derive(Debug, Clone)]
pub struct SessionMetadataInput {
    pub context: Option<String>,
    pub participants: Vec<String>,
    pub audio_sample_rate_hz: u32,
    pub audio_channels: u16,
    pub audio_sources: Vec<String>,
    pub transcribe_provider: String,
    pub transcribe_model: String,
    pub summarize_provider: String,
    pub summarize_model: String,
}

#[derive(Debug, Clone)]
pub struct SessionFactory {
    paths: ConfigPaths,
    export_dir: Option<PathBuf>,
    audio_sample_rate_hz: u32,
    audio_channels: u16,
    audio_sources: Vec<String>,
}

impl SessionFactory {
    pub fn new(
        paths: ConfigPaths,
        export_dir: Option<PathBuf>,
        audio_sample_rate_hz: u32,
        audio_channels: u16,
        audio_sources: Vec<String>,
    ) -> Self {
        Self {
            paths,
            export_dir,
            audio_sample_rate_hz,
            audio_channels,
            audio_sources,
        }
    }

    pub fn create(
        &self,
        transcribe_provider: String,
        transcribe_model: String,
        summarize_provider: String,
        summarize_model: String,
        context: Option<String>,
        participants: Vec<String>,
    ) -> Result<SessionHandle, SessionError> {
        let metadata = SessionMetadata::new(SessionMetadataInput {
            context,
            participants,
            audio_sample_rate_hz: self.audio_sample_rate_hz,
            audio_channels: self.audio_channels,
            audio_sources: self.audio_sources.clone(),
            transcribe_provider,
            transcribe_model,
            summarize_provider,
            summarize_model,
        })?;
        SessionHandle::start(&self.paths, metadata, self.export_dir.clone())
    }

    pub fn sessions_dir(&self) -> &Path {
        &self.paths.sessions_dir
    }
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
            transcribe_provider: input.transcribe_provider,
            transcribe_model: input.transcribe_model,
            summarize_provider: input.summarize_provider,
            summarize_model: input.summarize_model,
        })
    }
}

#[derive(Debug, Clone)]
pub struct SessionHandle {
    dir: PathBuf,
    export_dir: Option<PathBuf>,
    metadata_path: PathBuf,
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
        fs::write(&audio_raw_path, [])?;
        set_strict_permissions(&audio_raw_path)?;
        fs::write(&transcript_path, [])?;
        set_strict_permissions(&transcript_path)?;
        let notes_snapshot = NotesSnapshot {
            updated_at: OffsetDateTime::now_utc().format(&Rfc3339)?,
            state: MeetingNotes::default(),
        };
        let notes_payload = serde_json::to_string_pretty(&notes_snapshot)?;
        write_atomic(&notes_path, notes_payload.as_bytes())?;

        Ok(Self {
            dir,
            export_dir,
            metadata_path,
            metadata,
        })
    }

    pub fn session_dir(&self) -> &Path {
        &self.dir
    }

    pub fn audio_raw_path(&self) -> PathBuf {
        self.dir.join(&self.metadata.audio_raw_file)
    }

    pub fn export_transcript_path(&self) -> Result<PathBuf, SessionError> {
        let root = self.export_root()?;
        Ok(root.join("transcript.md"))
    }

    pub fn export_notes_path(&self) -> Result<PathBuf, SessionError> {
        let root = self.export_root()?;
        Ok(root.join("notes.md"))
    }

    pub fn update_transcribe(
        &mut self,
        provider: String,
        model: String,
    ) -> Result<(), SessionError> {
        self.metadata.transcribe_provider = provider;
        self.metadata.transcribe_model = model;
        self.touch_metadata()
    }

    pub fn update_summarize(
        &mut self,
        provider: String,
        model: String,
    ) -> Result<(), SessionError> {
        self.metadata.summarize_provider = provider;
        self.metadata.summarize_model = model;
        self.touch_metadata()
    }

    pub fn is_finalized(&self) -> bool {
        self.metadata.finalized
    }

    pub fn open_audio_raw(&self) -> Result<std::fs::File, SessionError> {
        warn_if_loose_permissions(&self.audio_raw_path())?;
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
        warn_if_loose_permissions(&self.transcript_path())?;
        for segment in segments {
            let record = TranscriptRecord::from_segment(segment);
            let line = serde_json::to_string(&record)?;
            file.write_all(line.as_bytes())?;
            file.write_all(b"\n")?;
        }
        self.touch_metadata()?;
        Ok(())
    }

    pub fn write_notes(&mut self, state: &MeetingNotes) -> Result<(), SessionError> {
        let snapshot = NotesSnapshot {
            updated_at: OffsetDateTime::now_utc().format(&Rfc3339)?,
            state: state.clone(),
        };
        let payload = serde_json::to_string_pretty(&snapshot)?;
        write_atomic(&self.notes_path(), payload.as_bytes())?;
        self.touch_metadata()?;
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

    pub fn export_audio_wav(&self) -> Result<(), SessionError> {
        let export_root = self.export_root()?;
        let wav_path = export_root.join(&self.metadata.audio_wav_file);
        let raw_path = self.audio_raw_path();
        write_wav_from_raw(
            &raw_path,
            &wav_path,
            self.metadata.audio_sample_rate_hz,
            self.metadata.audio_channels,
        )
    }

    pub fn export_notes_markdown(&self, state: &MeetingNotes) -> Result<(), SessionError> {
        let export_root = self.export_root()?;
        let path = export_root.join("notes.md");
        let mut output = String::from("# Notes\n\n");

        if state.bullets.is_empty() {
            output.push_str("- (none)\n");
        } else {
            for item in &state.bullets {
                output.push_str(&format!("- {}\n", item.text.trim()));
            }
        }

        write_atomic(&path, output.as_bytes())?;
        Ok(())
    }

    pub fn export_on_exit(
        &mut self,
        segments: &[TranscriptSegment],
        state: &MeetingNotes,
    ) -> Result<(), SessionError> {
        self.write_notes(state)?;
        self.export_audio_wav()?;
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
    state: MeetingNotes,
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
    set_strict_permissions(&tmp_path)?;
    fs::rename(tmp_path, path)?;
    Ok(())
}

fn set_strict_permissions(path: &Path) -> Result<(), SessionError> {
    #[cfg(unix)]
    {
        let perm = fs::Permissions::from_mode(0o600);
        fs::set_permissions(path, perm)?;
    }
    Ok(())
}

fn warn_if_loose_permissions(path: &Path) -> Result<(), SessionError> {
    #[cfg(unix)]
    {
        let metadata = fs::metadata(path)?;
        let mode = metadata.permissions().mode() & 0o777;
        if mode & 0o077 != 0 {
            eprintln!(
                "session file {} is group/world readable; set permissions to 0600",
                path.display()
            );
        }
    }
    Ok(())
}

fn write_wav_from_raw(
    raw_path: &Path,
    wav_path: &Path,
    sample_rate: u32,
    channels: u16,
) -> Result<(), SessionError> {
    let metadata = fs::metadata(raw_path)?;
    let byte_len = metadata.len();
    let channels = channels.max(1);
    let frame_bytes = u64::from(channels) * 4;
    if byte_len % frame_bytes != 0 {
        return Err(io::Error::other("audio.raw length is not aligned to channel frames").into());
    }
    let frames = byte_len / frame_bytes;

    let tmp_path = wav_path.with_extension("tmp");
    let mut reader = fs::File::open(raw_path)?;
    let mut writer = io::BufWriter::new(fs::File::create(&tmp_path)?);
    write_wav_header(&mut writer, sample_rate, channels, frames)?;
    io::copy(&mut reader, &mut writer)?;
    writer.flush()?;
    set_strict_permissions(&tmp_path)?;
    fs::rename(tmp_path, wav_path)?;
    Ok(())
}

fn write_wav_header(
    writer: &mut impl Write,
    sample_rate: u32,
    channels: u16,
    frames: u64,
) -> Result<(), SessionError> {
    let bits_per_sample: u16 = 32;
    let block_align = channels * (bits_per_sample / 8);
    let byte_rate = sample_rate * u32::from(block_align);
    let data_size = frames
        .checked_mul(u64::from(block_align))
        .ok_or_else(|| io::Error::other("audio length overflow"))?;
    let data_size_u32: u32 = data_size
        .try_into()
        .map_err(|_| io::Error::other("audio too large for wav header"))?;
    let frames_u32: u32 = frames
        .try_into()
        .map_err(|_| io::Error::other("audio too large for wav header"))?;

    let fmt_chunk_size: u32 = 18;
    let fact_chunk_size: u32 = 4;
    let file_size = 4 + (8 + fmt_chunk_size) + (8 + fact_chunk_size) + (8 + data_size_u32);

    writer.write_all(b"RIFF")?;
    writer.write_all(&file_size.to_le_bytes())?;
    writer.write_all(b"WAVE")?;

    writer.write_all(b"fmt ")?;
    writer.write_all(&fmt_chunk_size.to_le_bytes())?;
    writer.write_all(&3u16.to_le_bytes())?;
    writer.write_all(&channels.to_le_bytes())?;
    writer.write_all(&sample_rate.to_le_bytes())?;
    writer.write_all(&byte_rate.to_le_bytes())?;
    writer.write_all(&block_align.to_le_bytes())?;
    writer.write_all(&bits_per_sample.to_le_bytes())?;
    writer.write_all(&0u16.to_le_bytes())?;

    writer.write_all(b"fact")?;
    writer.write_all(&fact_chunk_size.to_le_bytes())?;
    writer.write_all(&frames_u32.to_le_bytes())?;

    writer.write_all(b"data")?;
    writer.write_all(&data_size_u32.to_le_bytes())?;
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
    use koe_core::types::{MeetingNotes, TranscriptSegment};
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
            transcribe_provider: "whisper".to_string(),
            transcribe_model: "base.en".to_string(),
            summarize_provider: "ollama".to_string(),
            summarize_model: "qwen3:30b-a3b".to_string(),
        })
        .unwrap();
        let session_id = metadata.id.clone();
        let notes_file = metadata.notes_file.clone();
        let audio_wav_file = metadata.audio_wav_file.clone();

        let mut session = SessionHandle::start(&paths, metadata, None).unwrap();

        let segments = vec![TranscriptSegment {
            id: 1,
            start_ms: 0,
            end_ms: 1_000,
            speaker: Some("Me".to_string()),
            text: "hello".to_string(),
            finalized: true,
        }];
        let mut state = MeetingNotes::default();
        state.bullets.push(koe_core::types::NoteBullet {
            id: "n1".to_string(),
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

        let wav_path = session_dir.join(audio_wav_file);
        assert!(wav_path.exists());
    }
}
