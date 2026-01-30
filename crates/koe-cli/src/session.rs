use crate::config::ConfigPaths;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
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
    #[error("session time error: {0}")]
    Time(#[from] time::error::Format),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    pub id: String,
    pub start_time: String,
    pub end_time: Option<String>,
    pub finalized: bool,
    pub context: Option<String>,
    pub participants: Vec<String>,
    pub title: Option<String>,
    pub description: Option<String>,
    pub tags: Vec<String>,
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

impl SessionMetadata {
    pub fn new(
        context: Option<String>,
        participants: Vec<String>,
        asr_provider: String,
        asr_model: String,
        summarizer_provider: String,
        summarizer_model: String,
    ) -> Result<Self, SessionError> {
        let id = Uuid::now_v7().to_string();
        let start_time = OffsetDateTime::now_utc().format(&Rfc3339)?;
        let context_file = file_name(CONTEXT_PREFIX, "txt", &id);
        let audio_raw_file = file_name(AUDIO_PREFIX, "raw", &id);
        let audio_wav_file = file_name(AUDIO_PREFIX, "wav", &id);
        let transcript_file = file_name(TRANSCRIPT_PREFIX, "jsonl", &id);
        let notes_file = file_name(NOTES_PREFIX, "json", &id);
        Ok(Self {
            id,
            start_time,
            end_time: None,
            finalized: false,
            context,
            participants,
            title: None,
            description: None,
            tags: Vec::new(),
            context_file,
            audio_raw_file,
            audio_wav_file,
            transcript_file,
            notes_file,
            asr_provider,
            asr_model,
            summarizer_provider,
            summarizer_model,
        })
    }
}

#[derive(Debug)]
pub struct SessionHandle {
    metadata_path: PathBuf,
    context_path: PathBuf,
    metadata: SessionMetadata,
}

impl SessionHandle {
    pub fn start(paths: &ConfigPaths, metadata: SessionMetadata) -> Result<Self, SessionError> {
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
        fs::write(notes_path, [])?;

        Ok(Self {
            metadata_path,
            context_path,
            metadata,
        })
    }

    pub fn context(&self) -> Option<&str> {
        self.metadata.context.as_deref()
    }

    pub fn update_context(&mut self, context: String) -> Result<(), SessionError> {
        self.metadata.context = Some(context.clone());
        write_atomic(&self.context_path, context.as_bytes())?;
        write_metadata(&self.metadata_path, &self.metadata)?;
        Ok(())
    }
}

fn file_name(prefix: &str, ext: &str, id: &str) -> String {
    format!("{prefix}-{id}.{ext}")
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
