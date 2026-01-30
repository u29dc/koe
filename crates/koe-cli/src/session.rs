use crate::config::ConfigPaths;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use thiserror::Error;
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;
use uuid::Uuid;

const CONTEXT_FILE: &str = "context.txt";
const AUDIO_RAW_FILE: &str = "audio.raw";
const AUDIO_WAV_FILE: &str = "audio.wav";
const TRANSCRIPT_FILE: &str = "transcript.jsonl";
const NOTES_FILE: &str = "notes.json";

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
        asr_provider: String,
        asr_model: String,
        summarizer_provider: String,
        summarizer_model: String,
    ) -> Result<Self, SessionError> {
        let id = Uuid::now_v7().to_string();
        let start_time = OffsetDateTime::now_utc().format(&Rfc3339)?;
        Ok(Self {
            id,
            start_time,
            end_time: None,
            finalized: false,
            context,
            context_file: CONTEXT_FILE.to_string(),
            audio_raw_file: AUDIO_RAW_FILE.to_string(),
            audio_wav_file: AUDIO_WAV_FILE.to_string(),
            transcript_file: TRANSCRIPT_FILE.to_string(),
            notes_file: NOTES_FILE.to_string(),
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
        let context_path = dir.join(CONTEXT_FILE);

        let context_value = metadata.context.clone().unwrap_or_default();
        write_atomic(&context_path, context_value.as_bytes())?;
        write_metadata(&metadata_path, &metadata)?;

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
