pub mod cloud;
pub mod local;
mod patch;

use crate::SummarizeError;
use crate::types::{MeetingNotes, SummarizeEvent, TranscriptSegment};

const DEFAULT_OLLAMA_MODEL: &str = "qwen3:30b-a3b";

pub trait SummarizeProvider: Send {
    fn name(&self) -> &'static str;
    fn summarize(
        &mut self,
        recent_segments: &[TranscriptSegment],
        notes: &MeetingNotes,
        context: Option<&str>,
        participants: &[String],
        on_event: &mut dyn FnMut(SummarizeEvent),
    ) -> Result<(), SummarizeError>;
}

pub fn create_summarize_provider(
    provider: &str,
    model: Option<&str>,
    api_key: Option<&str>,
) -> Result<Box<dyn SummarizeProvider>, SummarizeError> {
    match provider {
        "ollama" => Ok(Box::new(local::OllamaProvider::new(
            model.unwrap_or(DEFAULT_OLLAMA_MODEL),
        )?)),
        "openrouter" => Ok(Box::new(cloud::OpenRouterProvider::new(model, api_key)?)),
        other => Err(SummarizeError::Failed(format!(
            "unknown summarize provider: {other}"
        ))),
    }
}
