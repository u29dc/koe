pub mod ollama;
pub mod openrouter;
mod patch;

use crate::SummarizerError;
use crate::types::{MeetingState, SummarizerEvent, TranscriptSegment};

const DEFAULT_OLLAMA_MODEL: &str = "qwen3:30b-a3b";

pub trait SummarizerProvider: Send {
    fn name(&self) -> &'static str;
    fn summarize(
        &mut self,
        recent_segments: &[TranscriptSegment],
        state: &MeetingState,
        on_event: &mut dyn FnMut(SummarizerEvent),
    ) -> Result<(), SummarizerError>;
}

pub fn create_summarizer(
    provider: &str,
    model: Option<&str>,
) -> Result<Box<dyn SummarizerProvider>, SummarizerError> {
    match provider {
        "ollama" => Ok(Box::new(ollama::OllamaProvider::new(
            model.unwrap_or(DEFAULT_OLLAMA_MODEL),
        )?)),
        "openrouter" => Ok(Box::new(openrouter::OpenRouterProvider::new(model)?)),
        other => Err(SummarizerError::Failed(format!(
            "unknown summarizer provider: {other}"
        ))),
    }
}
