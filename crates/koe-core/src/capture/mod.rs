mod handler;
mod sck;

use crate::error::CaptureError;
use crate::types::AudioFrame;

/// Trait for audio capture backends.
pub trait AudioCapture: Send {
    fn start(&mut self) -> Result<(), CaptureError>;
    fn stop(&mut self);
    fn try_recv_system(&mut self) -> Option<AudioFrame>;
    fn try_recv_mic(&mut self) -> Option<AudioFrame>;
}

/// Create the platform-specific audio capture backend.
pub fn create_capture() -> Result<Box<dyn AudioCapture>, CaptureError> {
    Ok(Box::new(sck::SckCapture::new()?))
}
