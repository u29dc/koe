mod handler;
mod sck;

use crate::error::CaptureError;
use crate::types::{AudioFrame, CaptureStats};
use screencapturekit::audio_devices::AudioInputDevice;

/// Trait for audio capture backends.
pub trait AudioCapture: Send {
    fn start(&mut self) -> Result<(), CaptureError>;
    fn stop(&mut self);
    fn try_recv_system(&mut self) -> Option<AudioFrame>;
    fn try_recv_mic(&mut self) -> Option<AudioFrame>;
}

#[derive(Debug, Clone)]
pub struct CaptureConfig {
    pub capture_system: bool,
    pub capture_microphone: bool,
    pub microphone_device_id: Option<String>,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            capture_system: true,
            capture_microphone: true,
            microphone_device_id: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AudioInputDeviceInfo {
    pub id: String,
    pub name: String,
    pub is_default: bool,
}

pub fn list_audio_inputs() -> Vec<AudioInputDeviceInfo> {
    AudioInputDevice::list()
        .into_iter()
        .map(|device| AudioInputDeviceInfo {
            id: device.id,
            name: device.name,
            is_default: device.is_default,
        })
        .collect()
}

/// Create the platform-specific audio capture backend.
pub fn create_capture(
    stats: CaptureStats,
    config: CaptureConfig,
) -> Result<Box<dyn AudioCapture>, CaptureError> {
    Ok(Box::new(sck::SckCapture::new(stats, config)?))
}
