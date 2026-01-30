use crate::capture::AudioCapture;
use crate::capture::handler::{self, AudioRing};
use crate::error::CaptureError;
use crate::types::{AudioFrame, CaptureStats};
use screencapturekit::shareable_content::SCShareableContent;
use screencapturekit::stream::configuration::SCStreamConfiguration;
use screencapturekit::stream::content_filter::SCContentFilter;
use screencapturekit::stream::output_type::SCStreamOutputType;
use screencapturekit::stream::sc_stream::SCStream;

const SAMPLE_RATE: u32 = 48_000;

/// ScreenCaptureKit-based audio capture for macOS.
pub struct SckCapture {
    stream: Option<SCStream>,
    system_ring: AudioRing,
    mic_ring: AudioRing,
}

impl SckCapture {
    pub fn new(stats: CaptureStats) -> Result<Self, CaptureError> {
        let (system_handler, mic_handler, system_ring, mic_ring) =
            handler::create_output_handlers(stats);

        // Enumerate displays
        let content =
            SCShareableContent::get().map_err(|e| CaptureError::Backend(format!("{e:?}")))?;
        let displays = content.displays();
        let display = displays.first().ok_or(CaptureError::NoDisplay)?;

        // Build content filter for the primary display
        let filter = SCContentFilter::create()
            .with_display(display)
            .with_excluding_windows(&[])
            .build();

        // Configure: mono 48 kHz, audio + microphone, exclude own audio, minimal video
        let config = SCStreamConfiguration::new()
            .with_width(2)
            .with_height(2)
            .with_captures_audio(true)
            .with_captures_microphone(true)
            .with_excludes_current_process_audio(true)
            .with_sample_rate(SAMPLE_RATE as i32)
            .with_channel_count(1);

        // Create stream and add output handlers for system audio and microphone
        let mut stream = SCStream::new(&filter, &config);
        stream.add_output_handler(system_handler, SCStreamOutputType::Audio);
        stream.add_output_handler(mic_handler, SCStreamOutputType::Microphone);

        Ok(Self {
            stream: Some(stream),
            system_ring,
            mic_ring,
        })
    }

    fn drain_ring(ring: &mut AudioRing) -> Option<AudioFrame> {
        let available = ring.samples.slots();
        if available == 0 {
            return None;
        }

        let mut samples = Vec::with_capacity(available);
        while let Ok(s) = ring.samples.pop() {
            samples.push(s);
        }

        // Get the most recent PTS
        let mut pts_ns: i128 = 0;
        while let Ok((pts, _len)) = ring.pts.pop() {
            pts_ns = pts;
        }

        if samples.is_empty() {
            return None;
        }

        Some(AudioFrame {
            pts_ns,
            sample_rate_hz: SAMPLE_RATE,
            channels: 1,
            samples_f32: samples,
        })
    }
}

impl AudioCapture for SckCapture {
    fn start(&mut self) -> Result<(), CaptureError> {
        if let Some(ref mut stream) = self.stream {
            stream
                .start_capture()
                .map_err(|e| CaptureError::StartFailed(format!("{e:?}")))?;
        }
        Ok(())
    }

    fn stop(&mut self) {
        if let Some(ref mut stream) = self.stream {
            let _ = stream.stop_capture();
        }
    }

    fn try_recv_system(&mut self) -> Option<AudioFrame> {
        Self::drain_ring(&mut self.system_ring)
    }

    fn try_recv_mic(&mut self) -> Option<AudioFrame> {
        Self::drain_ring(&mut self.mic_ring)
    }
}
