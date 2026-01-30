use crate::capture::AudioCapture;
use crate::capture::handler::{self, AudioRing};
use crate::error::CaptureError;
use crate::types::{AudioFrame, CaptureStats};
use screencapturekit::shareable_content::SCShareableContent;
use screencapturekit::stream::configuration::SCStreamConfiguration;
use screencapturekit::stream::content_filter::SCContentFilter;
use screencapturekit::stream::output_type::SCStreamOutputType;
use screencapturekit::stream::sc_stream::SCStream;
use std::collections::VecDeque;

const SAMPLE_RATE: u32 = 48_000;

/// ScreenCaptureKit-based audio capture for macOS.
pub struct SckCapture {
    stream: Option<SCStream>,
    system_ring: AudioRing,
    mic_ring: AudioRing,
    system_pts: PtsTracker,
    mic_pts: PtsTracker,
    capture_system: bool,
    capture_microphone: bool,
}

impl SckCapture {
    pub fn new(
        stats: CaptureStats,
        capture_config: crate::capture::CaptureConfig,
    ) -> Result<Self, CaptureError> {
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
        let mut stream_config = SCStreamConfiguration::new()
            .with_width(2)
            .with_height(2)
            .with_captures_audio(capture_config.capture_system)
            .with_captures_microphone(capture_config.capture_microphone)
            .with_excludes_current_process_audio(true)
            .with_sample_rate(SAMPLE_RATE as i32)
            .with_channel_count(1);

        if capture_config.capture_microphone
            && let Some(device_id) = capture_config.microphone_device_id.as_deref()
            && !device_id.trim().is_empty()
        {
            stream_config.set_microphone_capture_device_id(device_id);
        }

        // Create stream and add output handlers for system audio and microphone
        let mut stream = SCStream::new(&filter, &stream_config);
        if capture_config.capture_system {
            stream.add_output_handler(system_handler, SCStreamOutputType::Audio);
        }
        if capture_config.capture_microphone {
            stream.add_output_handler(mic_handler, SCStreamOutputType::Microphone);
        }

        Ok(Self {
            stream: Some(stream),
            system_ring,
            mic_ring,
            system_pts: PtsTracker::new(),
            mic_pts: PtsTracker::new(),
            capture_system: capture_config.capture_system,
            capture_microphone: capture_config.capture_microphone,
        })
    }

    fn drain_ring(ring: &mut AudioRing, pts: &mut PtsTracker) -> Option<AudioFrame> {
        let available = ring.samples.slots();
        if available == 0 {
            return None;
        }

        let mut samples = Vec::with_capacity(available);
        while let Ok(s) = ring.samples.pop() {
            samples.push(s);
        }

        while let Ok(entry) = ring.pts.pop() {
            pts.push(entry);
        }

        if samples.is_empty() {
            return None;
        }

        let pts_ns = pts.start_pts(samples.len(), SAMPLE_RATE);

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
        if !self.capture_system {
            return None;
        }
        Self::drain_ring(&mut self.system_ring, &mut self.system_pts)
    }

    fn try_recv_mic(&mut self) -> Option<AudioFrame> {
        if !self.capture_microphone {
            return None;
        }
        Self::drain_ring(&mut self.mic_ring, &mut self.mic_pts)
    }
}

struct PtsTracker {
    pending: VecDeque<(i128, usize)>,
    pending_offset: usize,
    last_pts_ns: i128,
}

impl PtsTracker {
    fn new() -> Self {
        Self {
            pending: VecDeque::new(),
            pending_offset: 0,
            last_pts_ns: 0,
        }
    }

    fn push(&mut self, entry: (i128, usize)) {
        self.last_pts_ns = entry.0;
        self.pending.push_back(entry);
    }

    fn start_pts(&mut self, samples: usize, sample_rate: u32) -> i128 {
        if samples == 0 {
            return self.last_pts_ns;
        }

        let mut remaining = samples;
        let mut start_pts = None;

        while remaining > 0 {
            let Some((pts, len)) = self.pending.front().copied() else {
                break;
            };

            let offset = self.pending_offset.min(len);
            if start_pts.is_none() {
                let offset_ns = (offset as i128 * 1_000_000_000) / sample_rate as i128;
                start_pts = Some(pts + offset_ns);
            }

            let available = len.saturating_sub(offset);
            if remaining >= available {
                remaining -= available;
                self.pending.pop_front();
                self.pending_offset = 0;
            } else {
                self.pending_offset += remaining;
                remaining = 0;
            }
        }

        start_pts.unwrap_or(self.last_pts_ns)
    }
}

#[cfg(test)]
mod tests {
    use super::PtsTracker;

    #[test]
    fn start_pts_tracks_offsets() {
        let mut tracker = PtsTracker::new();
        tracker.push((1_000, 4));
        tracker.push((2_000, 4));

        let start = tracker.start_pts(2, 4);
        assert_eq!(start, 1_000);

        let start = tracker.start_pts(4, 4);
        assert_eq!(start, 1_000 + 500_000_000);

        let start = tracker.start_pts(2, 4);
        assert_eq!(start, 2_000 + 500_000_000);
    }

    #[test]
    fn start_pts_uses_last_when_missing() {
        let mut tracker = PtsTracker::new();
        tracker.push((5_000, 2));
        let _ = tracker.start_pts(2, 4);
        let start = tracker.start_pts(3, 4);
        assert_eq!(start, 5_000);
    }
}
