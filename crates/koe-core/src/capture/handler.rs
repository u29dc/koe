use crate::types::CaptureStats;
use screencapturekit::cm::CMSampleBuffer;
use screencapturekit::stream::output_trait::SCStreamOutputTrait;
use screencapturekit::stream::output_type::SCStreamOutputType;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Ring buffer capacity: 10 seconds at 48 kHz mono.
const RING_CAPACITY: usize = 480_000;

/// PTS ring capacity: one entry per callback.
const PTS_RING_CAPACITY: usize = 1_000;

/// Consumer side of an audio ring buffer.
pub struct AudioRing {
    pub samples: rtrb::Consumer<f32>,
    pub pts: rtrb::Consumer<(i128, usize)>,
}

struct AudioProducer {
    samples: rtrb::Producer<f32>,
    pts: rtrb::Producer<(i128, usize)>,
    drop_count: Arc<AtomicU64>,
    mono_scratch: Vec<f32>,
}

/// Shared state between system and microphone output handlers.
struct SharedHandlerState {
    system_producer: Mutex<AudioProducer>,
    mic_producer: Mutex<AudioProducer>,
    stats: CaptureStats,
}

/// RT-safe ScreenCaptureKit output handler.
/// Copies f32 audio samples into SPSC ring buffers.
/// Each instance holds a shared reference to the producer state.
pub struct OutputHandler {
    state: Arc<SharedHandlerState>,
}

/// Create a pair of output handlers (system audio + microphone) with their paired consumer rings.
/// Returns two handlers so each can be registered for a separate `SCStreamOutputType`.
pub fn create_output_handlers(
    stats: CaptureStats,
) -> (OutputHandler, OutputHandler, AudioRing, AudioRing) {
    let (sys_samples_prod, sys_samples_cons) = rtrb::RingBuffer::new(RING_CAPACITY);
    let (sys_pts_prod, sys_pts_cons) = rtrb::RingBuffer::new(PTS_RING_CAPACITY);

    let (mic_samples_prod, mic_samples_cons) = rtrb::RingBuffer::new(RING_CAPACITY);
    let (mic_pts_prod, mic_pts_cons) = rtrb::RingBuffer::new(PTS_RING_CAPACITY);

    let state = Arc::new(SharedHandlerState {
        system_producer: Mutex::new(AudioProducer {
            samples: sys_samples_prod,
            pts: sys_pts_prod,
            drop_count: Arc::new(AtomicU64::new(0)),
            mono_scratch: Vec::with_capacity(RING_CAPACITY),
        }),
        mic_producer: Mutex::new(AudioProducer {
            samples: mic_samples_prod,
            pts: mic_pts_prod,
            drop_count: Arc::new(AtomicU64::new(0)),
            mono_scratch: Vec::with_capacity(RING_CAPACITY),
        }),
        stats,
    });

    let system_handler = OutputHandler {
        state: Arc::clone(&state),
    };
    let mic_handler = OutputHandler { state };

    let system_ring = AudioRing {
        samples: sys_samples_cons,
        pts: sys_pts_cons,
    };

    let mic_ring = AudioRing {
        samples: mic_samples_cons,
        pts: mic_pts_cons,
    };

    (system_handler, mic_handler, system_ring, mic_ring)
}

impl SCStreamOutputTrait for OutputHandler {
    fn did_output_sample_buffer(&self, sample: CMSampleBuffer, of_type: SCStreamOutputType) {
        let mutex = match of_type {
            SCStreamOutputType::Audio => &self.state.system_producer,
            SCStreamOutputType::Microphone => &self.state.mic_producer,
            SCStreamOutputType::Screen => return,
        };

        let Ok(mut producer) = mutex.try_lock() else {
            self.state.stats.inc_frames_dropped();
            return;
        };

        let Some(audio_list) = sample.audio_buffer_list() else {
            return;
        };

        let pts = sample.presentation_timestamp();
        let pts_ns = if pts.timescale > 0 {
            (pts.value as i128 * 1_000_000_000) / pts.timescale as i128
        } else {
            0
        };

        for buffer in audio_list.iter() {
            let raw_bytes = buffer.data();
            let channels = buffer.number_channels as usize;

            if raw_bytes.is_empty() || channels == 0 {
                continue;
            }

            // Audio data is f32, possibly interleaved
            let f32_count = raw_bytes.len() / std::mem::size_of::<f32>();
            let f32_slice =
                unsafe { std::slice::from_raw_parts(raw_bytes.as_ptr().cast::<f32>(), f32_count) };

            let AudioProducer {
                samples: samples_prod,
                pts: pts_prod,
                drop_count,
                mono_scratch,
            } = &mut *producer;

            let samples = match downmix_to_mono(f32_slice, channels, mono_scratch) {
                Some(samples) => samples,
                None => {
                    drop_count.fetch_add(1, Ordering::Relaxed);
                    self.state.stats.inc_frames_dropped();
                    return;
                }
            };

            // Check if ring has space
            let available = samples_prod.slots();
            if available < samples.len() {
                drop_count.fetch_add(1, Ordering::Relaxed);
                self.state.stats.inc_frames_dropped();
                return;
            }

            // Write samples
            let mut written = 0;
            while written < samples.len() {
                match samples_prod.write_chunk_uninit(samples.len() - written) {
                    Ok(mut chunk) => {
                        let len = chunk.len();
                        let (first, second) = chunk.as_mut_slices();
                        let first_len = first.len();
                        for (i, slot) in first.iter_mut().enumerate() {
                            slot.write(samples[written + i]);
                        }
                        for (i, slot) in second.iter_mut().enumerate() {
                            slot.write(samples[written + first_len + i]);
                        }
                        unsafe { chunk.commit_all() };
                        written += len;
                    }
                    Err(_) => {
                        producer.drop_count.fetch_add(1, Ordering::Relaxed);
                        self.state.stats.inc_frames_dropped();
                        return;
                    }
                }
            }

            // Record PTS entry
            let _ = pts_prod.push((pts_ns, samples.len()));
        }
    }
}

fn downmix_to_mono<'a>(
    interleaved: &'a [f32],
    channels: usize,
    scratch: &'a mut Vec<f32>,
) -> Option<&'a [f32]> {
    if interleaved.is_empty() {
        return Some(&[]);
    }

    if channels == 0 {
        return None;
    }

    if channels == 1 {
        return Some(interleaved);
    }

    let frames = interleaved.len() / channels;
    if frames == 0 {
        return Some(&[]);
    }

    if frames > scratch.capacity() {
        return None;
    }

    scratch.clear();
    scratch.resize(frames, 0.0);

    for (frame, slot) in scratch.iter_mut().enumerate().take(frames) {
        let mut sum = 0.0;
        let base = frame * channels;
        for ch in 0..channels {
            sum += interleaved[base + ch];
        }
        *slot = sum / channels as f32;
    }

    Some(&scratch[..frames])
}

#[cfg(test)]
mod tests {
    use super::downmix_to_mono;

    #[test]
    fn downmix_returns_input_for_mono() {
        let mut scratch = Vec::with_capacity(8);
        let input = [0.1, -0.2, 0.3];
        let output = downmix_to_mono(&input, 1, &mut scratch).unwrap();
        assert_eq!(output, &input);
        assert!(scratch.is_empty());
    }

    #[test]
    fn downmix_averages_channels() {
        let mut scratch = Vec::with_capacity(4);
        let input = [1.0, 3.0, 2.0, 4.0];
        let output = downmix_to_mono(&input, 2, &mut scratch).unwrap();
        assert_eq!(output, &[2.0, 3.0]);
    }

    #[test]
    fn downmix_refuses_to_allocate() {
        let mut scratch = Vec::with_capacity(1);
        let input = [1.0, 3.0, 2.0, 4.0];
        assert!(downmix_to_mono(&input, 2, &mut scratch).is_none());
        assert_eq!(scratch.capacity(), 1);
    }
}
