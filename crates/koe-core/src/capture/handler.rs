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
}

/// Shared state between system and microphone output handlers.
struct SharedHandlerState {
    system_producer: Mutex<AudioProducer>,
    mic_producer: Mutex<AudioProducer>,
}

/// RT-safe ScreenCaptureKit output handler.
/// Copies f32 audio samples into SPSC ring buffers.
/// Each instance holds a shared reference to the producer state.
pub struct OutputHandler {
    state: Arc<SharedHandlerState>,
}

/// Create a pair of output handlers (system audio + microphone) with their paired consumer rings.
/// Returns two handlers so each can be registered for a separate `SCStreamOutputType`.
pub fn create_output_handlers() -> (OutputHandler, OutputHandler, AudioRing, AudioRing) {
    let (sys_samples_prod, sys_samples_cons) = rtrb::RingBuffer::new(RING_CAPACITY);
    let (sys_pts_prod, sys_pts_cons) = rtrb::RingBuffer::new(PTS_RING_CAPACITY);

    let (mic_samples_prod, mic_samples_cons) = rtrb::RingBuffer::new(RING_CAPACITY);
    let (mic_pts_prod, mic_pts_cons) = rtrb::RingBuffer::new(PTS_RING_CAPACITY);

    let state = Arc::new(SharedHandlerState {
        system_producer: Mutex::new(AudioProducer {
            samples: sys_samples_prod,
            pts: sys_pts_prod,
            drop_count: Arc::new(AtomicU64::new(0)),
        }),
        mic_producer: Mutex::new(AudioProducer {
            samples: mic_samples_prod,
            pts: mic_pts_prod,
            drop_count: Arc::new(AtomicU64::new(0)),
        }),
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

        let Ok(mut producer) = mutex.lock() else {
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

            // Convert to mono by averaging channels if stereo
            let mono_samples: Vec<f32>;
            let samples: &[f32] = if channels > 1 {
                let frame_count = f32_slice.len() / channels;
                mono_samples = (0..frame_count)
                    .map(|i| {
                        let sum: f32 = (0..channels).map(|ch| f32_slice[i * channels + ch]).sum();
                        sum / channels as f32
                    })
                    .collect();
                &mono_samples
            } else {
                f32_slice
            };

            // Check if ring has space
            let available = producer.samples.slots();
            if available < samples.len() {
                producer.drop_count.fetch_add(1, Ordering::Relaxed);
                return;
            }

            // Write samples
            let mut written = 0;
            while written < samples.len() {
                match producer.samples.write_chunk_uninit(samples.len() - written) {
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
                        return;
                    }
                }
            }

            // Record PTS entry
            let _ = producer.pts.push((pts_ns, samples.len()));
        }
    }
}
