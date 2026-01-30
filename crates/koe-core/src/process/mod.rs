pub mod chunker;
mod queue;
pub mod resample;
pub mod vad;

use crate::capture::AudioCapture;
use crate::error::ProcessError;
use crate::types::{AudioSource, CaptureStats};
use chunker::Chunker;
use queue::{ChunkReceiver, ChunkSender, SendOutcome, chunk_channel};
use resample::ResampleConverter;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use vad::VadDetector;

/// Resampler chunk size at 48 kHz (10 ms).
const RESAMPLE_CHUNK: usize = 480;

/// Audio processor: drains capture ring buffers, resamples, runs VAD, and emits chunks.
pub struct AudioProcessor {
    running: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

struct StreamPipeline {
    resampler: ResampleConverter,
    vad: VadDetector,
    chunker: Chunker,
    resample_remainder: Vec<f32>,
    vad_remainder: Vec<f32>,
}

impl StreamPipeline {
    fn new(source: AudioSource) -> Result<Self, ProcessError> {
        Ok(Self {
            resampler: ResampleConverter::new()?,
            vad: VadDetector::new()?,
            chunker: Chunker::new(source),
            resample_remainder: Vec::new(),
            vad_remainder: Vec::new(),
        })
    }

    fn process(
        &mut self,
        input_48k: &[f32],
        pts_ns: i128,
        chunk_tx: &ChunkSender,
        stats: &CaptureStats,
    ) {
        // Prepend remainder and feed complete chunks to resampler
        self.resample_remainder.extend_from_slice(input_48k);

        let full_chunks = (self.resample_remainder.len() / RESAMPLE_CHUNK) * RESAMPLE_CHUNK;
        if full_chunks == 0 {
            return;
        }

        let to_resample = &self.resample_remainder[..full_chunks];
        let resampled = match self.resampler.process(to_resample) {
            Ok(r) => r,
            Err(_) => return,
        };
        self.resample_remainder.drain(..full_chunks);

        // Feed resampled 16 kHz through VAD in 512-sample frames
        self.vad_remainder.extend_from_slice(&resampled);

        const VAD_FRAME: usize = 512;
        let mut offset = 0;
        while offset + VAD_FRAME <= self.vad_remainder.len() {
            let frame = &self.vad_remainder[offset..offset + VAD_FRAME];
            let is_speech = self.vad.process_frame(frame);

            if let Some(chunk) = self.chunker.push(frame, pts_ns, is_speech) {
                stats.inc_chunks_emitted();
                match chunk_tx.send_drop_oldest(chunk) {
                    SendOutcome::Sent => {}
                    SendOutcome::DroppedOldest => {
                        stats.inc_chunks_dropped();
                    }
                    SendOutcome::Disconnected => return,
                }
            }

            offset += VAD_FRAME;
        }
        self.vad_remainder.drain(..offset);
    }

    #[cfg(test)]
    fn process_with_speech(
        &mut self,
        input_48k: &[f32],
        pts_ns: i128,
        chunk_tx: &ChunkSender,
        stats: &CaptureStats,
        speech: bool,
    ) {
        // Prepend remainder and feed complete chunks to resampler
        self.resample_remainder.extend_from_slice(input_48k);

        let full_chunks = (self.resample_remainder.len() / RESAMPLE_CHUNK) * RESAMPLE_CHUNK;
        if full_chunks == 0 {
            return;
        }

        let to_resample = &self.resample_remainder[..full_chunks];
        let resampled = match self.resampler.process(to_resample) {
            Ok(r) => r,
            Err(_) => return,
        };
        self.resample_remainder.drain(..full_chunks);

        self.vad_remainder.extend_from_slice(&resampled);

        const VAD_FRAME: usize = 512;
        let mut offset = 0;
        while offset + VAD_FRAME <= self.vad_remainder.len() {
            let frame = &self.vad_remainder[offset..offset + VAD_FRAME];

            if let Some(chunk) = self.chunker.push(frame, pts_ns, speech) {
                stats.inc_chunks_emitted();
                match chunk_tx.send_drop_oldest(chunk) {
                    SendOutcome::Sent => {}
                    SendOutcome::DroppedOldest => {
                        stats.inc_chunks_dropped();
                    }
                    SendOutcome::Disconnected => return,
                }
            }

            offset += VAD_FRAME;
        }
        self.vad_remainder.drain(..offset);
    }

    fn flush(&mut self, chunk_tx: &ChunkSender, stats: &CaptureStats) {
        if let Some(chunk) = self.chunker.flush() {
            stats.inc_chunks_emitted();
            if chunk_tx.send_drop_oldest(chunk) == SendOutcome::DroppedOldest {
                stats.inc_chunks_dropped();
            }
        }
    }
}

impl AudioProcessor {
    /// Start the processor thread. Returns a receiver for audio chunks.
    pub fn start(
        mut capture: Box<dyn AudioCapture>,
        stats: CaptureStats,
    ) -> Result<(Self, ChunkReceiver), ProcessError> {
        capture.start().map_err(ProcessError::Capture)?;

        let (chunk_tx, chunk_rx) = chunk_channel(4);
        let running = Arc::new(AtomicBool::new(true));
        let running_clone = Arc::clone(&running);

        let mut system_pipeline = StreamPipeline::new(AudioSource::System)?;
        let mut mic_pipeline = StreamPipeline::new(AudioSource::Microphone)?;

        let thread = thread::Builder::new()
            .name("koe-audio-processor".into())
            .spawn(move || {
                while running_clone.load(Ordering::Relaxed) {
                    let mut had_data = false;

                    if let Some(frame) = capture.try_recv_system() {
                        stats.inc_frames_captured();
                        system_pipeline.process(
                            &frame.samples_f32,
                            frame.pts_ns,
                            &chunk_tx,
                            &stats,
                        );
                        had_data = true;
                    }

                    if let Some(frame) = capture.try_recv_mic() {
                        stats.inc_frames_captured();
                        mic_pipeline.process(&frame.samples_f32, frame.pts_ns, &chunk_tx, &stats);
                        had_data = true;
                    }

                    if !had_data {
                        thread::sleep(std::time::Duration::from_millis(2));
                    }
                }

                // Flush remaining data
                system_pipeline.flush(&chunk_tx, &stats);
                mic_pipeline.flush(&chunk_tx, &stats);
                capture.stop();
            })
            .map_err(|e| ProcessError::ResamplerInit(format!("thread spawn failed: {e}")))?;

        Ok((
            Self {
                running,
                thread: Some(thread),
            },
            chunk_rx,
        ))
    }

    /// Signal the processor to stop and wait for the thread to finish.
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for AudioProcessor {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::{RESAMPLE_CHUNK, StreamPipeline, chunk_channel};
    use crate::types::{AudioSource, CaptureStats};

    #[test]
    fn system_audio_increments_chunk_counters() {
        let mut pipeline = StreamPipeline::new(AudioSource::System).unwrap();
        let stats = CaptureStats::new();
        let (chunk_tx, _chunk_rx) = chunk_channel(4);

        let input = vec![0.1f32; RESAMPLE_CHUNK * 700];
        pipeline.process_with_speech(&input, 0, &chunk_tx, &stats, true);

        assert!(stats.chunks_emitted() > 0);
        assert_eq!(stats.chunks_dropped(), 0);
    }
}
