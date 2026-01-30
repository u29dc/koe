use koe_core::types::{AudioFrame, AudioSource};
use std::collections::VecDeque;
use std::io::{BufWriter, Write};
use std::sync::{Arc, Mutex};

pub struct RawAudioWriter {
    file: BufWriter<std::fs::File>,
    system: VecDeque<f32>,
    mic: VecDeque<f32>,
    pending_flush_samples: usize,
}

impl RawAudioWriter {
    const FLUSH_SAMPLES: usize = 48_000;

    pub fn new(file: std::fs::File) -> Self {
        Self {
            file: BufWriter::new(file),
            system: VecDeque::new(),
            mic: VecDeque::new(),
            pending_flush_samples: 0,
        }
    }

    pub fn write_frame(&mut self, source: AudioSource, frame: &AudioFrame) -> std::io::Result<()> {
        match source {
            AudioSource::System => self.system.extend(frame.samples_f32.iter().copied()),
            AudioSource::Microphone => self.mic.extend(frame.samples_f32.iter().copied()),
            AudioSource::Mixed => {
                self.drain_buffers()?;
                self.write_samples(&frame.samples_f32)?;
                return Ok(());
            }
        }
        self.drain_buffers()
    }

    fn drain_buffers(&mut self) -> std::io::Result<()> {
        let mix_len = self.system.len().min(self.mic.len());
        for _ in 0..mix_len {
            let left = self.system.pop_front().unwrap_or(0.0);
            let right = self.mic.pop_front().unwrap_or(0.0);
            let mixed = ((left + right) * 0.5).clamp(-1.0, 1.0);
            self.write_sample(mixed)?;
        }

        if self.mic.is_empty() {
            while let Some(sample) = self.system.pop_front() {
                self.write_sample(sample)?;
            }
        }
        if self.system.is_empty() {
            while let Some(sample) = self.mic.pop_front() {
                self.write_sample(sample)?;
            }
        }

        Ok(())
    }

    fn write_samples(&mut self, samples: &[f32]) -> std::io::Result<()> {
        for sample in samples {
            self.write_sample(*sample)?;
        }
        Ok(())
    }

    fn write_sample(&mut self, sample: f32) -> std::io::Result<()> {
        self.file.write_all(&sample.to_le_bytes())?;
        self.pending_flush_samples += 1;
        if self.pending_flush_samples >= Self::FLUSH_SAMPLES {
            self.file.flush()?;
            self.pending_flush_samples = 0;
        }
        Ok(())
    }
}

#[derive(Clone, Default)]
pub struct SharedRawAudioWriter {
    inner: Arc<Mutex<Option<RawAudioWriter>>>,
}

impl SharedRawAudioWriter {
    pub fn new(writer: Option<RawAudioWriter>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(writer)),
        }
    }

    pub fn set(&self, writer: Option<RawAudioWriter>) {
        if let Ok(mut guard) = self.inner.lock() {
            *guard = writer;
        }
    }

    pub fn write_frame(&self, source: AudioSource, frame: &AudioFrame) -> std::io::Result<()> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| std::io::Error::other("raw audio writer lock poisoned"))?;
        if let Some(writer) = guard.as_mut() {
            writer.write_frame(source, frame)?;
        }
        Ok(())
    }
}
