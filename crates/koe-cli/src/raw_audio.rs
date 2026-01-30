use koe_core::types::AudioSource;
use std::collections::VecDeque;
use std::io::{BufWriter, Write};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

pub struct RawAudioWriter {
    file: BufWriter<std::fs::File>,
    system: VecDeque<f32>,
    mic: VecDeque<f32>,
    pending_flush_samples: usize,
    last_system_at: Option<Instant>,
    last_mic_at: Option<Instant>,
}

impl RawAudioWriter {
    const FLUSH_SAMPLES: usize = 48_000;
    const MISSING_SOURCE_TIMEOUT: Duration = Duration::from_millis(500);

    pub fn new(file: std::fs::File) -> Self {
        Self {
            file: BufWriter::new(file),
            system: VecDeque::new(),
            mic: VecDeque::new(),
            pending_flush_samples: 0,
            last_system_at: None,
            last_mic_at: None,
        }
    }

    pub fn write_samples(&mut self, source: AudioSource, samples: &[f32]) -> std::io::Result<()> {
        match source {
            AudioSource::System => {
                self.system.extend(samples.iter().copied());
                self.last_system_at = Some(Instant::now());
            }
            AudioSource::Microphone => {
                self.mic.extend(samples.iter().copied());
                self.last_mic_at = Some(Instant::now());
            }
            AudioSource::Mixed => {
                self.mix_available()?;
                self.write_samples_inner(samples)?;
                return Ok(());
            }
        }
        self.mix_available()?;
        self.drain_if_missing_source()
    }

    fn mix_available(&mut self) -> std::io::Result<()> {
        let mix_len = self.system.len().min(self.mic.len());
        for _ in 0..mix_len {
            let left = self.system.pop_front().unwrap_or(0.0);
            let right = self.mic.pop_front().unwrap_or(0.0);
            let mixed = ((left + right) * 0.5).clamp(-1.0, 1.0);
            self.write_sample(mixed)?;
        }
        Ok(())
    }

    fn drain_if_missing_source(&mut self) -> std::io::Result<()> {
        if self.system.is_empty()
            && !self.mic.is_empty()
            && self.source_timed_out(self.last_system_at)
        {
            self.drain_remaining_source(AudioSource::Microphone)?;
        }
        if self.mic.is_empty() && !self.system.is_empty() && self.source_timed_out(self.last_mic_at)
        {
            self.drain_remaining_source(AudioSource::System)?;
        }
        Ok(())
    }

    fn source_timed_out(&self, last_seen: Option<Instant>) -> bool {
        match last_seen {
            Some(at) => at.elapsed() > Self::MISSING_SOURCE_TIMEOUT,
            None => true,
        }
    }

    fn drain_remaining_source(&mut self, source: AudioSource) -> std::io::Result<()> {
        match source {
            AudioSource::System => {
                while let Some(sample) = self.system.pop_front() {
                    self.write_sample(sample)?;
                }
            }
            AudioSource::Microphone => {
                while let Some(sample) = self.mic.pop_front() {
                    self.write_sample(sample)?;
                }
            }
            AudioSource::Mixed => {}
        }
        Ok(())
    }

    fn write_samples_inner(&mut self, samples: &[f32]) -> std::io::Result<()> {
        for sample in samples {
            self.write_sample(*sample)?;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> std::io::Result<()> {
        self.mix_available()?;
        self.drain_remaining_source(AudioSource::System)?;
        self.drain_remaining_source(AudioSource::Microphone)?;
        self.file.flush()?;
        self.pending_flush_samples = 0;
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
            if let Some(existing) = guard.as_mut() {
                let _ = existing.flush();
            }
            *guard = writer;
        }
    }

    pub fn write_samples(&self, source: AudioSource, samples: &[f32]) -> std::io::Result<()> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| std::io::Error::other("raw audio writer lock poisoned"))?;
        if let Some(writer) = guard.as_mut() {
            writer.write_samples(source, samples)?;
        }
        Ok(())
    }
}

pub struct RawAudioMessage {
    pub source: AudioSource,
    pub samples: Vec<f32>,
}

pub fn spawn_raw_audio_writer(
    rx: Receiver<RawAudioMessage>,
    writer: SharedRawAudioWriter,
) -> std::io::Result<JoinHandle<()>> {
    thread::Builder::new()
        .name("koe-raw-audio-writer".into())
        .spawn(move || {
            while let Ok(msg) = rx.recv() {
                let _ = writer.write_samples(msg.source, &msg.samples);
            }
        })
}
