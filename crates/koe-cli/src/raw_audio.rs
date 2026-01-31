use crate::config::{AgcConfig, DenoiseConfig, MixdownConfig};
use koe_core::types::AudioSource;
use std::collections::VecDeque;
use std::f32::consts::PI;
use std::io::{BufWriter, Write};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const AGC_RMS_WINDOW_MS: f32 = 50.0;
const AGC_PEAK_RELEASE_MS: f32 = 50.0;
const DENOISE_RMS_WINDOW_MS: f32 = 30.0;
const EPSILON: f32 = 1.0e-6;

fn db_to_gain(db: f32) -> f32 {
    10_f32.powf(db / 20.0)
}

fn smoothing_coeff(sample_rate_hz: f32, time_ms: f32) -> f32 {
    if time_ms <= 0.0 || sample_rate_hz <= 0.0 {
        return 1.0;
    }
    let tau = time_ms / 1000.0;
    1.0 - (-1.0 / (tau * sample_rate_hz)).exp()
}

fn decay_coeff(sample_rate_hz: f32, time_ms: f32) -> f32 {
    if time_ms <= 0.0 || sample_rate_hz <= 0.0 {
        return 0.0;
    }
    (-1.0 / ((time_ms / 1000.0) * sample_rate_hz)).exp()
}

struct MixdownProcessor {
    agc: Option<AgcProcessor>,
    denoise: Option<NoiseReducer>,
    high_pass: Option<HighPassFilter>,
}

impl MixdownProcessor {
    fn new(sample_rate_hz: u32, config: &MixdownConfig) -> Self {
        let agc = if config.agc.enabled {
            Some(AgcProcessor::new(sample_rate_hz, &config.agc))
        } else {
            None
        };
        let denoise = if config.denoise.enabled {
            Some(NoiseReducer::new(sample_rate_hz, &config.denoise))
        } else {
            None
        };
        let high_pass = if config.high_pass.enabled {
            HighPassFilter::new(sample_rate_hz, config.high_pass.cutoff_hz)
        } else {
            None
        };
        Self {
            agc,
            denoise,
            high_pass,
        }
    }

    fn process(&mut self, sample: f32) -> f32 {
        let mut output = sample;
        if let Some(high_pass) = self.high_pass.as_mut() {
            output = high_pass.process(output);
        }
        if let Some(denoise) = self.denoise.as_mut() {
            output = denoise.process(output);
        }
        if let Some(agc) = self.agc.as_mut() {
            output = agc.process(output);
        }
        output
    }
}

struct AgcProcessor {
    target_rms: f32,
    min_gain: f32,
    max_gain: f32,
    attack_coeff: f32,
    release_coeff: f32,
    rms_coeff: f32,
    limiter_ceiling: f32,
    peak_decay: f32,
    gain: f32,
    power: f32,
    peak: f32,
}

impl AgcProcessor {
    fn new(sample_rate_hz: u32, config: &AgcConfig) -> Self {
        let sample_rate_hz = sample_rate_hz.max(1) as f32;
        let target_rms = db_to_gain(config.target_rms_dbfs).max(EPSILON);
        let min_gain = db_to_gain(config.min_gain_db);
        let max_gain = db_to_gain(config.max_gain_db);
        let limiter_ceiling = db_to_gain(config.limiter_ceiling_dbfs).min(1.0);
        Self {
            target_rms,
            min_gain,
            max_gain,
            attack_coeff: smoothing_coeff(sample_rate_hz, config.attack_ms as f32),
            release_coeff: smoothing_coeff(sample_rate_hz, config.release_ms as f32),
            rms_coeff: smoothing_coeff(sample_rate_hz, AGC_RMS_WINDOW_MS),
            limiter_ceiling,
            peak_decay: decay_coeff(sample_rate_hz, AGC_PEAK_RELEASE_MS),
            gain: 1.0,
            power: 0.0,
            peak: 0.0,
        }
    }

    fn process(&mut self, sample: f32) -> f32 {
        let abs = sample.abs();
        self.power += self.rms_coeff * (abs.mul_add(abs, -self.power));
        let rms = self.power.sqrt().max(EPSILON);
        let mut desired_gain = self.target_rms / rms;
        desired_gain = desired_gain.clamp(self.min_gain, self.max_gain);
        let coeff = if desired_gain < self.gain {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.gain += (desired_gain - self.gain) * coeff;

        self.peak *= self.peak_decay;
        if abs > self.peak {
            self.peak = abs;
        }
        let limit_gain = if self.peak > 0.0 {
            self.limiter_ceiling / self.peak
        } else {
            1.0
        };
        let applied_gain = self.gain.min(limit_gain);
        (sample * applied_gain).clamp(-1.0, 1.0)
    }
}

struct NoiseReducer {
    threshold: f32,
    reduction_gain: f32,
    attack_coeff: f32,
    release_coeff: f32,
    rms_coeff: f32,
    gain: f32,
    power: f32,
}

impl NoiseReducer {
    fn new(sample_rate_hz: u32, config: &DenoiseConfig) -> Self {
        let sample_rate_hz = sample_rate_hz.max(1) as f32;
        let threshold = db_to_gain(config.threshold_dbfs).max(EPSILON);
        let reduction_gain = db_to_gain(-config.reduction_db).clamp(0.0, 1.0);
        Self {
            threshold,
            reduction_gain,
            attack_coeff: smoothing_coeff(sample_rate_hz, config.attack_ms as f32),
            release_coeff: smoothing_coeff(sample_rate_hz, config.release_ms as f32),
            rms_coeff: smoothing_coeff(sample_rate_hz, DENOISE_RMS_WINDOW_MS),
            gain: 1.0,
            power: 0.0,
        }
    }

    fn process(&mut self, sample: f32) -> f32 {
        let abs = sample.abs();
        self.power += self.rms_coeff * (abs.mul_add(abs, -self.power));
        let rms = self.power.sqrt();
        let target_gain = if rms < self.threshold {
            self.reduction_gain
        } else {
            1.0
        };
        let coeff = if target_gain < self.gain {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.gain += (target_gain - self.gain) * coeff;
        sample * self.gain
    }
}

struct HighPassFilter {
    alpha: f32,
    prev_input: f32,
    prev_output: f32,
}

impl HighPassFilter {
    fn new(sample_rate_hz: u32, cutoff_hz: f32) -> Option<Self> {
        let sample_rate_hz = sample_rate_hz.max(1) as f32;
        if cutoff_hz <= 0.0 {
            return None;
        }
        let dt = 1.0 / sample_rate_hz;
        let rc = 1.0 / (2.0 * PI * cutoff_hz);
        let alpha = rc / (rc + dt);
        Some(Self {
            alpha,
            prev_input: 0.0,
            prev_output: 0.0,
        })
    }

    fn process(&mut self, sample: f32) -> f32 {
        let output = self.alpha * (self.prev_output + sample - self.prev_input);
        self.prev_input = sample;
        self.prev_output = output;
        output
    }
}

pub struct RawAudioWriter {
    file: BufWriter<std::fs::File>,
    system: VecDeque<f32>,
    mic: VecDeque<f32>,
    pending_flush_samples: usize,
    last_system_at: Option<Instant>,
    last_mic_at: Option<Instant>,
    mixdown: MixdownProcessor,
}

impl RawAudioWriter {
    const FLUSH_SAMPLES: usize = 48_000;
    const MISSING_SOURCE_TIMEOUT: Duration = Duration::from_millis(500);

    pub fn new(file: std::fs::File, sample_rate_hz: u32, mixdown: MixdownConfig) -> Self {
        Self {
            file: BufWriter::new(file),
            system: VecDeque::new(),
            mic: VecDeque::new(),
            pending_flush_samples: 0,
            last_system_at: None,
            last_mic_at: None,
            mixdown: MixdownProcessor::new(sample_rate_hz, &mixdown),
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
        let processed = self.mixdown.process(sample);
        self.file.write_all(&processed.to_le_bytes())?;
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

#[cfg(test)]
mod tests {
    use super::{MixdownProcessor, db_to_gain};
    use crate::config::MixdownConfig;

    fn rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum: f32 = samples.iter().map(|sample| sample * sample).sum();
        (sum / samples.len() as f32).sqrt()
    }

    #[test]
    fn agc_can_be_disabled() {
        let mut config = MixdownConfig::default();
        config.agc.enabled = false;
        config.high_pass.enabled = false;
        let mut processor = MixdownProcessor::new(48_000, &config);
        let input = 0.25;
        let output = processor.process(input);
        assert!((output - input).abs() < 1.0e-6);
    }

    #[test]
    fn agc_targets_rms() {
        let mut config = MixdownConfig::default();
        config.high_pass.enabled = false;
        let mut processor = MixdownProcessor::new(48_000, &config);
        let target = db_to_gain(config.agc.target_rms_dbfs);
        let max_gain = db_to_gain(config.agc.max_gain_db);
        let input = target / max_gain;
        let mut samples = Vec::with_capacity(48_000);
        for _ in 0..48_000 {
            samples.push(processor.process(input));
        }
        let tail = &samples[24_000..];
        let measured = rms(tail);
        assert!((measured - target).abs() <= target * 0.2);
    }

    #[test]
    fn agc_limits_peaks_after_gain() {
        let mut config = MixdownConfig::default();
        config.high_pass.enabled = false;
        let mut processor = MixdownProcessor::new(48_000, &config);
        let target = db_to_gain(config.agc.target_rms_dbfs);
        let max_gain = db_to_gain(config.agc.max_gain_db);
        let input = target / max_gain;
        for _ in 0..48_000 {
            processor.process(input);
        }
        let peak = processor.process(1.0);
        let ceiling = db_to_gain(config.agc.limiter_ceiling_dbfs);
        assert!(peak.abs() <= ceiling + 0.01);
    }

    #[test]
    fn denoise_reduces_low_level_noise() {
        let mut config = MixdownConfig::default();
        config.agc.enabled = false;
        config.denoise.enabled = true;
        config.denoise.threshold_dbfs = -35.0;
        config.denoise.reduction_db = 12.0;
        config.high_pass.enabled = false;
        let mut processor = MixdownProcessor::new(48_000, &config);
        let input = db_to_gain(-45.0);
        let mut samples = Vec::with_capacity(48_000);
        for _ in 0..48_000 {
            samples.push(processor.process(input));
        }
        let tail = &samples[24_000..];
        let measured = rms(tail);
        assert!(measured <= input * 0.7);
    }

    #[test]
    fn denoise_preserves_signal_above_threshold() {
        let mut config = MixdownConfig::default();
        config.agc.enabled = false;
        config.denoise.enabled = true;
        config.denoise.threshold_dbfs = -35.0;
        config.high_pass.enabled = false;
        let mut processor = MixdownProcessor::new(48_000, &config);
        let input = db_to_gain(-25.0);
        let mut samples = Vec::with_capacity(48_000);
        for _ in 0..48_000 {
            samples.push(processor.process(input));
        }
        let tail = &samples[24_000..];
        let measured = rms(tail);
        assert!((measured - input).abs() <= input * 0.15);
    }

    #[test]
    fn high_pass_reduces_dc() {
        let mut config = MixdownConfig::default();
        config.agc.enabled = false;
        config.denoise.enabled = false;
        config.high_pass.enabled = true;
        config.high_pass.cutoff_hz = 100.0;
        let mut processor = MixdownProcessor::new(48_000, &config);
        let mut samples = Vec::with_capacity(48_000);
        for _ in 0..48_000 {
            samples.push(processor.process(0.2));
        }
        let tail = &samples[24_000..];
        let max = tail
            .iter()
            .fold(0.0_f32, |acc, sample| acc.max(sample.abs()));
        assert!(max < 0.01);
    }
}
