mod config;
mod config_cmd;
mod init;
mod session;
mod tui;

use clap::{Parser, Subcommand};
use config::{Config, ConfigPaths};
use koe_core::asr::{AsrProvider, create_asr};
use koe_core::capture::create_capture;
use koe_core::process::ChunkRecvTimeoutError;
use koe_core::types::{AudioFrame, AudioSource, CaptureStats, NotesPatch};
use session::SessionHandle;
use std::collections::VecDeque;
use std::io::{BufWriter, Write};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};
use tui::{AsrCommand, UiEvent};

#[derive(Parser)]
#[command(name = "koe", version, about = "meeting transcription engine")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    #[command(flatten)]
    run: RunArgs,
}

#[derive(Subcommand)]
enum Command {
    Init(init::InitArgs),
    Config(config_cmd::ConfigArgs),
}

#[derive(Parser, Debug, Clone)]
struct RunArgs {
    /// ASR provider: whisper or groq
    #[arg(long)]
    asr: Option<String>,

    /// Transcriber model. whisper: path to GGML file, groq: model name [default: whisper-large-v3-turbo]
    #[arg(long)]
    model_trn: Option<String>,

    /// Summarizer provider: ollama or openrouter
    #[arg(long)]
    summarizer: Option<String>,

    /// Summarizer model. ollama: model tag [default: qwen3:30b-a3b], openrouter: model id [default: google/gemini-2.5-flash]
    #[arg(long)]
    model_sum: Option<String>,

    /// Meeting context to pass to summarizer and session metadata
    #[arg(long)]
    context: Option<String>,

    /// Preferred participant names (comma-separated)
    #[arg(long, value_delimiter = ',', value_name = "name,...")]
    participants: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
struct ResolvedRunArgs {
    asr: String,
    model_trn: Option<String>,
    summarizer: String,
    model_sum: Option<String>,
    context: Option<String>,
    participants: Vec<String>,
}

impl RunArgs {
    fn resolve(self, config: &Config) -> ResolvedRunArgs {
        let asr = self.asr.unwrap_or_else(|| config.asr.provider.clone());
        let summarizer = self
            .summarizer
            .unwrap_or_else(|| config.summarizer.provider.clone());

        let config_model_trn = if config.asr.model.trim().is_empty() {
            None
        } else {
            Some(config.asr.model.clone())
        };
        let config_model_sum = if config.summarizer.model.trim().is_empty() {
            None
        } else {
            Some(config.summarizer.model.clone())
        };

        let model_trn = self.model_trn.or(config_model_trn);
        let model_sum = self.model_sum.or(config_model_sum);
        let context = self.context.or_else(|| {
            let value = config.session.context.clone();
            if value.is_empty() { None } else { Some(value) }
        });
        let participants = self
            .participants
            .unwrap_or_else(|| config.session.participants.clone())
            .into_iter()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .collect();

        ResolvedRunArgs {
            asr,
            model_trn,
            summarizer,
            model_sum,
            context,
            participants,
        }
    }
}

fn main() {
    dotenvy::dotenv().ok();
    let cli = Cli::parse();

    let paths = match ConfigPaths::from_home() {
        Ok(paths) => paths,
        Err(err) => {
            eprintln!("config paths error: {err}");
            std::process::exit(1);
        }
    };

    let config = match Config::load_or_create(&paths) {
        Ok(config) => config,
        Err(err) => {
            eprintln!("config load failed: {err}");
            std::process::exit(1);
        }
    };

    if let Some(command) = cli.command {
        match command {
            Command::Init(args) => {
                if let Err(e) = init::run(&args, &paths) {
                    eprintln!("init failed: {e}");
                    std::process::exit(1);
                }
                return;
            }
            Command::Config(args) => {
                if let Err(e) = config_cmd::run(&args, &paths) {
                    eprintln!("config failed: {e}");
                    std::process::exit(1);
                }
                return;
            }
        }
    }

    let run = cli.run.resolve(&config);
    apply_config_env(&config, &run);
    let stats = CaptureStats::new();
    let stats_display = stats.clone();
    let models_dir = paths.models_dir.clone();

    let whisper_model_env = std::env::var("KOE_WHISPER_MODEL").ok();
    let groq_model_env = std::env::var("KOE_GROQ_MODEL").ok();
    let mut whisper_model = match run.asr.as_str() {
        "whisper" => run.model_trn.clone().or(whisper_model_env),
        _ => whisper_model_env,
    };
    let groq_model = match run.asr.as_str() {
        "groq" => run.model_trn.clone().or(groq_model_env),
        _ => groq_model_env,
    };

    if run.asr == "whisper" {
        let hint = run.model_trn.as_deref();
        if let Err(e) = ensure_whisper_model(&mut whisper_model, hint, &models_dir) {
            eprintln!("init failed: {e}");
            std::process::exit(1);
        }
    }

    let mut asr = match create_asr(
        run.asr.as_str(),
        model_for_provider(run.asr.as_str(), &whisper_model, groq_model.as_deref()),
    ) {
        Ok(provider) => provider,
        Err(e) => {
            eprintln!("asr init failed: {e}");
            std::process::exit(1);
        }
    };

    let session = match create_session(&paths, &run, &config, &whisper_model, &groq_model) {
        Ok(session) => session,
        Err(err) => {
            eprintln!("session init failed: {err}");
            std::process::exit(1);
        }
    };
    let audio_raw = match session.open_audio_raw() {
        Ok(handle) => handle,
        Err(err) => {
            eprintln!("session audio init failed: {err}");
            std::process::exit(1);
        }
    };
    let mut raw_writer = RawAudioWriter::new(audio_raw);

    let capture = match create_capture(stats.clone()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("capture init failed: {e}");
            std::process::exit(1);
        }
    };

    let raw_sink: Option<koe_core::process::RawAudioSink> = Some(Box::new(move |source, frame| {
        if let Err(err) = raw_writer.write_frame(source, frame) {
            eprintln!("audio write failed: {err}");
        }
    }));

    let (processor, chunk_rx) =
        match koe_core::process::AudioProcessor::start(capture, stats, raw_sink) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("processor start failed: {e}");
                std::process::exit(1);
            }
        };

    let asr_name = asr.name().to_string();
    let (ui_tx, ui_rx) = mpsc::channel();
    let _ = ui_tx.send(UiEvent::NotesPatch(NotesPatch { ops: Vec::new() }));
    let (asr_cmd_tx, asr_cmd_rx) = mpsc::channel();

    let asr_thread = match thread::Builder::new()
        .name("koe-asr".into())
        .spawn(move || {
            let mut current_provider = run.asr.clone();
            let mut whisper_model = whisper_model;
            let groq_model = groq_model;
            let models_dir = models_dir.clone();

            let send_status = |name: String, connected: bool| {
                let _ = ui_tx.send(UiEvent::AsrStatus { name, connected });
            };

            send_status(asr.name().to_string(), true);
            let mut latency_ms: Option<u128> = None;

            loop {
                while let Ok(cmd) = asr_cmd_rx.try_recv() {
                    match cmd {
                        AsrCommand::ToggleProvider => {
                            let next = if current_provider == "whisper" {
                                "groq"
                            } else {
                                "whisper"
                            };

                            let next_model = match next {
                                "whisper" => {
                                    if let Err(e) =
                                        ensure_whisper_model(&mut whisper_model, None, &models_dir)
                                    {
                                        eprintln!("init failed: {e}");
                                        continue;
                                    }
                                    whisper_model.as_deref()
                                }
                                "groq" => groq_model.as_deref(),
                                _ => None,
                            };

                            match create_asr(next, next_model) {
                                Ok(provider) => {
                                    asr = provider;
                                    current_provider = next.to_string();
                                    send_status(asr.name().to_string(), true);
                                }
                                Err(e) => {
                                    eprintln!("asr switch failed: {e}");
                                    send_status(current_provider.clone(), false);
                                }
                            }
                        }
                    }
                }

                let chunk = match chunk_rx.recv_timeout(Duration::from_millis(50)) {
                    Ok(chunk) => chunk,
                    Err(ChunkRecvTimeoutError::Timeout) => continue,
                    Err(ChunkRecvTimeoutError::Disconnected) => break,
                };

                let (mut segments, elapsed) = match transcribe_with_latency(asr.as_mut(), &chunk) {
                    Ok(result) => result,
                    Err(e) => {
                        eprintln!("asr error: {e}");
                        continue;
                    }
                };

                let smoothed = match latency_ms {
                    Some(prev) => (prev * 9 + elapsed) / 10,
                    None => elapsed,
                };
                latency_ms = Some(smoothed);
                let _ = ui_tx.send(UiEvent::AsrLag { last_ms: smoothed });

                if segments.is_empty() {
                    continue;
                }

                if let Some(speaker) = default_speaker(chunk.source) {
                    for seg in &mut segments {
                        if seg.speaker.is_none() {
                            seg.speaker = Some(speaker.to_string());
                        }
                    }
                }

                if ui_tx.send(UiEvent::Transcript(segments)).is_err() {
                    break;
                }
            }
        }) {
        Ok(handle) => Some(handle),
        Err(e) => {
            eprintln!("asr thread spawn failed: {e}");
            std::process::exit(1);
        }
    };

    if let Err(e) = tui::run(
        processor,
        ui_rx,
        stats_display,
        asr_name,
        asr_cmd_tx,
        config.ui.clone(),
        session,
    ) {
        eprintln!("tui error: {e}");
        std::process::exit(1);
    }

    if let Some(handle) = asr_thread {
        let _ = handle.join();
    }
}

fn default_speaker(source: AudioSource) -> Option<&'static str> {
    match source {
        AudioSource::Microphone => Some("Me"),
        AudioSource::System => Some("Them"),
        AudioSource::Mixed => Some("Unknown"),
    }
}

fn ensure_whisper_model(
    model: &mut Option<String>,
    hint: Option<&str>,
    models_dir: &std::path::Path,
) -> Result<(), String> {
    let candidate = hint
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .map(|value| value.to_string())
        .or_else(|| model.clone());

    if let Some(value) = candidate.as_deref() {
        let is_path = value.ends_with(".bin")
            || value.contains(std::path::MAIN_SEPARATOR)
            || value.contains('/');
        if is_path {
            let path = std::path::Path::new(value);
            if path.exists() {
                *model = Some(value.to_string());
                return Ok(());
            }
            return Err(format!("whisper model not found at {}", path.display()));
        }
    }

    let model_name = candidate.unwrap_or_else(|| init::DEFAULT_WHISPER_MODEL.to_string());
    let path = init::download_model(&model_name, models_dir, false).map_err(|e| e.to_string())?;
    *model = Some(path.to_string_lossy().to_string());
    Ok(())
}

fn model_for_provider<'a>(
    provider: &str,
    whisper_model: &'a Option<String>,
    groq_model: Option<&'a str>,
) -> Option<&'a str> {
    match provider {
        "whisper" => whisper_model.as_deref(),
        "groq" => groq_model,
        _ => None,
    }
}

struct RawAudioWriter {
    file: BufWriter<std::fs::File>,
    system: VecDeque<f32>,
    mic: VecDeque<f32>,
    pending_flush_samples: usize,
}

impl RawAudioWriter {
    const FLUSH_SAMPLES: usize = 48_000;

    fn new(file: std::fs::File) -> Self {
        Self {
            file: BufWriter::new(file),
            system: VecDeque::new(),
            mic: VecDeque::new(),
            pending_flush_samples: 0,
        }
    }

    fn write_frame(&mut self, source: AudioSource, frame: &AudioFrame) -> std::io::Result<()> {
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

fn transcribe_with_latency(
    asr: &mut dyn AsrProvider,
    chunk: &koe_core::types::AudioChunk,
) -> Result<(Vec<koe_core::types::TranscriptSegment>, u128), koe_core::AsrError> {
    let start = Instant::now();
    let segments = asr.transcribe(chunk)?;
    Ok((segments, start.elapsed().as_millis()))
}

fn apply_config_env(config: &Config, run: &ResolvedRunArgs) {
    if run.asr == "groq"
        && !config.asr.api_key.trim().is_empty()
        && std::env::var("GROQ_API_KEY").is_err()
    {
        unsafe {
            std::env::set_var("GROQ_API_KEY", config.asr.api_key.trim());
        }
    }
    if run.summarizer == "openrouter"
        && !config.summarizer.api_key.trim().is_empty()
        && std::env::var("OPENROUTER_API_KEY").is_err()
    {
        unsafe {
            std::env::set_var("OPENROUTER_API_KEY", config.summarizer.api_key.trim());
        }
    }
    let _ = &run.model_sum;
}

fn create_session(
    paths: &ConfigPaths,
    run: &ResolvedRunArgs,
    config: &Config,
    whisper_model: &Option<String>,
    groq_model: &Option<String>,
) -> Result<SessionHandle, session::SessionError> {
    let asr_model = match run.asr.as_str() {
        "whisper" => whisper_model.clone().unwrap_or_default(),
        "groq" => groq_model.clone().unwrap_or_default(),
        _ => String::new(),
    };
    let summarizer_model = run.model_sum.clone().unwrap_or_default();
    let metadata = session::SessionMetadata::new(session::SessionMetadataInput {
        context: run.context.clone(),
        participants: run.participants.clone(),
        audio_sample_rate_hz: config.audio.sample_rate,
        audio_channels: config.audio.channels,
        audio_sources: config.audio.sources.clone(),
        asr_provider: run.asr.clone(),
        asr_model,
        summarizer_provider: run.summarizer.clone(),
        summarizer_model,
    })?;
    session::SessionHandle::start(paths, metadata)
}

#[cfg(test)]
mod tests {
    use super::{default_speaker, transcribe_with_latency};
    use koe_core::asr::AsrProvider;
    use koe_core::types::{AudioChunk, AudioSource, TranscriptSegment};
    use std::time::Duration;

    #[test]
    fn default_speaker_maps_sources() {
        assert_eq!(default_speaker(AudioSource::Microphone), Some("Me"));
        assert_eq!(default_speaker(AudioSource::System), Some("Them"));
        assert_eq!(default_speaker(AudioSource::Mixed), Some("Unknown"));
    }

    struct DummyAsr {
        delay: Duration,
    }

    impl AsrProvider for DummyAsr {
        fn name(&self) -> &'static str {
            "dummy"
        }

        fn transcribe(
            &mut self,
            _chunk: &AudioChunk,
        ) -> Result<Vec<TranscriptSegment>, koe_core::AsrError> {
            std::thread::sleep(self.delay);
            Ok(vec![TranscriptSegment {
                id: 0,
                start_ms: 0,
                end_ms: 10,
                speaker: None,
                text: "ok".to_string(),
                finalized: false,
            }])
        }
    }

    #[test]
    fn transcribe_latency_under_budget() {
        let mut asr = DummyAsr {
            delay: Duration::from_millis(20),
        };
        let chunk = AudioChunk {
            source: AudioSource::System,
            start_pts_ns: 0,
            sample_rate_hz: 16_000,
            pcm_mono_f32: vec![0.0; 160],
        };
        let (_segments, elapsed) = transcribe_with_latency(&mut asr, &chunk).unwrap();
        assert!(elapsed < 4_000);
    }
}
