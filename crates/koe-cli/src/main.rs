mod config;
mod config_cmd;
mod init;
mod raw_audio;
mod session;
mod tui;

use clap::{Parser, Subcommand};
use config::{Config, ConfigPaths, ProviderConfig};
use koe_core::asr::{AsrProvider, create_asr};
use koe_core::capture::{CaptureConfig, create_capture, list_audio_inputs};
use koe_core::process::ChunkRecvTimeoutError;
use koe_core::types::{AudioSource, CaptureStats, NotesPatch};
use raw_audio::SharedRawAudioWriter;
use session::SessionFactory;
use std::path::PathBuf;
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
    /// ASR selection: local, cloud, whisper, or groq
    #[arg(long)]
    asr: Option<String>,

    /// Transcriber model. whisper: path to GGML file, groq: model name [default: whisper-large-v3-turbo]
    #[arg(long)]
    model_trn: Option<String>,

    /// Summarizer selection: local, cloud, ollama, or openrouter
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
    asr: ResolvedProfile,
    summarizer: ResolvedProfile,
    context: Option<String>,
    participants: Vec<String>,
}

#[derive(Debug, Clone)]
struct ResolvedProfile {
    provider: String,
    model: Option<String>,
    api_key: String,
}

impl RunArgs {
    fn resolve(self, config: &Config) -> ResolvedRunArgs {
        let asr_profile = select_asr_profile(config, self.asr.as_deref());
        let summarizer_profile = select_summarizer_profile(config, self.summarizer.as_deref());

        let model_trn = self
            .model_trn
            .or_else(|| non_empty_value(asr_profile.model.as_str()));
        let model_sum = self
            .model_sum
            .or_else(|| non_empty_value(summarizer_profile.model.as_str()));
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
            asr: ResolvedProfile {
                provider: asr_profile.provider.clone(),
                model: model_trn,
                api_key: asr_profile.api_key.clone(),
            },
            summarizer: ResolvedProfile {
                provider: summarizer_profile.provider.clone(),
                model: model_sum,
                api_key: summarizer_profile.api_key.clone(),
            },
            context,
            participants,
        }
    }
}

fn select_asr_profile<'a>(config: &'a Config, selector: Option<&str>) -> &'a ProviderConfig {
    select_profile(
        config.asr.active.as_str(),
        &config.asr.local,
        &config.asr.cloud,
        selector,
    )
}

fn select_summarizer_profile<'a>(config: &'a Config, selector: Option<&str>) -> &'a ProviderConfig {
    select_profile(
        config.summarizer.active.as_str(),
        &config.summarizer.local,
        &config.summarizer.cloud,
        selector,
    )
}

fn select_profile<'a>(
    active: &str,
    local: &'a ProviderConfig,
    cloud: &'a ProviderConfig,
    selector: Option<&str>,
) -> &'a ProviderConfig {
    let fallback = if active == "cloud" { cloud } else { local };
    match selector {
        Some("local") => local,
        Some("cloud") => cloud,
        Some(provider) if local.provider == provider => local,
        Some(provider) if cloud.provider == provider => cloud,
        _ => fallback,
    }
}

fn non_empty_value(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
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
    apply_config_env(&run);
    let stats = CaptureStats::new();
    let stats_display = stats.clone();
    let models_dir = paths.models_dir.clone();

    let whisper_model_env = std::env::var("KOE_WHISPER_MODEL").ok();
    let groq_model_env = std::env::var("KOE_GROQ_MODEL").ok();
    let mut whisper_model = match run.asr.provider.as_str() {
        "whisper" => run.asr.model.clone().or(whisper_model_env),
        _ => whisper_model_env,
    };
    let groq_model = match run.asr.provider.as_str() {
        "groq" => run.asr.model.clone().or(groq_model_env),
        _ => groq_model_env,
    };

    if run.asr.provider == "whisper" {
        let hint = run.asr.model.as_deref();
        if let Err(e) = ensure_whisper_model(&mut whisper_model, hint, &models_dir) {
            eprintln!("init failed: {e}");
            std::process::exit(1);
        }
    }

    let asr_models = tui::AsrModels::new(whisper_model.clone(), groq_model.clone());

    let mut asr = match create_asr(
        run.asr.provider.as_str(),
        model_for_provider(
            run.asr.provider.as_str(),
            &whisper_model,
            groq_model.as_deref(),
        ),
    ) {
        Ok(provider) => provider,
        Err(e) => {
            eprintln!("asr init failed: {e}");
            std::process::exit(1);
        }
    };

    let shared_writer = SharedRawAudioWriter::new(None);

    let capture_config =
        capture_config_from_sources(&config.audio.sources, &config.audio.microphone_device_id);
    let capture = match create_capture(stats.clone(), capture_config) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("capture init failed: {e}");
            std::process::exit(1);
        }
    };

    let raw_sink: Option<koe_core::process::RawAudioSink> = {
        let shared_writer = shared_writer.clone();
        Some(Box::new(move |source, frame| {
            if let Err(err) = shared_writer.write_frame(source, frame) {
                eprintln!("audio write failed: {err}");
            }
        }))
    };

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
            let mut current_provider = run.asr.provider.clone();
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

    let export_dir = export_dir_from_config(&paths, &config.session.export_dir);
    let session_factory = SessionFactory::new(
        paths.clone(),
        export_dir,
        config.audio.sample_rate,
        config.audio.channels,
        config.audio.sources.clone(),
    );
    let ctx = tui::TuiContext {
        processor,
        ui_rx,
        stats: stats_display,
        asr_name,
        asr_cmd_tx,
        ui_config: config.ui.clone(),
        session_factory,
        shared_writer,
        initial_context: run.context.clone().unwrap_or_default(),
        participants: run.participants.clone(),
        summarizer_provider: run.summarizer.provider.clone(),
        summarizer_model: run.summarizer.model.clone().unwrap_or_default(),
        asr_models,
    };

    if let Err(e) = tui::run(ctx) {
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

fn export_dir_from_config(paths: &ConfigPaths, value: &str) -> Option<PathBuf> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    let path = PathBuf::from(trimmed);
    if path.is_absolute() {
        Some(path)
    } else {
        Some(paths.base_dir.join(path))
    }
}

fn capture_config_from_sources(sources: &[String], mic_device_id: &str) -> CaptureConfig {
    let capture_system = sources
        .iter()
        .any(|source| matches!(source.as_str(), "system" | "mixed"));
    let capture_microphone = sources
        .iter()
        .any(|source| matches!(source.as_str(), "microphone" | "mixed"));
    let microphone_device_id = resolve_microphone_device_id(capture_microphone, mic_device_id);
    CaptureConfig {
        capture_system,
        capture_microphone,
        microphone_device_id,
    }
}

fn resolve_microphone_device_id(capture_microphone: bool, mic_device_id: &str) -> Option<String> {
    if !capture_microphone {
        return None;
    }
    let trimmed = mic_device_id.trim();
    if !trimmed.is_empty() {
        return Some(trimmed.to_string());
    }
    let inputs = list_audio_inputs();
    select_default_microphone(&inputs)
}

fn select_default_microphone(inputs: &[koe_core::capture::AudioInputDeviceInfo]) -> Option<String> {
    if inputs.is_empty() {
        return None;
    }
    let built_in = inputs.iter().find(|device| {
        let name = device.name.to_lowercase();
        device.id == "BuiltInMicrophoneDevice"
            || name.contains("built-in")
            || name.contains("built in")
            || name.contains("macbook")
    });
    if let Some(device) = built_in {
        return Some(device.id.clone());
    }
    inputs
        .iter()
        .find(|device| device.is_default)
        .map(|device| device.id.clone())
}

fn transcribe_with_latency(
    asr: &mut dyn AsrProvider,
    chunk: &koe_core::types::AudioChunk,
) -> Result<(Vec<koe_core::types::TranscriptSegment>, u128), koe_core::AsrError> {
    let start = Instant::now();
    let segments = asr.transcribe(chunk)?;
    Ok((segments, start.elapsed().as_millis()))
}

fn apply_config_env(run: &ResolvedRunArgs) {
    if run.asr.provider == "groq"
        && !run.asr.api_key.trim().is_empty()
        && std::env::var("GROQ_API_KEY").is_err()
    {
        unsafe {
            std::env::set_var("GROQ_API_KEY", run.asr.api_key.trim());
        }
    }
    if run.summarizer.provider == "openrouter"
        && !run.summarizer.api_key.trim().is_empty()
        && std::env::var("OPENROUTER_API_KEY").is_err()
    {
        unsafe {
            std::env::set_var("OPENROUTER_API_KEY", run.summarizer.api_key.trim());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{default_speaker, select_default_microphone, transcribe_with_latency};
    use koe_core::asr::AsrProvider;
    use koe_core::capture::AudioInputDeviceInfo;
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

    #[test]
    fn select_default_microphone_prefers_built_in() {
        let inputs = vec![
            AudioInputDeviceInfo {
                id: "BT-MIC".to_string(),
                name: "WH-1000XM5".to_string(),
                is_default: true,
            },
            AudioInputDeviceInfo {
                id: "BuiltInMicrophoneDevice".to_string(),
                name: "MacBook Pro Microphone".to_string(),
                is_default: false,
            },
        ];
        assert_eq!(
            select_default_microphone(&inputs).as_deref(),
            Some("BuiltInMicrophoneDevice")
        );
    }

    #[test]
    fn select_default_microphone_falls_back_to_default() {
        let inputs = vec![AudioInputDeviceInfo {
            id: "USB-MIC".to_string(),
            name: "USB Microphone".to_string(),
            is_default: true,
        }];
        assert_eq!(
            select_default_microphone(&inputs).as_deref(),
            Some("USB-MIC")
        );
    }

    #[test]
    fn select_default_microphone_handles_empty_list() {
        let inputs = Vec::new();
        assert_eq!(select_default_microphone(&inputs), None);
    }
}
