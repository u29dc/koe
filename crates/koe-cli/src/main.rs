mod config;
mod config_cmd;
mod init;
mod raw_audio;
mod session;
mod tui;

use clap::{Parser, Subcommand};
use config::{Config, ConfigPaths, ProviderConfig};
use koe_core::capture::{CaptureConfig, create_capture, list_audio_inputs};
use koe_core::process::ChunkRecvTimeoutError;
use koe_core::summarize::create_summarize_provider;
use koe_core::summarize::filter::{build_participant_tokens, normalize_text, should_keep_segment};
use koe_core::transcribe::{TranscribeProvider, create_transcribe_provider};
use koe_core::transcript::TranscriptLedger;
use koe_core::types::{
    AudioSource, CaptureStats, MeetingNotes, NoteBullet, NotesOp, NotesPatch, SummarizeEvent,
};
use raw_audio::{RawAudioMessage, SharedRawAudioWriter, spawn_raw_audio_writer};
use session::SessionFactory;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};
use tui::{SummarizeCommand, TranscribeCommand, UiEvent};

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
    /// Transcribe mode: local or cloud
    #[arg(long)]
    transcribe: Option<String>,

    /// Transcribe model override for the selected mode
    #[arg(long, value_name = "model")]
    transcribe_model: Option<String>,

    /// Summarize mode: local or cloud
    #[arg(long)]
    summarize: Option<String>,

    /// Summarize model override for the selected mode
    #[arg(long, value_name = "model")]
    summarize_model: Option<String>,

    /// Meeting context to pass to summarize and session metadata
    #[arg(long)]
    context: Option<String>,

    /// Preferred participant names (comma-separated)
    #[arg(long, value_delimiter = ',', value_name = "name,...")]
    participants: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
struct ResolvedRunArgs {
    transcribe_profiles: RuntimeProfiles,
    summarize_profiles: RuntimeProfiles,
    context: Option<String>,
    participants: Vec<String>,
}

#[derive(Debug, Clone)]
struct RuntimeProfiles {
    active: String,
    local: ProviderConfig,
    cloud: ProviderConfig,
}

const RAW_AUDIO_QUEUE_CAP: usize = 16;

impl RuntimeProfiles {
    fn from_config(active: &str, local: &ProviderConfig, cloud: &ProviderConfig) -> Self {
        Self {
            active: active.to_string(),
            local: local.clone(),
            cloud: cloud.clone(),
        }
    }

    fn active_profile(&self) -> &ProviderConfig {
        self.profile_for_mode(self.active.as_str())
    }

    fn active_profile_mut(&mut self) -> &mut ProviderConfig {
        let is_cloud = self.active == "cloud";
        if is_cloud {
            &mut self.cloud
        } else {
            &mut self.local
        }
    }

    fn profile_for_mode(&self, mode: &str) -> &ProviderConfig {
        if mode == "cloud" {
            &self.cloud
        } else {
            &self.local
        }
    }
}

impl RunArgs {
    fn resolve(self, config: &Config) -> Result<ResolvedRunArgs, String> {
        let mut transcribe_profiles = RuntimeProfiles::from_config(
            config.transcribe.active.as_str(),
            &config.transcribe.local,
            &config.transcribe.cloud,
        );
        let mut summarize_profiles = RuntimeProfiles::from_config(
            config.summarize.active.as_str(),
            &config.summarize.local,
            &config.summarize.cloud,
        );

        apply_env_overrides(&mut transcribe_profiles, &mut summarize_profiles);

        let transcribe_mode = select_mode(
            transcribe_profiles.active.as_str(),
            self.transcribe.as_deref(),
            "transcribe",
        )?;
        let summarize_mode = select_mode(
            summarize_profiles.active.as_str(),
            self.summarize.as_deref(),
            "summarize",
        )?;
        transcribe_profiles.active = transcribe_mode;
        summarize_profiles.active = summarize_mode;

        if let Some(model) = self.transcribe_model {
            transcribe_profiles.active_profile_mut().model = model;
        }
        if let Some(model) = self.summarize_model {
            summarize_profiles.active_profile_mut().model = model;
        }

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

        Ok(ResolvedRunArgs {
            transcribe_profiles,
            summarize_profiles,
            context,
            participants,
        })
    }
}

fn select_mode(active: &str, selector: Option<&str>, label: &str) -> Result<String, String> {
    match selector {
        None => Ok(if active == "cloud" {
            "cloud".to_string()
        } else {
            "local".to_string()
        }),
        Some("local") => Ok("local".to_string()),
        Some("cloud") => Ok("cloud".to_string()),
        Some(other) => Err(format!("{label} must be local or cloud (got {other})")),
    }
}

fn env_override(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn apply_env_overrides(transcribe: &mut RuntimeProfiles, summarize: &mut RuntimeProfiles) {
    if transcribe.local.model.trim().is_empty()
        && let Some(value) = env_override("KOE_TRANSCRIBE_LOCAL_MODEL")
    {
        transcribe.local.model = value;
    }
    if transcribe.cloud.model.trim().is_empty()
        && let Some(value) = env_override("KOE_TRANSCRIBE_CLOUD_MODEL")
    {
        transcribe.cloud.model = value;
    }
    if transcribe.cloud.api_key.trim().is_empty()
        && let Some(value) = env_override("KOE_TRANSCRIBE_CLOUD_API_KEY")
    {
        transcribe.cloud.api_key = value;
    }
    if summarize.local.model.trim().is_empty()
        && let Some(value) = env_override("KOE_SUMMARIZE_LOCAL_MODEL")
    {
        summarize.local.model = value;
    }
    if summarize.cloud.model.trim().is_empty()
        && let Some(value) = env_override("KOE_SUMMARIZE_CLOUD_MODEL")
    {
        summarize.cloud.model = value;
    }
    if summarize.cloud.api_key.trim().is_empty()
        && let Some(value) = env_override("KOE_SUMMARIZE_CLOUD_API_KEY")
    {
        summarize.cloud.api_key = value;
    }
    if transcribe.cloud.api_key.trim().is_empty()
        && let Some(value) = env_override("GROQ_API_KEY")
    {
        transcribe.cloud.api_key = value;
    }
    if summarize.cloud.api_key.trim().is_empty()
        && let Some(value) = env_override("OPENROUTER_API_KEY")
    {
        summarize.cloud.api_key = value;
    }
}

fn non_empty_str(value: &str) -> Option<&str> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

fn looks_like_path(value: &str) -> bool {
    value.ends_with(".bin") || value.contains('/') || value.contains(std::path::MAIN_SEPARATOR)
}

fn to_ui_profiles(runtime: &RuntimeProfiles) -> tui::ModeProfiles {
    tui::ModeProfiles {
        active: runtime.active.clone(),
        local: tui::ProfileSummary {
            provider: runtime.local.provider.clone(),
            model: runtime.local.model.clone(),
        },
        cloud: tui::ProfileSummary {
            provider: runtime.cloud.provider.clone(),
            model: runtime.cloud.model.clone(),
        },
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

    let mut run = match cli.run.resolve(&config) {
        Ok(run) => run,
        Err(err) => {
            eprintln!("run args error: {err}");
            std::process::exit(1);
        }
    };
    let stats = CaptureStats::new();
    let stats_display = stats.clone();
    let models_dir = paths.models_dir.clone();

    if run.transcribe_profiles.active_profile().provider == "whisper" {
        let profile = run.transcribe_profiles.active_profile_mut();
        if let Err(e) = ensure_whisper_model(&mut profile.model, &models_dir) {
            eprintln!("init failed: {e}");
            std::process::exit(1);
        }
    }

    let transcribe_profiles_ui = to_ui_profiles(&run.transcribe_profiles);
    let summarize_profiles_ui = to_ui_profiles(&run.summarize_profiles);

    let active_transcribe = run.transcribe_profiles.active_profile();
    let mut transcribe = match create_transcribe_provider(
        active_transcribe.provider.as_str(),
        Some(active_transcribe.model.as_str()),
        non_empty_str(active_transcribe.api_key.as_str()),
    ) {
        Ok(provider) => provider,
        Err(e) => {
            eprintln!("transcribe init failed: {e}");
            std::process::exit(1);
        }
    };

    let shared_writer = SharedRawAudioWriter::new(None);
    let (raw_tx, raw_rx) = mpsc::sync_channel(RAW_AUDIO_QUEUE_CAP);
    let raw_writer_handle = match spawn_raw_audio_writer(raw_rx, shared_writer.clone()) {
        Ok(handle) => Some(handle),
        Err(err) => {
            eprintln!("raw audio writer spawn failed: {err}");
            None
        }
    };

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
        let raw_tx = raw_tx.clone();
        let stats_raw = stats.clone();
        Some(Box::new(move |source, frame| {
            let message = RawAudioMessage {
                source,
                samples: frame.samples_f32.clone(),
            };
            if raw_tx.try_send(message).is_err() {
                stats_raw.inc_raw_frames_dropped();
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

    let (ui_tx, ui_rx) = mpsc::channel();
    let _ = ui_tx.send(UiEvent::NotesPatch(NotesPatch { ops: Vec::new() }));
    let (transcribe_cmd_tx, transcribe_cmd_rx) = mpsc::channel();
    let (summarize_cmd_tx, summarize_cmd_rx) = mpsc::channel();
    let (summarize_tx_transcribe, summarize_rx) = mpsc::sync_channel(1);
    let ui_tx_summarize = ui_tx.clone();
    let ui_tx_transcribe = ui_tx.clone();
    let transcribe_profiles_runtime = run.transcribe_profiles.clone();
    let summarize_profiles_runtime = run.summarize_profiles.clone();
    let summarize_context = run.context.clone().unwrap_or_default();
    let summarize_participants = run.participants.clone();

    let summarize_thread =
        match thread::Builder::new()
            .name("koe-summarize".into())
            .spawn(move || {
                const SUMMARIZE_INTERVAL: Duration = Duration::from_secs(4);
                const STABLE_WINDOW_MS: i64 = 4_000;
                const TAIL_WINDOW_MS: i64 = 15_000;
                const MAX_NOTES_FOR_PROMPT: usize = 50;
                const MIN_NEW_WORDS: usize = 4;

                let current_mode = summarize_profiles_runtime.active.clone();
                let mut context = summarize_context;
                let participants = summarize_participants;
                let participant_tokens = build_participant_tokens(&participants);
                let mut ledger = TranscriptLedger::new();
                let mut meeting_notes = MeetingNotes::default();
                let mut last_summary_at = Instant::now() - SUMMARIZE_INTERVAL;
                let mut last_summarized_id: u64 = 0;

                let send_status = |mode: String, provider: String| {
                    let _ = ui_tx_summarize.send(UiEvent::SummarizeStatus { mode, provider });
                };

                let mut summarize =
                    match create_summarize_for_mode(&summarize_profiles_runtime, &current_mode) {
                        Ok(provider) => {
                            let profile = summarize_profiles_runtime.active_profile();
                            send_status(current_mode.clone(), profile.provider.clone());
                            Some(provider)
                        }
                        Err(e) => {
                            eprintln!("summarize init failed: {e}");
                            let profile = summarize_profiles_runtime.active_profile();
                            send_status(current_mode.clone(), profile.provider.clone());
                            None
                        }
                    };

                loop {
                    while let Ok(cmd) = summarize_cmd_rx.try_recv() {
                        match cmd {
                            SummarizeCommand::Reset => {
                                ledger = TranscriptLedger::new();
                                meeting_notes = MeetingNotes::default();
                                last_summarized_id = 0;
                                last_summary_at = Instant::now() - SUMMARIZE_INTERVAL;
                            }
                            SummarizeCommand::UpdateContext(value) => {
                                context = value;
                            }
                        }
                    }

                    match summarize_rx.recv_timeout(Duration::from_millis(200)) {
                        Ok(segments) => {
                            ledger.append(segments);
                        }
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    }

                    let due = Instant::now().duration_since(last_summary_at) >= SUMMARIZE_INTERVAL;

                    if !due {
                        continue;
                    }

                    let highest_end_ms = ledger.highest_end_ms();
                    let stable_cutoff = highest_end_ms - STABLE_WINDOW_MS;
                    let tail_cutoff = highest_end_ms - TAIL_WINDOW_MS;
                    let mut new_word_count = 0usize;
                    let mut max_new_id = last_summarized_id;
                    let mut has_new_segment = false;
                    let mut segments_for_prompt = Vec::new();

                    for seg in ledger.segments() {
                        if seg.end_ms > stable_cutoff {
                            continue;
                        }
                        let is_new = seg.id > last_summarized_id;
                        let in_tail = seg.end_ms >= tail_cutoff;
                        if !(is_new || in_tail) {
                            continue;
                        }
                        if !should_keep_segment(&seg.text, &participant_tokens) {
                            continue;
                        }
                        if is_new {
                            has_new_segment = true;
                            max_new_id = max_new_id.max(seg.id);
                            new_word_count += seg.text.split_whitespace().count();
                        }
                        segments_for_prompt.push(seg.clone());
                    }

                    if !has_new_segment || new_word_count < MIN_NEW_WORDS {
                        continue;
                    }

                    if segments_for_prompt.is_empty() {
                        continue;
                    }

                    let Some(provider) = summarize.as_mut() else {
                        last_summary_at = Instant::now();
                        continue;
                    };

                    let mut patch_ready: Option<NotesPatch> = None;
                    let context_ref = if context.trim().is_empty() {
                        None
                    } else {
                        Some(context.as_str())
                    };
                    let notes_for_prompt = if meeting_notes.bullets.len() > MAX_NOTES_FOR_PROMPT {
                        let start = meeting_notes.bullets.len() - MAX_NOTES_FOR_PROMPT;
                        MeetingNotes {
                            bullets: meeting_notes.bullets[start..].to_vec(),
                        }
                    } else {
                        meeting_notes.clone()
                    };

                    let result = provider.summarize(
                        &segments_for_prompt,
                        &notes_for_prompt,
                        context_ref,
                        &participants,
                        &mut |event| match event {
                            SummarizeEvent::DraftToken(_) => {}
                            SummarizeEvent::PatchReady(patch) => {
                                patch_ready = Some(patch);
                            }
                        },
                    );

                    match result {
                        Ok(()) => {
                            last_summary_at = Instant::now();
                            if let Some(patch) = patch_ready {
                                apply_notes_patch_state(&mut meeting_notes, patch.clone());
                                let _ = ui_tx_summarize.send(UiEvent::NotesPatch(patch));
                            }
                            last_summarized_id = max_new_id;
                        }
                        Err(e) => {
                            eprintln!("summarize error: {e}");
                            last_summary_at = Instant::now();
                        }
                    }
                }
            }) {
            Ok(handle) => Some(handle),
            Err(e) => {
                eprintln!("summarize thread spawn failed: {e}");
                None
            }
        };

    let transcribe_thread =
        match thread::Builder::new()
            .name("koe-transcribe".into())
            .spawn(move || {
                let current_mode = transcribe_profiles_runtime.active.clone();

                let send_status = |mode: String, provider: String, connected: bool| {
                    let _ = ui_tx_transcribe.send(UiEvent::TranscribeStatus {
                        mode,
                        provider,
                        connected,
                    });
                };

                let active_profile = transcribe_profiles_runtime.active_profile();
                send_status(current_mode.clone(), active_profile.provider.clone(), true);
                let mut latency_ms: Option<u128> = None;
                let mut drain_ack: Option<mpsc::Sender<()>> = None;

                loop {
                    while let Ok(cmd) = transcribe_cmd_rx.try_recv() {
                        match cmd {
                            TranscribeCommand::Drain(ack) => {
                                drain_ack = Some(ack);
                            }
                        }
                    }

                    let chunk = match chunk_rx.recv_timeout(Duration::from_millis(50)) {
                        Ok(chunk) => chunk,
                        Err(ChunkRecvTimeoutError::Timeout) => {
                            if let Some(ack) = drain_ack.take() {
                                let _ = ack.send(());
                            }
                            continue;
                        }
                        Err(ChunkRecvTimeoutError::Disconnected) => {
                            if let Some(ack) = drain_ack.take() {
                                let _ = ack.send(());
                            }
                            break;
                        }
                    };

                    let (mut segments, elapsed) =
                        match transcribe_with_latency(transcribe.as_mut(), &chunk) {
                            Ok(result) => result,
                            Err(e) => {
                                eprintln!("transcribe error: {e}");
                                continue;
                            }
                        };

                    let smoothed = match latency_ms {
                        Some(prev) => (prev * 9 + elapsed) / 10,
                        None => elapsed,
                    };
                    latency_ms = Some(smoothed);
                    let _ = ui_tx_transcribe.send(UiEvent::TranscribeLag { last_ms: smoothed });

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

                    let _ = summarize_tx_transcribe.try_send(segments.clone());

                    if ui_tx_transcribe
                        .send(UiEvent::Transcript(segments))
                        .is_err()
                    {
                        break;
                    }
                }
            }) {
            Ok(handle) => Some(handle),
            Err(e) => {
                eprintln!("transcribe thread spawn failed: {e}");
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
        transcribe_cmd_tx,
        summarize_cmd_tx,
        ui_config: config.ui.clone(),
        audio_sample_rate_hz: config.audio.sample_rate,
        audio_mixdown: config.audio.mixdown.clone(),
        session_factory,
        shared_writer,
        initial_context: run.context.clone().unwrap_or_default(),
        participants: run.participants.clone(),
        transcribe_profiles: transcribe_profiles_ui,
        summarize_profiles: summarize_profiles_ui,
    };

    if let Err(e) = tui::run(ctx) {
        eprintln!("tui error: {e}");
        std::process::exit(1);
    }

    if let Some(handle) = transcribe_thread {
        let _ = handle.join();
    }
    if let Some(handle) = summarize_thread {
        let _ = handle.join();
    }
    if let Some(handle) = raw_writer_handle {
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

fn ensure_whisper_model(model: &mut String, models_dir: &std::path::Path) -> Result<(), String> {
    let trimmed = model.trim();
    let candidate = if trimmed.is_empty() {
        init::DEFAULT_WHISPER_MODEL.to_string()
    } else {
        trimmed.to_string()
    };

    if looks_like_path(candidate.as_str()) {
        let path = std::path::Path::new(candidate.as_str());
        if path.exists() {
            *model = candidate;
            return Ok(());
        }
        return Err(format!("whisper model not found at {}", path.display()));
    }

    let path = init::download_model(&candidate, models_dir, false).map_err(|e| e.to_string())?;
    *model = path.to_string_lossy().to_string();
    Ok(())
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
    transcribe: &mut dyn TranscribeProvider,
    chunk: &koe_core::types::AudioChunk,
) -> Result<(Vec<koe_core::types::TranscriptSegment>, u128), koe_core::TranscribeError> {
    let start = Instant::now();
    let segments = transcribe.transcribe(chunk)?;
    Ok((segments, start.elapsed().as_millis()))
}

fn create_summarize_for_mode(
    profiles: &RuntimeProfiles,
    mode: &str,
) -> Result<Box<dyn koe_core::summarize::SummarizeProvider>, koe_core::SummarizeError> {
    let profile = profiles.profile_for_mode(mode);
    create_summarize_provider(
        profile.provider.as_str(),
        Some(profile.model.as_str()),
        non_empty_str(profile.api_key.as_str()),
    )
}

fn apply_notes_patch_state(notes: &mut MeetingNotes, patch: NotesPatch) -> bool {
    let mut changed = false;
    let mut existing_ids: HashSet<String> = notes
        .bullets
        .iter()
        .map(|bullet| bullet.id.clone())
        .collect();
    let mut existing_normalized: HashSet<String> = notes
        .bullets
        .iter()
        .map(|bullet| normalize_text(&bullet.text))
        .collect();

    for op in patch.ops {
        match op {
            NotesOp::Add { id, text, evidence } => {
                let normalized_text = normalize_text(&text);
                if normalized_text.is_empty()
                    || existing_ids.contains(&id)
                    || existing_normalized.contains(&normalized_text)
                    || is_near_duplicate(&normalized_text, &existing_normalized)
                    || !has_min_content_words(&normalized_text)
                {
                    continue;
                }
                notes.bullets.push(NoteBullet {
                    id: id.clone(),
                    text,
                    evidence,
                });
                existing_ids.insert(id);
                existing_normalized.insert(normalized_text);
                changed = true;
            }
        }
    }

    changed
}

fn is_near_duplicate(candidate: &str, existing: &HashSet<String>) -> bool {
    let candidate_tokens = content_tokens(candidate);
    if candidate_tokens.is_empty() {
        return true;
    }
    let candidate_set: HashSet<&str> = candidate_tokens.iter().map(|t| t.as_str()).collect();
    for existing_text in existing {
        let existing_tokens = content_tokens(existing_text);
        if existing_tokens.is_empty() {
            continue;
        }
        let existing_set: HashSet<&str> = existing_tokens.iter().map(|t| t.as_str()).collect();
        let overlap = candidate_set.intersection(&existing_set).count();
        let union = candidate_set.len() + existing_set.len() - overlap;
        if union == 0 {
            continue;
        }
        let jaccard = overlap as f32 / union as f32;
        if jaccard >= 0.75 || overlap >= 5 {
            return true;
        }
    }
    false
}

fn has_min_content_words(normalized: &str) -> bool {
    content_tokens(normalized).len() >= 5
}

fn content_tokens(normalized: &str) -> Vec<String> {
    normalized
        .split_whitespace()
        .filter(|token| !is_stopword(token))
        .map(|token| token.to_string())
        .collect()
}

fn is_stopword(token: &str) -> bool {
    const STOPWORDS: [&str; 28] = [
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have", "if",
        "in", "is", "it", "its", "of", "on", "or", "that", "the", "to", "was", "were", "with",
        "will",
    ];

    STOPWORDS.contains(&token)
}

#[cfg(test)]
mod tests {
    use super::{default_speaker, select_default_microphone, transcribe_with_latency};
    use koe_core::capture::AudioInputDeviceInfo;
    use koe_core::transcribe::TranscribeProvider;
    use koe_core::types::{AudioChunk, AudioSource, TranscriptSegment};
    use std::time::Duration;

    #[test]
    fn default_speaker_maps_sources() {
        assert_eq!(default_speaker(AudioSource::Microphone), Some("Me"));
        assert_eq!(default_speaker(AudioSource::System), Some("Them"));
        assert_eq!(default_speaker(AudioSource::Mixed), Some("Unknown"));
    }

    struct DummyTranscribe {
        delay: Duration,
    }

    impl TranscribeProvider for DummyTranscribe {
        fn name(&self) -> &'static str {
            "dummy"
        }

        fn transcribe(
            &mut self,
            _chunk: &AudioChunk,
        ) -> Result<Vec<TranscriptSegment>, koe_core::TranscribeError> {
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
        let mut transcribe = DummyTranscribe {
            delay: Duration::from_millis(20),
        };
        let chunk = AudioChunk {
            source: AudioSource::System,
            start_pts_ns: 0,
            sample_rate_hz: 16_000,
            pcm_mono_f32: vec![0.0; 160],
        };
        let (_segments, elapsed) = transcribe_with_latency(&mut transcribe, &chunk).unwrap();
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
