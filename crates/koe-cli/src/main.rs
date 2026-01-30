mod init;
mod tui;

use clap::{Parser, Subcommand};
use koe_core::asr::create_asr;
use koe_core::capture::create_capture;
use koe_core::process::ChunkRecvTimeoutError;
use koe_core::types::{AudioSource, CaptureStats};
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
}

#[derive(Parser, Debug, Clone)]
struct RunArgs {
    /// ASR provider: whisper or groq
    #[arg(long, default_value = "whisper")]
    asr: String,

    /// Transcriber model. whisper: path to GGML file, groq: model name [default: whisper-large-v3-turbo]
    #[arg(long)]
    model_trn: Option<String>,

    /// Summarizer provider: ollama or openrouter
    #[arg(long, default_value = "ollama")]
    summarizer: String,

    /// Summarizer model. ollama: model tag [default: qwen3:30b-a3b], openrouter: model id [default: google/gemini-2.5-flash]
    #[arg(long)]
    model_sum: Option<String>,
}

fn main() {
    dotenvy::dotenv().ok();
    let cli = Cli::parse();
    if let Some(Command::Init(args)) = cli.command {
        if let Err(e) = init::run(&args) {
            eprintln!("init failed: {e}");
            std::process::exit(1);
        }
        return;
    }

    let run = cli.run;
    let stats = CaptureStats::new();
    let stats_display = stats.clone();

    let capture = match create_capture(stats.clone()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("capture init failed: {e}");
            std::process::exit(1);
        }
    };

    let (processor, chunk_rx) = match koe_core::process::AudioProcessor::start(capture, stats) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("processor start failed: {e}");
            std::process::exit(1);
        }
    };

    let whisper_model_env = std::env::var("KOE_WHISPER_MODEL").ok();
    let groq_model_env = std::env::var("KOE_GROQ_MODEL").ok();
    let mut whisper_model = run.model_trn.clone().or(whisper_model_env);
    let groq_model = run.model_trn.clone().or(groq_model_env);

    if run.asr == "whisper" {
        if let Err(e) = ensure_whisper_model(&mut whisper_model) {
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

    let asr_name = asr.name().to_string();
    let (ui_tx, ui_rx) = mpsc::channel();
    let (asr_cmd_tx, asr_cmd_rx) = mpsc::channel();

    let asr_thread = match thread::Builder::new()
        .name("koe-asr".into())
        .spawn(move || {
            let mut current_provider = run.asr.clone();
            let mut whisper_model = whisper_model;
            let groq_model = groq_model;

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
                                    if let Err(e) = ensure_whisper_model(&mut whisper_model) {
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

                let start = Instant::now();
                let mut segments = match asr.transcribe(&chunk) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("asr error: {e}");
                        continue;
                    }
                };

                if segments.is_empty() {
                    continue;
                }

                let elapsed = start.elapsed().as_millis();
                let smoothed = match latency_ms {
                    Some(prev) => (prev * 9 + elapsed) / 10,
                    None => elapsed,
                };
                latency_ms = Some(smoothed);
                let _ = ui_tx.send(UiEvent::AsrLag { last_ms: smoothed });

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

    if let Err(e) = tui::run(processor, ui_rx, stats_display, asr_name, asr_cmd_tx) {
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

fn ensure_whisper_model(model: &mut Option<String>) -> Result<(), String> {
    if model.is_some() {
        return Ok(());
    }

    let init_args = init::InitArgs {
        model: "base.en".to_string(),
        dir: None,
        force: false,
        groq_key: None,
    };
    let path = init::run(&init_args).map_err(|e| e.to_string())?;
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

#[cfg(test)]
mod tests {
    use super::default_speaker;
    use koe_core::types::AudioSource;

    #[test]
    fn default_speaker_maps_sources() {
        assert_eq!(default_speaker(AudioSource::Microphone), Some("Me"));
        assert_eq!(default_speaker(AudioSource::System), Some("Them"));
        assert_eq!(default_speaker(AudioSource::Mixed), Some("Unknown"));
    }
}
