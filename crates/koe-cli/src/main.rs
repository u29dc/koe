mod init;
mod tui;

use clap::{Parser, Subcommand};
use koe_core::asr::create_asr;
use koe_core::capture::create_capture;
use koe_core::types::{AudioSource, CaptureStats};
use std::sync::mpsc;
use std::thread;

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

    let model_trn_env = std::env::var("KOE_WHISPER_MODEL").ok();
    let model_groq_env = std::env::var("KOE_GROQ_MODEL").ok();
    let mut model_trn_owned = if run.asr == "groq" {
        run.model_trn.clone().or(model_groq_env)
    } else {
        run.model_trn.clone().or(model_trn_env)
    };

    if run.asr == "whisper" && model_trn_owned.is_none() {
        let init_args = init::InitArgs {
            model: "base.en".to_string(),
            dir: None,
            force: false,
            groq_key: None,
        };
        match init::run(&init_args) {
            Ok(path) => model_trn_owned = Some(path.to_string_lossy().to_string()),
            Err(e) => {
                eprintln!("init failed: {e}");
                std::process::exit(1);
            }
        }
    }

    let mut asr = match create_asr(run.asr.as_str(), model_trn_owned.as_deref()) {
        Ok(provider) => provider,
        Err(e) => {
            eprintln!("asr init failed: {e}");
            std::process::exit(1);
        }
    };

    let asr_name = asr.name().to_string();
    let (transcript_tx, transcript_rx) = mpsc::channel();

    let asr_thread = match thread::Builder::new()
        .name("koe-asr".into())
        .spawn(move || {
            while let Ok(chunk) = chunk_rx.recv() {
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

                if let Some(speaker) = default_speaker(chunk.source) {
                    for seg in &mut segments {
                        if seg.speaker.is_none() {
                            seg.speaker = Some(speaker.to_string());
                        }
                    }
                }

                if transcript_tx.send(segments).is_err() {
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

    if let Err(e) = tui::run(processor, transcript_rx, stats_display, asr_name) {
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
