mod tui;

use clap::Parser;
use koe_core::asr::create_asr;
use koe_core::capture::create_capture;
use koe_core::types::{AudioSource, CaptureStats};
use std::sync::mpsc;
use std::thread;

#[derive(Parser)]
#[command(name = "koe", version, about = "meeting transcription engine")]
struct Cli {
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
    let cli = Cli::parse();

    let stats = CaptureStats::new();
    let stats_display = stats.clone();

    let capture = match create_capture() {
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

    let mut asr = match create_asr(cli.asr.as_str(), cli.model_trn.as_deref()) {
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

                let speaker = match chunk.source {
                    AudioSource::Microphone => "me",
                    AudioSource::System => "them",
                    AudioSource::Mixed => "mixed",
                };

                for seg in &mut segments {
                    if seg.speaker.is_none() {
                        seg.speaker = Some(speaker.to_string());
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
