use clap::Parser;
use koe_core::capture::create_capture;
use koe_core::types::CaptureStats;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Parser)]
#[command(name = "koe", version, about = "meeting transcription engine")]
struct Cli {
    /// ASR provider: whisper or groq
    #[arg(long, default_value = "whisper")]
    asr: String,

    /// Summarizer provider: ollama or openrouter
    #[arg(long, default_value = "ollama")]
    summarizer: String,
}

fn main() {
    let _cli = Cli::parse();

    let stats = CaptureStats::new();
    let stats_display = stats.clone();

    let capture = match create_capture() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("capture init failed: {e}");
            std::process::exit(1);
        }
    };

    let (mut processor, chunk_rx) = match koe_core::process::AudioProcessor::start(capture, stats) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("processor start failed: {e}");
            std::process::exit(1);
        }
    };

    // Ctrl+C handler using signal-hook (transitive dep via crossterm)
    let running = Arc::new(AtomicBool::new(true));
    signal_hook::flag::register(signal_hook::consts::SIGINT, Arc::clone(&running))
        .expect("failed to register SIGINT handler");

    println!(
        "koe {} - capturing audio (ctrl+c to stop)",
        koe_core::version()
    );

    // running starts as true, signal sets it to false
    while running.load(Ordering::Relaxed) {
        match chunk_rx.recv_timeout(std::time::Duration::from_secs(1)) {
            Ok(chunk) => {
                let duration_ms = (chunk.pcm_mono_f32.len() as f64 / 16.0) as u64;
                println!(
                    "[{:?}] chunk: {}ms, {} samples | frames: {}, drops: {}, chunks: {}/{}",
                    chunk.source,
                    duration_ms,
                    chunk.pcm_mono_f32.len(),
                    stats_display.frames_captured(),
                    stats_display.frames_dropped(),
                    stats_display.chunks_emitted(),
                    stats_display.chunks_dropped(),
                );
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if !running.load(Ordering::Relaxed) {
                    break;
                }
                println!(
                    "waiting... frames: {}, drops: {}, chunks: {}/{}",
                    stats_display.frames_captured(),
                    stats_display.frames_dropped(),
                    stats_display.chunks_emitted(),
                    stats_display.chunks_dropped(),
                );
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    println!("stopping...");
    processor.stop();
    println!("done.");
}
