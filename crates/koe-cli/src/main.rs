mod tui;

use clap::Parser;
use koe_core::capture::create_capture;
use koe_core::types::CaptureStats;

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

    let (processor, chunk_rx) = match koe_core::process::AudioProcessor::start(capture, stats) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("processor start failed: {e}");
            std::process::exit(1);
        }
    };

    if let Err(e) = tui::run(processor, chunk_rx, stats_display) {
        eprintln!("tui error: {e}");
        std::process::exit(1);
    }
}
