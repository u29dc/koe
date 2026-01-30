use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use koe_core::process::AudioProcessor;
use koe_core::transcript::TranscriptLedger;
use koe_core::types::{CaptureStats, TranscriptSegment};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Layout};
use ratatui::text::Text;
use ratatui::widgets::Paragraph;
use std::io;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::Duration;

pub fn run(
    mut processor: AudioProcessor,
    transcript_rx: Receiver<Vec<TranscriptSegment>>,
    stats: CaptureStats,
    asr_name: String,
) -> Result<(), Box<dyn std::error::Error>> {
    terminal::enable_raw_mode()?;
    let mut stdout = io::stdout();
    crossterm::execute!(stdout, EnterAlternateScreen)?;

    // Panic hook to restore terminal on panic
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = terminal::disable_raw_mode();
        let _ = crossterm::execute!(io::stdout(), LeaveAlternateScreen);
        original_hook(info);
    }));

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    let mut ledger = TranscriptLedger::new();
    let mut transcript_text = String::from("waiting for transcript...");
    let mut asr_connected = true;
    let mut needs_render = false;

    loop {
        // Drain all pending transcript updates
        loop {
            match transcript_rx.try_recv() {
                Ok(segments) => {
                    ledger.append(segments);
                    needs_render = true;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    asr_connected = false;
                    break;
                }
            }
        }

        if needs_render {
            if ledger.is_empty() {
                transcript_text = "waiting for transcript...".to_string();
            } else {
                transcript_text = render_transcript(&ledger);
            }
            needs_render = false;
        }

        // Render
        terminal.draw(|frame| {
            let [transcript_area, status_area] =
                Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).areas(frame.area());

            let transcript = Paragraph::new(Text::raw(transcript_text.as_str()));
            frame.render_widget(transcript, transcript_area);

            let asr_status = if asr_connected { "ok" } else { "disconnected" };
            let status = Paragraph::new(Text::raw(format!(
                "asr: {} ({})  frames: {}  drops: {}  chunks: {}/{}  segments: {}",
                asr_name,
                asr_status,
                stats.frames_captured(),
                stats.frames_dropped(),
                stats.chunks_emitted(),
                stats.chunks_dropped(),
                ledger.len(),
            )));
            frame.render_widget(status, status_area);
        })?;

        // Poll for input
        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.code == KeyCode::Char('q')
                    || (key.code == KeyCode::Char('c')
                        && key.modifiers.contains(KeyModifiers::CONTROL))
                {
                    break;
                }
            }
        }
    }

    processor.stop();
    terminal::disable_raw_mode()?;
    crossterm::execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(())
}

fn render_transcript(ledger: &TranscriptLedger) -> String {
    const MAX_SEGMENTS: usize = 200;
    let segments = ledger.segments();
    let start = segments.len().saturating_sub(MAX_SEGMENTS);
    let mut lines = Vec::with_capacity(segments.len().saturating_sub(start));

    for seg in &segments[start..] {
        let mut line = String::new();
        if let Some(speaker) = seg.speaker.as_deref() {
            line.push('[');
            line.push_str(speaker);
            line.push_str("] ");
        }
        line.push_str(seg.text.trim());
        lines.push(line);
    }

    lines.join("\n")
}
