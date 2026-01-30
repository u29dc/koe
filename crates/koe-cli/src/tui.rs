use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use koe_core::process::AudioProcessor;
use koe_core::types::{AudioChunk, CaptureStats};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Layout};
use ratatui::text::Text;
use ratatui::widgets::Paragraph;
use std::io;
use std::sync::mpsc::Receiver;
use std::time::Duration;

pub fn run(
    mut processor: AudioProcessor,
    chunk_rx: Receiver<AudioChunk>,
    stats: CaptureStats,
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
    let mut chunks_received: u64 = 0;

    loop {
        // Drain all pending chunks
        while let Ok(_chunk) = chunk_rx.try_recv() {
            chunks_received += 1;
        }

        // Render
        terminal.draw(|frame| {
            let [transcript_area, status_area] =
                Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).areas(frame.area());

            let transcript = Paragraph::new(Text::raw("waiting for transcript..."));
            frame.render_widget(transcript, transcript_area);

            let status = Paragraph::new(Text::raw(format!(
                "frames: {}  drops: {}  chunks: {}/{}  received: {}",
                stats.frames_captured(),
                stats.frames_dropped(),
                stats.chunks_emitted(),
                stats.chunks_dropped(),
                chunks_received,
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
