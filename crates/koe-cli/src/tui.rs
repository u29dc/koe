use crate::config::UiConfig;
use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use koe_core::process::AudioProcessor;
use koe_core::transcript::TranscriptLedger;
use koe_core::types::{CaptureStats, MeetingState, TranscriptSegment};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use std::io;
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::time::Duration;

#[derive(Debug, Clone, Copy)]
pub enum AsrCommand {
    ToggleProvider,
}

pub enum UiEvent {
    Transcript(Vec<TranscriptSegment>),
    AsrStatus { name: String, connected: bool },
    AsrLag { last_ms: u128 },
}

#[derive(Debug, Clone)]
struct UiTheme {
    me: Color,
    them: Color,
    heading: Color,
    muted: Color,
    neutral: Color,
}

impl UiTheme {
    fn from_config(config: &UiConfig) -> Self {
        let _ = config.color_theme.as_str();
        Self::minimal()
    }

    fn minimal() -> Self {
        Self {
            me: Color::Rgb(140, 140, 140),
            them: Color::Rgb(90, 140, 210),
            heading: Color::Rgb(120, 120, 120),
            muted: Color::Rgb(100, 100, 100),
            neutral: Color::Rgb(210, 210, 210),
        }
    }
}

pub fn run(
    mut processor: AudioProcessor,
    ui_rx: Receiver<UiEvent>,
    stats: CaptureStats,
    asr_name: String,
    asr_cmd_tx: Sender<AsrCommand>,
    ui_config: UiConfig,
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
    let meeting_state = MeetingState::default();
    let mut transcript_lines = vec![Line::from("waiting for transcript...")];
    let mut notes_lines = vec![Line::from("waiting for notes...")];
    let mut asr_connected = true;
    let mut needs_render = false;
    let mut asr_name = asr_name;
    let mut asr_lag_ms: Option<u128> = None;
    let theme = UiTheme::from_config(&ui_config);
    let mut show_transcript = if ui_config.notes_only_default {
        false
    } else {
        ui_config.show_transcript
    };

    loop {
        // Drain all pending transcript updates
        loop {
            match ui_rx.try_recv() {
                Ok(UiEvent::Transcript(segments)) => {
                    ledger.append(segments);
                    needs_render = true;
                }
                Ok(UiEvent::AsrStatus { name, connected }) => {
                    asr_name = name;
                    asr_connected = connected;
                    needs_render = true;
                }
                Ok(UiEvent::AsrLag { last_ms }) => {
                    asr_lag_ms = Some(last_ms);
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
            transcript_lines = render_transcript_lines(&ledger, &theme);
            notes_lines = render_notes_lines(&meeting_state, &theme);
            needs_render = false;
        }

        // Render
        terminal.draw(|frame| {
            let [transcript_area, status_area] =
                Layout::vertical([Constraint::Min(1), Constraint::Length(1)]).areas(frame.area());

            if show_transcript {
                let [notes_area, transcript_area] = Layout::horizontal([
                    Constraint::Percentage(55),
                    Constraint::Percentage(45),
                ])
                .areas(transcript_area);

                let notes = Paragraph::new(Text::from(notes_lines.clone()))
                    .block(Block::default().borders(Borders::ALL).title(Span::styled(
                        "Notes",
                        Style::default().fg(theme.heading),
                    )))
                    .wrap(Wrap { trim: true });
                frame.render_widget(notes, notes_area);

                let transcript = Paragraph::new(Text::from(transcript_lines.clone()))
                    .block(Block::default().borders(Borders::ALL).title(Span::styled(
                        "Transcript",
                        Style::default().fg(theme.heading),
                    )))
                    .wrap(Wrap { trim: true });
                frame.render_widget(transcript, transcript_area);
            } else {
                let notes = Paragraph::new(Text::from(notes_lines.clone()))
                    .block(Block::default().borders(Borders::ALL).title(Span::styled(
                        "Notes",
                        Style::default().fg(theme.heading),
                    )))
                    .wrap(Wrap { trim: true });
                frame.render_widget(notes, transcript_area);
            }

            let asr_status = if asr_connected { "ok" } else { "disconnected" };
            let lag_text = asr_lag_ms
                .map(|ms| format!("{ms}ms"))
                .unwrap_or_else(|| "n/a".to_string());
            let status = Paragraph::new(Text::raw(format!(
                "asr: {} ({})  lag: {}  frames: {}  drops: {}  chunks: {}/{}  segments: {}  |  q quit  t transcript  p provider  c context",
                asr_name,
                asr_status,
                lag_text,
                stats.frames_captured(),
                stats.frames_dropped(),
                stats.chunks_emitted(),
                stats.chunks_dropped(),
                ledger.len(),
            )))
            .style(Style::default().fg(theme.muted));
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
                if key.code == KeyCode::Char('p') {
                    let _ = asr_cmd_tx.send(AsrCommand::ToggleProvider);
                }
                if key.code == KeyCode::Char('t') {
                    show_transcript = !show_transcript;
                    needs_render = true;
                }
            }
        }
    }

    processor.stop();
    terminal::disable_raw_mode()?;
    crossterm::execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(())
}

fn render_transcript_lines(ledger: &TranscriptLedger, theme: &UiTheme) -> Vec<Line<'static>> {
    const MAX_SEGMENTS: usize = 200;
    let segments = ledger.segments();
    let start = segments.len().saturating_sub(MAX_SEGMENTS);
    let mut lines = Vec::with_capacity(segments.len().saturating_sub(start).max(1));

    for seg in &segments[start..] {
        let mut spans = Vec::new();
        if let Some(speaker) = seg.speaker.as_deref() {
            let style = speaker_style(theme, speaker);
            spans.push(Span::styled(format!("{speaker}: "), style));
        }
        spans.push(Span::styled(
            seg.text.trim().to_string(),
            Style::default().fg(theme.neutral),
        ));
        lines.push(Line::from(spans));
    }

    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "waiting for transcript...",
            Style::default().fg(theme.muted),
        )));
    }

    lines
}

fn render_notes_lines(state: &MeetingState, theme: &UiTheme) -> Vec<Line<'static>> {
    let mut lines = Vec::new();

    if state.key_points.is_empty() && state.actions.is_empty() && state.decisions.is_empty() {
        lines.push(Line::from(Span::styled(
            "waiting for notes...".to_string(),
            Style::default().fg(theme.muted),
        )));
        return lines;
    }

    if !state.key_points.is_empty() {
        lines.push(heading_line("Key Points".to_string(), theme));
        for item in &state.key_points {
            lines.push(note_line(item.text.clone(), theme));
        }
        lines.push(Line::from(Span::raw("")));
    }

    if !state.actions.is_empty() {
        lines.push(heading_line("Actions".to_string(), theme));
        for item in &state.actions {
            let mut text = item.text.clone();
            if item.owner.is_some() || item.due.is_some() {
                let owner = item.owner.clone().unwrap_or_default();
                let due = item.due.clone().unwrap_or_default();
                text.push_str(" — ");
                if !owner.is_empty() {
                    text.push_str(owner.as_str());
                }
                if !due.is_empty() {
                    if !owner.is_empty() {
                        text.push_str(", ");
                    }
                    text.push_str(due.as_str());
                }
            }
            lines.push(note_line(text, theme));
        }
        lines.push(Line::from(Span::raw("")));
    }

    if !state.decisions.is_empty() {
        lines.push(heading_line("Decisions".to_string(), theme));
        for item in &state.decisions {
            lines.push(note_line(item.text.clone(), theme));
        }
    }

    lines
}

fn heading_line(label: String, theme: &UiTheme) -> Line<'static> {
    Line::from(Span::styled(label, Style::default().fg(theme.heading)))
}

fn note_line(text: String, theme: &UiTheme) -> Line<'static> {
    if let Some(rest) = text.strip_prefix("Me:") {
        return Line::from(vec![
            Span::styled("Me:".to_string(), Style::default().fg(theme.me)),
            Span::styled(
                format!(" {}", rest.trim_start()),
                Style::default().fg(theme.neutral),
            ),
        ]);
    }
    if let Some(rest) = text.strip_prefix("Them:") {
        return Line::from(vec![
            Span::styled("Them:".to_string(), Style::default().fg(theme.them)),
            Span::styled(
                format!(" {}", rest.trim_start()),
                Style::default().fg(theme.neutral),
            ),
        ]);
    }

    Line::from(Span::styled(
        format!("- {text}"),
        Style::default().fg(theme.neutral),
    ))
}

fn speaker_style(theme: &UiTheme, speaker: &str) -> Style {
    match speaker {
        "Me" => Style::default().fg(theme.me),
        "Them" => Style::default().fg(theme.them),
        _ => Style::default().fg(theme.muted),
    }
}
