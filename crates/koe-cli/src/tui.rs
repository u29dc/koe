use crate::config::UiConfig;
use crate::session::SessionHandle;
use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use koe_core::process::AudioProcessor;
use koe_core::transcript::TranscriptLedger;
use koe_core::types::{
    ActionItem, CaptureStats, MeetingState, NoteItem, NotesOp, NotesPatch, TranscriptSegment,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use std::io;
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::time::Duration;

#[derive(Debug, Clone, Copy)]
pub enum AsrCommand {
    ToggleProvider,
}

pub enum UiEvent {
    Transcript(Vec<TranscriptSegment>),
    NotesPatch(NotesPatch),
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

struct TerminalGuard;

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = terminal::disable_raw_mode();
        let _ = crossterm::execute!(io::stdout(), LeaveAlternateScreen);
    }
}

struct ProcessorGuard(Option<AudioProcessor>);

impl ProcessorGuard {
    fn new(processor: AudioProcessor) -> Self {
        Self(Some(processor))
    }
}

impl Drop for ProcessorGuard {
    fn drop(&mut self) {
        if let Some(mut processor) = self.0.take() {
            processor.stop();
        }
    }
}

pub fn run(
    processor: AudioProcessor,
    ui_rx: Receiver<UiEvent>,
    stats: CaptureStats,
    asr_name: String,
    asr_cmd_tx: Sender<AsrCommand>,
    ui_config: UiConfig,
    mut session: SessionHandle,
) -> Result<(), Box<dyn std::error::Error>> {
    terminal::enable_raw_mode()?;
    let mut stdout = io::stdout();
    crossterm::execute!(stdout, EnterAlternateScreen)?;
    let _terminal_guard = TerminalGuard;
    let _processor_guard = ProcessorGuard::new(processor);

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
    let mut meeting_state = MeetingState::default();
    let mut transcript_lines = vec![Line::from("waiting for transcript...")];
    let mut notes_lines = vec![Line::from("waiting for notes...")];
    let mut asr_connected = true;
    let mut transcript_needs_render = false;
    let mut notes_needs_render = false;
    let mut asr_name = asr_name;
    let mut asr_lag_ms: Option<u128> = None;
    let theme = UiTheme::from_config(&ui_config);
    let mut show_transcript = if ui_config.notes_only_default {
        false
    } else {
        ui_config.show_transcript
    };
    let mut context = session.context().unwrap_or("").to_string();
    let mut editing_context = false;
    let mut context_buffer = String::new();

    loop {
        // Drain all pending transcript updates
        loop {
            match ui_rx.try_recv() {
                Ok(UiEvent::Transcript(segments)) => {
                    if let Err(err) = session.append_transcript(&segments) {
                        eprintln!("session transcript write failed: {err}");
                    }
                    ledger.append(segments);
                    transcript_needs_render = true;
                }
                Ok(UiEvent::NotesPatch(patch)) => {
                    if apply_notes_patch(&mut meeting_state, patch) {
                        if let Err(err) = session.write_notes(&meeting_state) {
                            eprintln!("session notes write failed: {err}");
                        }
                        notes_needs_render = true;
                    }
                }
                Ok(UiEvent::AsrStatus { name, connected }) => {
                    asr_name = name;
                    asr_connected = connected;
                    transcript_needs_render = true;
                }
                Ok(UiEvent::AsrLag { last_ms }) => {
                    asr_lag_ms = Some(last_ms);
                    transcript_needs_render = true;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    asr_connected = false;
                    break;
                }
            }
        }

        if transcript_needs_render {
            transcript_lines = render_transcript_lines(&ledger, &theme);
            transcript_needs_render = false;
        }
        if notes_needs_render {
            notes_lines = render_notes_lines(&meeting_state, &theme);
            notes_needs_render = false;
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

            if editing_context {
                let area = centered_rect(70, 60, frame.area());
                frame.render_widget(Clear, area);
                let title = Span::styled(
                    "Context (Ctrl+S save, Esc cancel)",
                    Style::default().fg(theme.heading),
                );
                let body = if context_buffer.is_empty() {
                    Text::from(Line::from(Span::styled(
                        "type context...",
                        Style::default().fg(theme.muted),
                    )))
                } else {
                    Text::from(context_buffer.clone())
                };
                let editor = Paragraph::new(body)
                    .block(Block::default().borders(Borders::ALL).title(title))
                    .wrap(Wrap { trim: false });
                frame.render_widget(editor, area);
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
                if editing_context {
                    match key.code {
                        KeyCode::Esc => {
                            editing_context = false;
                            context_buffer.clear();
                        }
                        KeyCode::Enter => {
                            context_buffer.push('\n');
                        }
                        KeyCode::Backspace => {
                            context_buffer.pop();
                        }
                        KeyCode::Char('s') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            context = context_buffer.clone();
                            if let Err(err) = session.update_context(context_buffer.clone()) {
                                eprintln!("context update failed: {err}");
                            }
                            editing_context = false;
                        }
                        KeyCode::Char(ch) => {
                            context_buffer.push(ch);
                        }
                        _ => {}
                    }
                    continue;
                }
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
                    transcript_needs_render = true;
                }
                if key.code == KeyCode::Char('c') {
                    editing_context = true;
                    context_buffer = context.clone();
                }
            }
        }
    }

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

fn apply_notes_patch(state: &mut MeetingState, patch: NotesPatch) -> bool {
    let mut changed = false;

    for op in patch.ops {
        match op {
            NotesOp::AddKeyPoint { id, text, evidence } => {
                changed |= upsert_note(&mut state.key_points, id, text, evidence);
            }
            NotesOp::AddDecision { id, text, evidence } => {
                changed |= upsert_note(&mut state.decisions, id, text, evidence);
            }
            NotesOp::AddAction {
                id,
                text,
                owner,
                due,
                evidence,
            } => {
                changed |= upsert_action(&mut state.actions, id, text, owner, due, evidence);
            }
            NotesOp::UpdateAction { id, owner, due } => {
                if let Some(item) = state.actions.iter_mut().find(|item| item.id == id) {
                    let mut updated = false;
                    if owner != item.owner {
                        item.owner = owner;
                        updated = true;
                    }
                    if due != item.due {
                        item.due = due;
                        updated = true;
                    }
                    changed |= updated;
                }
            }
        }
    }

    changed
}

fn upsert_note(items: &mut Vec<NoteItem>, id: String, text: String, evidence: Vec<u64>) -> bool {
    if let Some(item) = items.iter_mut().find(|item| item.id == id) {
        let mut updated = false;
        if item.text != text {
            item.text = text;
            updated = true;
        }
        if item.evidence != evidence {
            item.evidence = evidence;
            updated = true;
        }
        return updated;
    }

    items.push(NoteItem { id, text, evidence });
    true
}

fn upsert_action(
    items: &mut Vec<ActionItem>,
    id: String,
    text: String,
    owner: Option<String>,
    due: Option<String>,
    evidence: Vec<u64>,
) -> bool {
    if let Some(item) = items.iter_mut().find(|item| item.id == id) {
        let mut updated = false;
        if item.text != text {
            item.text = text;
            updated = true;
        }
        if item.owner != owner {
            item.owner = owner;
            updated = true;
        }
        if item.due != due {
            item.due = due;
            updated = true;
        }
        if item.evidence != evidence {
            item.evidence = evidence;
            updated = true;
        }
        return updated;
    }

    items.push(ActionItem {
        id,
        text,
        owner,
        due,
        evidence,
    });
    true
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

fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let [_, middle, _] = Layout::vertical([
        Constraint::Percentage((100 - percent_y) / 2),
        Constraint::Percentage(percent_y),
        Constraint::Percentage((100 - percent_y) / 2),
    ])
    .areas(area);

    let [_, center, _] = Layout::horizontal([
        Constraint::Percentage((100 - percent_x) / 2),
        Constraint::Percentage(percent_x),
        Constraint::Percentage((100 - percent_x) / 2),
    ])
    .areas(middle);

    center
}

#[cfg(test)]
mod tests {
    use super::apply_notes_patch;
    use koe_core::types::{MeetingState, NotesOp, NotesPatch};

    #[test]
    fn apply_notes_patch_upserts_key_point() {
        let mut state = MeetingState::default();
        let patch = NotesPatch {
            ops: vec![NotesOp::AddKeyPoint {
                id: "k1".to_string(),
                text: "first".to_string(),
                evidence: vec![1],
            }],
        };

        assert!(apply_notes_patch(&mut state, patch));
        assert_eq!(state.key_points.len(), 1);
        assert_eq!(state.key_points[0].text, "first");

        let patch = NotesPatch {
            ops: vec![NotesOp::AddKeyPoint {
                id: "k1".to_string(),
                text: "updated".to_string(),
                evidence: vec![1, 2],
            }],
        };
        assert!(apply_notes_patch(&mut state, patch));
        assert_eq!(state.key_points.len(), 1);
        assert_eq!(state.key_points[0].text, "updated");
        assert_eq!(state.key_points[0].evidence, vec![1, 2]);
    }

    #[test]
    fn apply_notes_patch_updates_action_owner() {
        let mut state = MeetingState::default();
        let patch = NotesPatch {
            ops: vec![NotesOp::AddAction {
                id: "a1".to_string(),
                text: "do it".to_string(),
                owner: None,
                due: None,
                evidence: vec![3],
            }],
        };
        assert!(apply_notes_patch(&mut state, patch));
        let patch = NotesPatch {
            ops: vec![NotesOp::UpdateAction {
                id: "a1".to_string(),
                owner: Some("Han".to_string()),
                due: Some("tomorrow".to_string()),
            }],
        };
        assert!(apply_notes_patch(&mut state, patch));
        assert_eq!(state.actions[0].owner.as_deref(), Some("Han"));
        assert_eq!(state.actions[0].due.as_deref(), Some("tomorrow"));
    }
}
