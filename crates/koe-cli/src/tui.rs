use crate::config::UiConfig;
use crate::raw_audio::{RawAudioWriter, SharedRawAudioWriter};
use crate::session::{SessionFactory, SessionHandle};
use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use koe_core::process::AudioProcessor;
use koe_core::transcript::TranscriptLedger;
use koe_core::types::{
    ActionItem, CaptureStats, MeetingState, NoteItem, NotesOp, NotesPatch, TranscriptSegment,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use std::io;
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::sync::mpsc::{Receiver, Sender, TryRecvError, channel};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy)]
pub enum TranscribeCommand {
    ToggleMode,
}

#[derive(Debug, Clone)]
pub enum SummarizeCommand {
    ToggleMode,
    Force,
    Reset,
    UpdateContext(String),
}

pub enum UiEvent {
    Transcript(Vec<TranscriptSegment>),
    NotesPatch(NotesPatch),
    TranscribeStatus {
        mode: String,
        provider: String,
        connected: bool,
    },
    SummarizeStatus {
        mode: String,
        provider: String,
    },
    TranscribeLag {
        last_ms: u128,
    },
}

#[derive(Debug, Clone)]
pub struct ProfileSummary {
    pub provider: String,
    pub model: String,
}

#[derive(Debug, Clone)]
pub struct ModeProfiles {
    pub active: String,
    pub local: ProfileSummary,
    pub cloud: ProfileSummary,
}

impl ModeProfiles {
    fn active_profile(&self) -> &ProfileSummary {
        self.profile_for_mode(self.active.as_str())
    }

    fn active_profile_mut(&mut self) -> &mut ProfileSummary {
        let is_cloud = self.active == "cloud";
        if is_cloud {
            &mut self.cloud
        } else {
            &mut self.local
        }
    }

    fn profile_for_mode(&self, mode: &str) -> &ProfileSummary {
        if mode == "cloud" {
            &self.cloud
        } else {
            &self.local
        }
    }

    fn profile_for_mode_mut(&mut self, mode: &str) -> &mut ProfileSummary {
        if mode == "cloud" {
            &mut self.cloud
        } else {
            &mut self.local
        }
    }

    fn set_provider(&mut self, mode: &str, provider: String) {
        self.profile_for_mode_mut(mode).provider = provider;
    }
}

pub struct TuiContext {
    pub processor: AudioProcessor,
    pub ui_rx: Receiver<UiEvent>,
    pub stats: CaptureStats,
    pub transcribe_cmd_tx: Sender<TranscribeCommand>,
    pub summarize_cmd_tx: Sender<SummarizeCommand>,
    pub ui_config: UiConfig,
    pub session_factory: SessionFactory,
    pub shared_writer: SharedRawAudioWriter,
    pub initial_context: String,
    pub participants: Vec<String>,
    pub transcribe_profiles: ModeProfiles,
    pub summarize_profiles: ModeProfiles,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MeetingPhase {
    Idle,
    MeetingActive,
    PostMeeting,
}

#[derive(Debug, Clone)]
struct UiTheme {
    accent: Color,
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
            accent: Color::Rgb(80, 200, 200),
            me: Color::Rgb(170, 170, 170),
            them: Color::Rgb(150, 150, 150),
            heading: Color::Rgb(140, 140, 140),
            muted: Color::Rgb(110, 110, 110),
            neutral: Color::Rgb(210, 210, 210),
        }
    }
}

#[derive(Debug, Clone)]
struct PaletteState {
    filter: String,
    selected: usize,
}

impl PaletteState {
    fn new() -> Self {
        Self {
            filter: String::new(),
            selected: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TextInputKind {
    Context,
    TranscribeModel,
    SummarizeModel,
}

#[derive(Debug, Clone)]
struct TextInputState {
    kind: TextInputKind,
    buffer: String,
}

#[derive(Debug, Clone)]
enum UiMode {
    Normal,
    Palette(PaletteState),
    TextInput(TextInputState),
}

#[derive(Debug, Clone, Copy)]
enum PaletteCommandId {
    StartMeeting,
    EndMeeting,
    PauseCapture,
    ResumeCapture,
    ForceSummarize,
    SwitchTranscribeMode,
    SwitchSummarizeMode,
    SetTranscribeModel,
    SetSummarizeModel,
    EditContext,
    BrowseSessions,
    CopyTranscriptPath,
    CopyNotesPath,
    CopyAudioPath,
    OpenSessionFolder,
    ExportMarkdown,
    StartNewMeeting,
}

#[derive(Debug, Clone, Copy)]
struct PaletteCommand {
    id: PaletteCommandId,
    label: &'static str,
    category: &'static str,
}

struct Waveform {
    frames: Vec<&'static str>,
    index: usize,
    last_tick: Instant,
}

impl Waveform {
    fn new() -> Self {
        Self {
            frames: vec!["▁▂▃▅▃▂▁", "▂▃▅▃▂▁▂", "▃▅▃▂▁▂▃", "▅▃▂▁▂▃▅"],
            index: 0,
            last_tick: Instant::now(),
        }
    }

    fn current(&self) -> &'static str {
        self.frames[self.index % self.frames.len()]
    }

    fn tick(&mut self) {
        if self.last_tick.elapsed() >= Duration::from_millis(120) {
            self.index = (self.index + 1) % self.frames.len();
            self.last_tick = Instant::now();
        }
    }
}

struct StartMeetingInput<'a> {
    factory: &'a SessionFactory,
    shared_writer: &'a SharedRawAudioWriter,
    transcribe_profiles: &'a ModeProfiles,
    summarize_profiles: &'a ModeProfiles,
    context: &'a str,
    participants: &'a [String],
}

struct FooterState<'a> {
    phase: MeetingPhase,
    capture_paused: bool,
    elapsed: Duration,
    waveform: &'a Waveform,
    transcribe_mode: &'a str,
    transcribe_connected: bool,
    transcribe_lag_ms: Option<u128>,
    stats: &'a CaptureStats,
    ledger: &'a TranscriptLedger,
}

struct TerminalGuard;

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = terminal::disable_raw_mode();
        let _ = crossterm::execute!(io::stdout(), LeaveAlternateScreen);
    }
}

pub fn run(ctx: TuiContext) -> Result<(), Box<dyn std::error::Error>> {
    terminal::enable_raw_mode()?;
    let mut stdout = io::stdout();
    crossterm::execute!(stdout, EnterAlternateScreen)?;
    let _terminal_guard = TerminalGuard;

    // Panic hook to restore terminal on panic
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = terminal::disable_raw_mode();
        let _ = crossterm::execute!(io::stdout(), LeaveAlternateScreen);
        original_hook(info);
    }));

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut processor = ctx.processor;
    let theme = UiTheme::from_config(&ctx.ui_config);
    let mut ledger = TranscriptLedger::new();
    let mut meeting_state = MeetingState::default();
    let mut transcript_lines = render_transcript_lines(&ledger, &theme);
    let mut notes_lines = render_notes_lines(&meeting_state, &theme);
    let mut transcribe_connected = true;
    let mut transcribe_lag_ms: Option<u128> = None;
    let mut phase = MeetingPhase::Idle;
    let mut mode = UiMode::Normal;
    let mut meeting_started_at: Option<Instant> = None;
    let mut meeting_elapsed = Duration::ZERO;
    let mut capture_paused = true;
    let mut context = ctx.initial_context.clone();
    let mut transcribe_profiles = ctx.transcribe_profiles.clone();
    let mut summarize_profiles = ctx.summarize_profiles.clone();
    let mut session: Option<SessionHandle> = None;
    let mut session_finalized = false;
    let mut waveform = Waveform::new();

    processor.pause();

    loop {
        // Drain all pending UI events
        loop {
            match ctx.ui_rx.try_recv() {
                Ok(UiEvent::Transcript(segments)) => {
                    if phase == MeetingPhase::MeetingActive && !capture_paused {
                        if let Some(active_session) = session.as_mut() {
                            if let Err(err) = active_session.append_transcript(&segments) {
                                eprintln!("session transcript write failed: {err}");
                            }
                        }
                        ledger.append(segments);
                        transcript_lines = render_transcript_lines(&ledger, &theme);
                    }
                }
                Ok(UiEvent::NotesPatch(patch)) => {
                    if phase == MeetingPhase::MeetingActive
                        && !capture_paused
                        && apply_notes_patch(&mut meeting_state, patch)
                    {
                        if let Some(active_session) = session.as_mut() {
                            if let Err(err) = active_session.write_notes(&meeting_state) {
                                eprintln!("session notes write failed: {err}");
                            }
                        }
                        notes_lines = render_notes_lines(&meeting_state, &theme);
                    }
                }
                Ok(UiEvent::TranscribeStatus {
                    mode,
                    provider,
                    connected,
                }) => {
                    transcribe_profiles.active = mode.clone();
                    transcribe_profiles.set_provider(&mode, provider);
                    transcribe_connected = connected;
                    if let Some(active_session) = session.as_mut() {
                        let profile = transcribe_profiles.active_profile();
                        if let Err(err) = active_session
                            .update_transcribe(profile.provider.clone(), profile.model.clone())
                        {
                            eprintln!("session transcribe update failed: {err}");
                        }
                    }
                }
                Ok(UiEvent::SummarizeStatus { mode, provider }) => {
                    summarize_profiles.active = mode.clone();
                    summarize_profiles.set_provider(&mode, provider);
                    if let Some(active_session) = session.as_mut() {
                        let profile = summarize_profiles.active_profile();
                        if let Err(err) = active_session
                            .update_summarize(profile.provider.clone(), profile.model.clone())
                        {
                            eprintln!("session summarize update failed: {err}");
                        }
                    }
                }
                Ok(UiEvent::TranscribeLag { last_ms }) => {
                    transcribe_lag_ms = Some(last_ms);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    transcribe_connected = false;
                    break;
                }
            }
        }

        if phase == MeetingPhase::MeetingActive {
            if let Some(started) = meeting_started_at {
                meeting_elapsed = started.elapsed();
            }
        }

        if phase == MeetingPhase::MeetingActive && !capture_paused {
            waveform.tick();
        }

        terminal.draw(|frame| {
            let [title_area, content_area, footer_area] = Layout::vertical([
                Constraint::Length(1),
                Constraint::Min(1),
                Constraint::Length(1),
            ])
            .areas(frame.area());

            render_title_bar(frame, title_area, &theme);

            let [notes_area, separator_area, transcript_area] = Layout::horizontal([
                Constraint::Percentage(55),
                Constraint::Length(1),
                Constraint::Percentage(45),
            ])
            .areas(content_area);

            let separator = Paragraph::new(Text::from(Line::from(Span::styled(
                "|",
                Style::default().fg(theme.muted),
            ))));
            frame.render_widget(separator, separator_area);

            render_scrolled_paragraph(frame, notes_area, &notes_lines);
            render_scrolled_paragraph(frame, transcript_area, &transcript_lines);

            let footer_state = FooterState {
                phase,
                capture_paused,
                elapsed: meeting_elapsed,
                waveform: &waveform,
                transcribe_mode: transcribe_profiles.active.as_str(),
                transcribe_connected,
                transcribe_lag_ms,
                stats: &ctx.stats,
                ledger: &ledger,
            };
            render_footer(frame, footer_area, &theme, footer_state);

            match &mode {
                UiMode::Palette(state) => {
                    render_palette(frame, state, &theme, phase, capture_paused);
                }
                UiMode::TextInput(input) => {
                    render_text_input(frame, input, &theme);
                }
                UiMode::Normal => {}
            }
        })?;

        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
                    break;
                }

                match &mut mode {
                    UiMode::Normal => {
                        if key.code == KeyCode::Char('q') {
                            break;
                        }
                        if key.code == KeyCode::Char('p')
                            && key.modifiers.contains(KeyModifiers::CONTROL)
                        {
                            mode = UiMode::Palette(PaletteState::new());
                        }
                    }
                    UiMode::Palette(state) => {
                        if key.code == KeyCode::Esc {
                            mode = UiMode::Normal;
                            continue;
                        }
                        if key.code == KeyCode::Up && state.selected > 0 {
                            state.selected -= 1;
                        }
                        if key.code == KeyCode::Down {
                            state.selected = state.selected.saturating_add(1);
                        }
                        if key.code == KeyCode::Backspace {
                            state.filter.pop();
                            state.selected = 0;
                        }
                        if let KeyCode::Char(ch) = key.code {
                            if !key.modifiers.contains(KeyModifiers::CONTROL) {
                                state.filter.push(ch);
                                state.selected = 0;
                            }
                        }
                        if key.code == KeyCode::Enter {
                            let commands = filtered_commands(phase, capture_paused, &state.filter);
                            if let Some(command) = commands.get(state.selected) {
                                match command.id {
                                    PaletteCommandId::StartMeeting => {
                                        let start_input = StartMeetingInput {
                                            factory: &ctx.session_factory,
                                            shared_writer: &ctx.shared_writer,
                                            transcribe_profiles: &transcribe_profiles,
                                            summarize_profiles: &summarize_profiles,
                                            context: &context,
                                            participants: &ctx.participants,
                                        };
                                        if let Ok(new_session) = start_meeting(start_input) {
                                            session = Some(new_session);
                                            session_finalized = false;
                                            meeting_state = MeetingState::default();
                                            ledger = TranscriptLedger::new();
                                            transcript_lines =
                                                render_transcript_lines(&ledger, &theme);
                                            notes_lines =
                                                render_notes_lines(&meeting_state, &theme);
                                            meeting_started_at = Some(Instant::now());
                                            meeting_elapsed = Duration::ZERO;
                                            phase = MeetingPhase::MeetingActive;
                                            capture_paused = false;
                                            processor.resume();
                                            let _ =
                                                ctx.summarize_cmd_tx.send(SummarizeCommand::Reset);
                                            let _ = ctx.summarize_cmd_tx.send(
                                                SummarizeCommand::UpdateContext(context.clone()),
                                            );
                                        }
                                    }
                                    PaletteCommandId::EndMeeting => {
                                        if let Some(active_session) = session.as_mut() {
                                            let segments = ledger.segments().to_vec();
                                            let state_snapshot = meeting_state.clone();
                                            if let Err(err) = export_session_with_timeout(
                                                active_session.clone(),
                                                segments,
                                                state_snapshot,
                                            ) {
                                                eprintln!("export failed: {err}");
                                            } else if let Err(err) = active_session.finalize() {
                                                eprintln!("session finalize failed: {err}");
                                            }
                                            session_finalized = true;
                                        }
                                        processor.pause();
                                        ctx.shared_writer.set(None);
                                        capture_paused = true;
                                        phase = MeetingPhase::PostMeeting;
                                    }
                                    PaletteCommandId::PauseCapture => {
                                        processor.pause();
                                        capture_paused = true;
                                    }
                                    PaletteCommandId::ResumeCapture => {
                                        processor.resume();
                                        capture_paused = false;
                                    }
                                    PaletteCommandId::ForceSummarize => {
                                        let _ = ctx.summarize_cmd_tx.send(SummarizeCommand::Force);
                                    }
                                    PaletteCommandId::SwitchTranscribeMode => {
                                        let _ = ctx
                                            .transcribe_cmd_tx
                                            .send(TranscribeCommand::ToggleMode);
                                    }
                                    PaletteCommandId::SwitchSummarizeMode => {
                                        let _ =
                                            ctx.summarize_cmd_tx.send(SummarizeCommand::ToggleMode);
                                    }
                                    PaletteCommandId::SetTranscribeModel => {
                                        let buffer =
                                            transcribe_profiles.active_profile().model.clone();
                                        mode = UiMode::TextInput(TextInputState {
                                            kind: TextInputKind::TranscribeModel,
                                            buffer,
                                        });
                                        continue;
                                    }
                                    PaletteCommandId::SetSummarizeModel => {
                                        mode = UiMode::TextInput(TextInputState {
                                            kind: TextInputKind::SummarizeModel,
                                            buffer: summarize_profiles
                                                .active_profile()
                                                .model
                                                .clone(),
                                        });
                                        continue;
                                    }
                                    PaletteCommandId::EditContext => {
                                        mode = UiMode::TextInput(TextInputState {
                                            kind: TextInputKind::Context,
                                            buffer: context.clone(),
                                        });
                                        continue;
                                    }
                                    PaletteCommandId::BrowseSessions => {
                                        if let Err(err) =
                                            open_path(ctx.session_factory.sessions_dir())
                                        {
                                            eprintln!("open sessions failed: {err}");
                                        }
                                    }
                                    PaletteCommandId::CopyTranscriptPath => {
                                        if let Some(active_session) = session.as_ref() {
                                            if let Ok(path) =
                                                active_session.export_transcript_path()
                                            {
                                                if let Err(err) = copy_to_clipboard(&path) {
                                                    eprintln!("copy failed: {err}");
                                                }
                                            }
                                        }
                                    }
                                    PaletteCommandId::CopyNotesPath => {
                                        if let Some(active_session) = session.as_ref() {
                                            if let Ok(path) = active_session.export_notes_path() {
                                                if let Err(err) = copy_to_clipboard(&path) {
                                                    eprintln!("copy failed: {err}");
                                                }
                                            }
                                        }
                                    }
                                    PaletteCommandId::CopyAudioPath => {
                                        if let Some(active_session) = session.as_ref() {
                                            if let Err(err) =
                                                copy_to_clipboard(&active_session.audio_raw_path())
                                            {
                                                eprintln!("copy failed: {err}");
                                            }
                                        }
                                    }
                                    PaletteCommandId::OpenSessionFolder => {
                                        if let Some(active_session) = session.as_ref() {
                                            if let Err(err) =
                                                open_path(active_session.session_dir())
                                            {
                                                eprintln!("open session failed: {err}");
                                            }
                                        }
                                    }
                                    PaletteCommandId::ExportMarkdown => {
                                        if let Some(active_session) = session.as_mut() {
                                            if let Err(err) = active_session
                                                .export_transcript_markdown(ledger.segments())
                                            {
                                                eprintln!("export transcript failed: {err}");
                                            }
                                            if let Err(err) =
                                                active_session.export_notes_markdown(&meeting_state)
                                            {
                                                eprintln!("export notes failed: {err}");
                                            }
                                        }
                                    }
                                    PaletteCommandId::StartNewMeeting => {
                                        if let Some(active_session) = session.as_mut() {
                                            if !active_session.is_finalized() && !session_finalized
                                            {
                                                let segments = ledger.segments().to_vec();
                                                let state_snapshot = meeting_state.clone();
                                                let _ = export_session_with_timeout(
                                                    active_session.clone(),
                                                    segments,
                                                    state_snapshot,
                                                );
                                                let _ = active_session.finalize();
                                            }
                                        }
                                        session = None;
                                        session_finalized = false;
                                        meeting_state = MeetingState::default();
                                        ledger = TranscriptLedger::new();
                                        transcript_lines = render_transcript_lines(&ledger, &theme);
                                        notes_lines = render_notes_lines(&meeting_state, &theme);
                                        meeting_started_at = None;
                                        meeting_elapsed = Duration::ZERO;
                                        phase = MeetingPhase::Idle;
                                        capture_paused = true;
                                        processor.pause();
                                        ctx.shared_writer.set(None);

                                        let _ = ctx.summarize_cmd_tx.send(SummarizeCommand::Reset);

                                        let start_input = StartMeetingInput {
                                            factory: &ctx.session_factory,
                                            shared_writer: &ctx.shared_writer,
                                            transcribe_profiles: &transcribe_profiles,
                                            summarize_profiles: &summarize_profiles,
                                            context: &context,
                                            participants: &ctx.participants,
                                        };
                                        if let Ok(new_session) = start_meeting(start_input) {
                                            session = Some(new_session);
                                            session_finalized = false;
                                            meeting_state = MeetingState::default();
                                            ledger = TranscriptLedger::new();
                                            transcript_lines =
                                                render_transcript_lines(&ledger, &theme);
                                            notes_lines =
                                                render_notes_lines(&meeting_state, &theme);
                                            meeting_started_at = Some(Instant::now());
                                            meeting_elapsed = Duration::ZERO;
                                            phase = MeetingPhase::MeetingActive;
                                            capture_paused = false;
                                            processor.resume();
                                            let _ = ctx.summarize_cmd_tx.send(
                                                SummarizeCommand::UpdateContext(context.clone()),
                                            );
                                        }
                                    }
                                }
                            }
                            mode = UiMode::Normal;
                        }
                    }
                    UiMode::TextInput(input) => {
                        if key.code == KeyCode::Esc {
                            mode = UiMode::Normal;
                            continue;
                        }
                        if key.code == KeyCode::Char('s')
                            && key.modifiers.contains(KeyModifiers::CONTROL)
                        {
                            match input.kind {
                                TextInputKind::Context => {
                                    context = input.buffer.clone();
                                    if let Some(active_session) = session.as_mut() {
                                        if let Err(err) =
                                            active_session.update_context(input.buffer.clone())
                                        {
                                            eprintln!("context update failed: {err}");
                                        }
                                    }
                                    let _ = ctx
                                        .summarize_cmd_tx
                                        .send(SummarizeCommand::UpdateContext(context.clone()));
                                }
                                TextInputKind::TranscribeModel => {
                                    transcribe_profiles.active_profile_mut().model =
                                        input.buffer.trim().to_string();
                                    if let Some(active_session) = session.as_mut() {
                                        let profile = transcribe_profiles.active_profile();
                                        if let Err(err) = active_session.update_transcribe(
                                            profile.provider.clone(),
                                            profile.model.clone(),
                                        ) {
                                            eprintln!("session transcribe update failed: {err}");
                                        }
                                    }
                                }
                                TextInputKind::SummarizeModel => {
                                    summarize_profiles.active_profile_mut().model =
                                        input.buffer.trim().to_string();
                                    if let Some(active_session) = session.as_mut() {
                                        let profile = summarize_profiles.active_profile();
                                        if let Err(err) = active_session.update_summarize(
                                            profile.provider.clone(),
                                            profile.model.clone(),
                                        ) {
                                            eprintln!("session summarize update failed: {err}");
                                        }
                                    }
                                }
                            }
                            mode = UiMode::Normal;
                            continue;
                        }
                        if key.code == KeyCode::Backspace {
                            input.buffer.pop();
                            continue;
                        }
                        if key.code == KeyCode::Enter {
                            match input.kind {
                                TextInputKind::Context => input.buffer.push('\n'),
                                TextInputKind::TranscribeModel | TextInputKind::SummarizeModel => {
                                    let buffer = input.buffer.clone();
                                    match input.kind {
                                        TextInputKind::TranscribeModel => {
                                            transcribe_profiles.active_profile_mut().model =
                                                buffer.trim().to_string();
                                            if let Some(active_session) = session.as_mut() {
                                                let profile = transcribe_profiles.active_profile();
                                                if let Err(err) = active_session.update_transcribe(
                                                    profile.provider.clone(),
                                                    profile.model.clone(),
                                                ) {
                                                    eprintln!(
                                                        "session transcribe update failed: {err}"
                                                    );
                                                }
                                            }
                                        }
                                        TextInputKind::SummarizeModel => {
                                            summarize_profiles.active_profile_mut().model =
                                                buffer.trim().to_string();
                                            if let Some(active_session) = session.as_mut() {
                                                let profile = summarize_profiles.active_profile();
                                                if let Err(err) = active_session.update_summarize(
                                                    profile.provider.clone(),
                                                    profile.model.clone(),
                                                ) {
                                                    eprintln!(
                                                        "session summarize update failed: {err}"
                                                    );
                                                }
                                            }
                                        }
                                        _ => {}
                                    }
                                    mode = UiMode::Normal;
                                }
                            }
                            continue;
                        }
                        if let KeyCode::Char(ch) = key.code {
                            if !key.modifiers.contains(KeyModifiers::CONTROL) {
                                input.buffer.push(ch);
                            }
                        }
                    }
                }
            }
        }
    }

    processor.stop();

    if let Some(mut active_session) = session {
        if !active_session.is_finalized() && !session_finalized {
            let segments = ledger.segments().to_vec();
            let state_snapshot = meeting_state.clone();
            let _ = export_session_with_timeout(active_session.clone(), segments, state_snapshot);
            let _ = active_session.finalize();
        }
    }

    Ok(())
}

fn start_meeting(
    input: StartMeetingInput<'_>,
) -> Result<SessionHandle, crate::session::SessionError> {
    let transcribe_profile = input.transcribe_profiles.active_profile();
    let summarize_profile = input.summarize_profiles.active_profile();
    let session = input.factory.create(
        transcribe_profile.provider.to_string(),
        transcribe_profile.model.to_string(),
        summarize_profile.provider.to_string(),
        summarize_profile.model.to_string(),
        if input.context.trim().is_empty() {
            None
        } else {
            Some(input.context.to_string())
        },
        input.participants.to_vec(),
    )?;
    let audio_raw = session.open_audio_raw()?;
    input
        .shared_writer
        .set(Some(RawAudioWriter::new(audio_raw)));
    Ok(session)
}

fn render_title_bar(frame: &mut ratatui::Frame, area: Rect, theme: &UiTheme) {
    let hint = "ctrl+p command palette";
    let hint_len = hint.len() as u16;
    let [left, right] =
        Layout::horizontal([Constraint::Min(1), Constraint::Length(hint_len + 1)]).areas(area);

    let version = env!("CARGO_PKG_VERSION");
    let left_line = Line::from(vec![
        Span::styled("■ ", Style::default().fg(theme.accent)),
        Span::styled(format!("koe v{version}"), Style::default().fg(theme.accent)),
    ]);
    let right_line = Line::from(Span::styled(hint, Style::default().fg(theme.muted)));

    frame.render_widget(Paragraph::new(left_line), left);
    frame.render_widget(
        Paragraph::new(right_line).alignment(Alignment::Right),
        right,
    );
}

fn render_footer(frame: &mut ratatui::Frame, area: Rect, theme: &UiTheme, state: FooterState) {
    let timer_text = match state.phase {
        MeetingPhase::MeetingActive => format_duration(state.elapsed),
        MeetingPhase::PostMeeting => format_duration(state.elapsed),
        MeetingPhase::Idle => "--:--".to_string(),
    };
    let timer_style = match state.phase {
        MeetingPhase::MeetingActive => Style::default().fg(theme.accent),
        _ => Style::default().fg(theme.muted),
    };

    let wave_text = if state.phase == MeetingPhase::MeetingActive && !state.capture_paused {
        state.waveform.current().to_string()
    } else {
        "----------".to_string()
    };

    let transcribe_state = if state.transcribe_connected {
        "ok"
    } else {
        "disc"
    };
    let lag = state
        .transcribe_lag_ms
        .map(|ms| format!("{:.1}", ms as f64 / 1000.0))
        .unwrap_or_else(|| "n/a".to_string());
    let metrics = format!(
        "transcribe:{} {transcribe_state} lag:{lag}s chunks:{}/{} segs:{}",
        state.transcribe_mode,
        state.stats.chunks_emitted(),
        state.stats.chunks_dropped(),
        state.ledger.len(),
    );

    let [left, middle, right] = Layout::horizontal([
        Constraint::Length(timer_text.len() as u16 + 1),
        Constraint::Length(wave_text.len() as u16 + 2),
        Constraint::Min(1),
    ])
    .areas(area);

    frame.render_widget(Paragraph::new(timer_text).style(timer_style), left);
    frame.render_widget(
        Paragraph::new(wave_text).style(Style::default().fg(theme.muted)),
        middle,
    );
    frame.render_widget(
        Paragraph::new(metrics)
            .alignment(Alignment::Right)
            .style(Style::default().fg(theme.muted)),
        right,
    );
}

fn render_palette(
    frame: &mut ratatui::Frame,
    state: &PaletteState,
    theme: &UiTheme,
    phase: MeetingPhase,
    capture_paused: bool,
) {
    let width = 60.min(frame.area().width.saturating_sub(4) as usize) as u16;
    let height = 2 + 1 + 12;
    let area = centered_rect(width, height, frame.area());
    frame.render_widget(Clear, area);
    frame.render_widget(Block::default().borders(Borders::ALL), area);

    let inner = Rect {
        x: area.x + 1,
        y: area.y + 1,
        width: area.width.saturating_sub(2),
        height: area.height.saturating_sub(2),
    };

    let [title_area, input_area, list_area] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Min(1),
    ])
    .areas(inner);

    frame.render_widget(
        Paragraph::new("Command Palette")
            .alignment(Alignment::Center)
            .style(Style::default().fg(theme.heading)),
        title_area,
    );

    let input_line = format!("> {}", state.filter);
    frame.render_widget(Paragraph::new(input_line), input_area);

    let commands = filtered_commands(phase, capture_paused, &state.filter);
    let selected = if commands.is_empty() {
        0
    } else {
        state.selected.min(commands.len().saturating_sub(1))
    };
    let visible = limit_commands(&commands, selected, list_area.height as usize);
    let lines = render_command_lines(visible, theme, list_area.width as usize);
    frame.render_widget(
        Paragraph::new(Text::from(lines)).wrap(Wrap { trim: true }),
        list_area,
    );
}

fn render_command_lines(
    commands: Vec<(PaletteCommand, bool)>,
    theme: &UiTheme,
    width: usize,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for (command, is_selected) in commands.into_iter() {
        let label = command.label;
        let category = command.category;
        let available = width.saturating_sub(1);
        let gap = available
            .saturating_sub(label.len())
            .saturating_sub(category.len())
            .max(1);
        let padding = " ".repeat(gap);
        let spans = if is_selected {
            vec![
                Span::styled(
                    label.to_string(),
                    Style::default().fg(Color::Black).bg(theme.accent),
                ),
                Span::styled(padding, Style::default().bg(theme.accent)),
                Span::styled(
                    category.to_string(),
                    Style::default().fg(Color::Black).bg(theme.accent),
                ),
            ]
        } else {
            vec![
                Span::styled(label.to_string(), Style::default().fg(theme.neutral)),
                Span::styled(padding, Style::default().fg(theme.neutral)),
                Span::styled(category.to_string(), Style::default().fg(theme.muted)),
            ]
        };

        lines.push(Line::from(spans));
    }
    lines
}

fn render_text_input(frame: &mut ratatui::Frame, input: &TextInputState, theme: &UiTheme) {
    let width = 70.min(frame.area().width.saturating_sub(4) as usize) as u16;
    let height = 10.min(frame.area().height.saturating_sub(4) as usize) as u16;
    let area = centered_rect(width, height, frame.area());
    frame.render_widget(Clear, area);

    let title = match input.kind {
        TextInputKind::Context => "Context (Ctrl+S save, Esc cancel)",
        TextInputKind::TranscribeModel => "Transcribe model (Enter save, Esc cancel)",
        TextInputKind::SummarizeModel => "Summarize Model (Enter save, Esc cancel)",
    };

    let body = if input.buffer.is_empty() {
        Text::from(Line::from(Span::styled(
            "type...",
            Style::default().fg(theme.muted),
        )))
    } else {
        Text::from(input.buffer.clone())
    };

    let editor = Paragraph::new(body)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(Span::styled(title, Style::default().fg(theme.heading))),
        )
        .wrap(Wrap { trim: false });
    frame.render_widget(editor, area);
}

fn render_scrolled_paragraph(frame: &mut ratatui::Frame, area: Rect, lines: &[Line<'static>]) {
    let scroll = lines.len().saturating_sub(area.height as usize) as u16;
    let padded = pad_lines(lines);
    let paragraph = Paragraph::new(Text::from(padded))
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));
    frame.render_widget(paragraph, area);
}

fn pad_lines(lines: &[Line<'static>]) -> Vec<Line<'static>> {
    lines
        .iter()
        .cloned()
        .map(|line| {
            let mut spans = vec![Span::raw(" ")];
            spans.extend(line.spans);
            Line::from(spans)
        })
        .collect()
}

fn render_transcript_lines(ledger: &TranscriptLedger, theme: &UiTheme) -> Vec<Line<'static>> {
    const MAX_SEGMENTS: usize = 200;
    let segments = ledger.segments();
    let start = segments.len().saturating_sub(MAX_SEGMENTS);
    let mut lines = Vec::new();

    lines.push(Line::from(Span::styled(
        "Transcript",
        Style::default().fg(theme.heading),
    )));

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

    if segments.is_empty() {
        lines.push(Line::from(Span::styled(
            "waiting for transcript...",
            Style::default().fg(theme.muted),
        )));
    }

    lines
}

fn render_notes_lines(state: &MeetingState, theme: &UiTheme) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    lines.push(Line::from(Span::styled(
        "Notes",
        Style::default().fg(theme.heading),
    )));

    if state.key_points.is_empty() && state.actions.is_empty() && state.decisions.is_empty() {
        lines.push(Line::from(Span::styled(
            "waiting for notes...",
            Style::default().fg(theme.muted),
        )));
        return lines;
    }

    for item in &state.key_points {
        lines.push(note_line(item.text.clone(), theme));
    }

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

    for item in &state.decisions {
        lines.push(note_line(item.text.clone(), theme));
    }

    lines
}

fn filtered_commands(
    phase: MeetingPhase,
    capture_paused: bool,
    filter: &str,
) -> Vec<PaletteCommand> {
    let commands = commands_for_phase(phase, capture_paused);
    if filter.trim().is_empty() {
        return commands;
    }
    commands
        .into_iter()
        .filter(|command| fuzzy_match(filter, command.label))
        .collect()
}

fn commands_for_phase(phase: MeetingPhase, capture_paused: bool) -> Vec<PaletteCommand> {
    match phase {
        MeetingPhase::Idle => vec![
            PaletteCommand {
                id: PaletteCommandId::StartMeeting,
                label: "start meeting",
                category: "meeting",
            },
            PaletteCommand {
                id: PaletteCommandId::SwitchTranscribeMode,
                label: "switch transcribe mode",
                category: "setting",
            },
            PaletteCommand {
                id: PaletteCommandId::SwitchSummarizeMode,
                label: "switch summarize mode",
                category: "setting",
            },
            PaletteCommand {
                id: PaletteCommandId::SetTranscribeModel,
                label: "set transcribe model",
                category: "setting",
            },
            PaletteCommand {
                id: PaletteCommandId::SetSummarizeModel,
                label: "set summarize model",
                category: "setting",
            },
            PaletteCommand {
                id: PaletteCommandId::EditContext,
                label: "edit context",
                category: "setting",
            },
            PaletteCommand {
                id: PaletteCommandId::BrowseSessions,
                label: "browse sessions",
                category: "view",
            },
        ],
        MeetingPhase::MeetingActive => {
            let mut commands = vec![
                PaletteCommand {
                    id: PaletteCommandId::EndMeeting,
                    label: "end meeting",
                    category: "meeting",
                },
                PaletteCommand {
                    id: PaletteCommandId::ForceSummarize,
                    label: "force summarize",
                    category: "meeting",
                },
                PaletteCommand {
                    id: PaletteCommandId::SwitchTranscribeMode,
                    label: "switch transcribe mode",
                    category: "setting",
                },
                PaletteCommand {
                    id: PaletteCommandId::SwitchSummarizeMode,
                    label: "switch summarize mode",
                    category: "setting",
                },
                PaletteCommand {
                    id: PaletteCommandId::EditContext,
                    label: "edit context",
                    category: "setting",
                },
            ];
            if capture_paused {
                commands.insert(
                    1,
                    PaletteCommand {
                        id: PaletteCommandId::ResumeCapture,
                        label: "resume capture",
                        category: "meeting",
                    },
                );
            } else {
                commands.insert(
                    1,
                    PaletteCommand {
                        id: PaletteCommandId::PauseCapture,
                        label: "pause capture",
                        category: "meeting",
                    },
                );
            }
            commands
        }
        MeetingPhase::PostMeeting => vec![
            PaletteCommand {
                id: PaletteCommandId::CopyTranscriptPath,
                label: "copy transcript path",
                category: "export",
            },
            PaletteCommand {
                id: PaletteCommandId::CopyNotesPath,
                label: "copy notes path",
                category: "export",
            },
            PaletteCommand {
                id: PaletteCommandId::CopyAudioPath,
                label: "copy audio path",
                category: "export",
            },
            PaletteCommand {
                id: PaletteCommandId::OpenSessionFolder,
                label: "open session folder",
                category: "export",
            },
            PaletteCommand {
                id: PaletteCommandId::ExportMarkdown,
                label: "export markdown",
                category: "export",
            },
            PaletteCommand {
                id: PaletteCommandId::StartNewMeeting,
                label: "start new meeting",
                category: "meeting",
            },
            PaletteCommand {
                id: PaletteCommandId::BrowseSessions,
                label: "browse sessions",
                category: "view",
            },
        ],
    }
}

fn limit_commands(
    commands: &[PaletteCommand],
    selected: usize,
    max_rows: usize,
) -> Vec<(PaletteCommand, bool)> {
    if commands.is_empty() {
        return Vec::new();
    }
    let mut selected = selected.min(commands.len().saturating_sub(1));
    let start = if selected >= max_rows {
        selected + 1 - max_rows
    } else {
        0
    };
    let end = (start + max_rows).min(commands.len());
    selected -= start;

    commands[start..end]
        .iter()
        .enumerate()
        .map(|(idx, command)| (*command, idx == selected))
        .collect()
}

fn fuzzy_match(needle: &str, haystack: &str) -> bool {
    let needle = needle.to_lowercase();
    let haystack = haystack.to_lowercase();
    let mut chars = needle.chars();
    let mut current = chars.next();

    for ch in haystack.chars() {
        if let Some(target) = current {
            if ch == target {
                current = chars.next();
            }
        } else {
            return true;
        }
    }

    current.is_none()
}

fn format_duration(duration: Duration) -> String {
    let total_secs = duration.as_secs();
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;
    if hours > 0 {
        format!("{hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes:02}:{seconds:02}")
    }
}

fn export_session_with_timeout(
    mut session: SessionHandle,
    segments: Vec<TranscriptSegment>,
    state: MeetingState,
) -> Result<(), Box<dyn std::error::Error>> {
    let (tx, rx) = channel();
    thread::spawn(move || {
        let result = session.export_on_exit(&segments, &state);
        let _ = tx.send(result);
    });

    match rx.recv_timeout(Duration::from_secs(2)) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(err)) => Err(Box::new(err)),
        Err(_) => {
            eprintln!("export timed out after 2s");
            Ok(())
        }
    }
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

fn note_line(text: String, theme: &UiTheme) -> Line<'static> {
    let bullet = "·";
    if let Some(rest) = text.strip_prefix("Me:") {
        return Line::from(vec![
            Span::styled(format!("{bullet} "), Style::default().fg(theme.neutral)),
            Span::styled("Me:", Style::default().fg(theme.me)),
            Span::styled(
                format!(" {}", rest.trim_start()),
                Style::default().fg(theme.neutral),
            ),
        ]);
    }
    if let Some(rest) = text.strip_prefix("Them:") {
        return Line::from(vec![
            Span::styled(format!("{bullet} "), Style::default().fg(theme.neutral)),
            Span::styled("Them:", Style::default().fg(theme.them)),
            Span::styled(
                format!(" {}", rest.trim_start()),
                Style::default().fg(theme.neutral),
            ),
        ]);
    }

    Line::from(Span::styled(
        format!("{bullet} {text}"),
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

fn centered_rect(width: u16, height: u16, area: Rect) -> Rect {
    let width = width.min(area.width.saturating_sub(2));
    let height = height.min(area.height.saturating_sub(2));

    let [_, middle, _] = Layout::vertical([
        Constraint::Length((area.height.saturating_sub(height)) / 2),
        Constraint::Length(height),
        Constraint::Min(0),
    ])
    .areas(area);

    let [_, center, _] = Layout::horizontal([
        Constraint::Length((area.width.saturating_sub(width)) / 2),
        Constraint::Length(width),
        Constraint::Min(0),
    ])
    .areas(middle);

    center
}

fn copy_to_clipboard(path: &Path) -> io::Result<()> {
    let output = path.to_string_lossy().to_string();
    let mut child = Command::new("pbcopy")
        .stdin(std::process::Stdio::piped())
        .spawn()?;
    if let Some(stdin) = child.stdin.as_mut() {
        stdin.write_all(output.as_bytes())?;
    }
    let _ = child.wait();
    Ok(())
}

fn open_path(path: &Path) -> io::Result<()> {
    let status = Command::new("open").arg(path).status()?;
    if status.success() {
        Ok(())
    } else {
        Err(io::Error::other("open command failed"))
    }
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
