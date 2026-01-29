# KOE

## Architecture

Ideal maintainable architecture (core-first, UI-agnostic):

- Workspace layout: `crates/koe-core` (engine) + `crates/koe-cli` (current TUI/CLI shell).
- `koe-core` exposes a stable event stream and command API with zero UI dependencies.
- `koe-cli` is a thin adapter that renders core events to a TUI and forwards user commands back to the core.
- Future macOS Swift UI can replace `koe-cli` by embedding or IPC to `koe-core` without touching engine logic.
- All providers (ASR, summarizer) and capture backends live behind traits in `koe-core`.

End-to-end data flow:

```
ScreenCaptureKit (system audio + mic)
  -> RT callback (copy f32 into SPSC ring buffers, no alloc/locks)
    -> Audio processor (align, mix/keep separate, resample 48k -> 16k)
      -> VAD + chunker (speech-gated, overlap) -> sync_channel<AudioChunk>
        -> ASR worker (local whisper-rs or Groq) -> TranscriptSegment[]
          -> Transcript ledger (mutable window + finalize) -> Event bus
            -> Notes engine (patch-only, Ollama/OpenRouter) -> NotesPatch
              -> TUI (transcript + notes + status)
```

Component responsibilities:

- ScreenCaptureKit adapter: enumerate content, configure stream, deliver audio buffers.
- RT callback: copy samples into ring buffer and return immediately.
- Audio processor: timestamp alignment, optional mix, resample to 16 kHz mono.
- VAD + chunker: detect speech and emit bounded, overlapped chunks.
- ASR provider: transcribe chunks into timestamped segments.
- Transcript ledger: merge overlaps, keep last N seconds mutable, finalize older segments.
- Notes engine: produce patch ops and apply to stable meeting state.
- TUI: render transcript/notes, show lag/drops/provider status, handle hotkeys.

Threading model and channel types:

- ScreenCaptureKit dispatch queue -> SPSC ring buffer (`rtrb::RingBuffer<f32>`) per stream.
- Audio processor thread drains ring buffers and emits chunks.
- Chunk queue: `std::sync::mpsc::sync_channel<AudioChunk>` (cap 4, drop-oldest).
- ASR worker thread reads chunks, outputs segments on `std::sync::mpsc::channel<TranscriptEvent>`.
- Notes thread ticks on interval/trigger; outputs `NotesPatch` on `std::sync::mpsc::channel<NotesPatch>`.
- UI thread (ratatui) consumes events from a single merged channel.

Event and command surface between core and UI shells:

- `CoreEvent`: transcript updates, finalized segment IDs, notes patches, provider status, stats, errors.
- `CoreCommand`: start/stop, provider switch, force summarize, export, pause/resume.
- Transport: in-process channels for `koe-cli`; NDJSON over stdout or a Unix socket for future Swift UI.

## Technical Decisions

Platform scope:

- Target macOS 15+ only (latest versions), no legacy fallback paths.
- Use latest ScreenCaptureKit APIs and features.

Audio capture (ScreenCaptureKit):

- Use ScreenCaptureKit audio capture with a normal content filter and `captures_audio=true`.
- Default to audio-only output (no screen frames attached).
- Provide a fallback flag to attach a tiny video output if a user’s system requires it for audio callbacks.
- Microphone capture via ScreenCaptureKit only (macOS 15+), no `cpal`.
- Permissions: Screen Recording (system audio) and Microphone; both typically require process restart after grant.

Chunking and VAD:

- Use Silero VAD (voice_activity_detector) for higher quality speech detection.
- Frame size: 512 samples at 16 kHz (32 ms).
- Threshold: 0.5 probability; min speech 200 ms; hangover 300 ms.
- Chunking: target 4.0 s windows, 1.0 s overlap, min 2.0 s, max 6.0 s.
- Rationale: balances latency, accuracy at boundaries, and cost per ASR call.

Backpressure:

- Ring buffer capacity: 10 s per stream (system and mic).
- Chunk queue capacity: 4; drop oldest pending chunk on overflow.
- Notes queue capacity: 1; skip cycle if busy.
- Policy favors freshness and UI responsiveness.

Notes stability:

- Patch-based updates with stable IDs (add/update ops only).
- Summarizer uses finalized segments only; notes referencing mutable evidence are marked tentative.
- Notes update cadence: every 10 s or keyword triggers (decision/action phrases).

Providers (day one):

- ASR: local whisper-rs (Metal) and Groq cloud (Whisper large-v3-turbo).
- Summarizer: local Ollama and OpenRouter cloud.
- CLI flags switch providers without restart; status bar shows active providers and latency stats.

Speaker labeling:

- Use stream-based attribution: mic stream -> "Me", system stream -> "Them".
- This is the most reliable low-latency option without diarization overhead.
- Mixed stream mode is optional and labeled "Unknown".

## Interfaces

```rust
// Rust 2024 edition idioms (async in traits is stable).

pub struct AudioFrame {
    pub pts_ns: i128,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub samples_f32: Vec<f32>,
}

pub enum AudioSource {
    System,
    Microphone,
    Mixed,
}

pub struct AudioChunk {
    pub source: AudioSource,
    pub start_pts_ns: i128,
    pub sample_rate_hz: u32, // 16_000
    pub pcm_mono_f32: Vec<f32>,
}

pub struct TranscriptSegment {
    pub id: u64,
    pub start_ms: i64,
    pub end_ms: i64,
    pub speaker: Option<String>,
    pub text: String,
    pub finalized: bool,
}

pub struct MeetingState {
    pub key_points: Vec<NoteItem>,
    pub actions: Vec<ActionItem>,
    pub decisions: Vec<NoteItem>,
}

pub struct NoteItem {
    pub id: String,
    pub text: String,
    pub evidence: Vec<u64>,
}

pub struct ActionItem {
    pub id: String,
    pub text: String,
    pub owner: Option<String>,
    pub due: Option<String>,
    pub evidence: Vec<u64>,
}

pub enum NotesOp {
    AddKeyPoint { id: String, text: String, evidence: Vec<u64> },
    AddAction { id: String, text: String, owner: Option<String>, due: Option<String>, evidence: Vec<u64> },
    AddDecision { id: String, text: String, evidence: Vec<u64> },
    UpdateAction { id: String, owner: Option<String>, due: Option<String> },
}

pub struct NotesPatch {
    pub ops: Vec<NotesOp>,
}

pub enum SummarizerEvent {
    DraftToken(String),
    PatchReady(NotesPatch),
}

pub trait AudioCapture: Send {
    fn start(&mut self) -> Result<(), CaptureError>;
    fn stop(&mut self);
    fn try_recv_system(&mut self) -> Option<AudioFrame>;
    fn try_recv_mic(&mut self) -> Option<AudioFrame>;
}

pub trait AsrProvider: Send {
    fn name(&self) -> &'static str;
    fn transcribe(&mut self, chunk: &AudioChunk) -> Result<Vec<TranscriptSegment>, AsrError>;
}

pub trait SummarizerProvider: Send {
    fn name(&self) -> &'static str;
    fn summarize(
        &mut self,
        recent_segments: &[TranscriptSegment],
        state: &MeetingState,
        on_event: &mut dyn FnMut(SummarizerEvent),
    ) -> Result<(), SummarizerError>;
}
```

## Dependencies

Minimal Cargo.toml (latest versions, MVP):

```toml
[package]
name = "koe"
version = "0.1.0"
edition = "2024"

[dependencies]
screencapturekit = { version = "1.5.0", features = ["macos_26_0"] }
rtrb = "0.3.2"
rubato = "0.16.2"
voice_activity_detector = "0.2.1"
whisper-rs = { version = "0.15.1", features = ["metal"] }
ratatui = "0.30.0"
crossterm = "0.29.0"
ureq = { version = "3.1.4", features = ["json"] }
serde = { version = "1.0.228", features = ["derive"] }
serde_json = "1.0.149"
thiserror = "2.0.18"
clap = { version = "4.5.56", features = ["derive"] }

[target.'cfg(target_os = "macos")'.dependencies]
core-foundation = "0.10.1"
```

Justifications:

- screencapturekit: ScreenCaptureKit bindings for system audio + mic (macOS 15+).
- rtrb: lock-free SPSC ring buffer for RT-safe audio callbacks.
- rubato: high-quality 48k -> 16k resampling.
- voice_activity_detector: Silero VAD for accurate speech detection (ONNX-based).
- whisper-rs: local Whisper inference with Metal acceleration.
- ratatui/crossterm: TUI rendering and input handling.
- ureq: simple blocking HTTP for Groq/OpenRouter and Ollama NDJSON/SSE.
- serde/serde_json: patch ops and config serialization.
- thiserror: clean error enums.
- clap: CLI flags for provider selection and tuning.
- core-foundation: macOS FFI helpers used by ScreenCaptureKit.

## Build Phases

Phase 0: Quality gate wiring

- Done criteria:
    - `bun run util:check` exists and runs format, lint, and tests.
    - rustfmt and clippy are installed (`rustup component add rustfmt clippy`).
- Smoke tests:
    - `bun run util:check` completes (even if no tests exist yet).

Phase 1: Audio capture + chunking

- Done criteria:
    - ScreenCaptureKit stream starts, audio callbacks firing for system and mic.
    - Audio processor emits VAD-gated chunks with overlap.
    - Ring buffer and chunk queue drop metrics visible in status.
- Smoke tests:
    - Play system audio and see chunk counters increase.
    - Speak into mic and see separate stream counters increase.
    - Artificially pause consumer to confirm drops increment.

Phase 2: ASR + transcript ledger + TUI

- Done criteria:
    - whisper-rs transcribes chunks and renders in transcript pane.
    - Groq cloud path works and can be switched at runtime.
    - Mutable window corrections only affect last 15 s.
- Smoke tests:
    - Short utterance appears within 4 s locally, faster on cloud.
    - Overlap does not duplicate text.
    - Switch providers without crash.

Phase 3: Notes engine (patch-only)

- Done criteria:
    - Ollama and OpenRouter paths both emit NotesPatch.
    - Notes pane updates without full rewrites.
    - Stable IDs persist across updates.
- Smoke tests:
    - Decision phrasing triggers new decision item.
    - Action phrasing triggers new action item.
    - No duplicates after multiple cycles.

Phase 4: Latency comparison + polish

- Done criteria:
    - Status bar shows ASR lag, drops, and provider.
    - Export transcript/notes on quit.
- Smoke tests:
    - Compare local vs cloud latency over a 3-minute session.
    - Export produces valid transcript.md and notes.json.

## Testing Strategy

- Unit tests:
    - VAD state machine and chunk boundary logic.
    - Transcript ledger overlap merge and finalize logic.
    - NotesPatch apply and stable ID handling.
- Integration tests:
    - Feed canned WAV chunks through chunker -> ASR mock -> ledger.
    - Summarizer mock returns patch; state applies correctly.
- Manual QA (required):
    - Permissions prompts, restart behavior, and capture correctness.
    - Long-running session (30 min) for drop/lag stability.

## Quality Gates and Commands

- `bun run util:check` runs: `cargo fmt --all`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo test --all`.
- Run `bun run util:check` after each major phase and before every commit.

## Resolved Decisions

- macOS target: 15+ only (latest versions).
- VAD: Silero VAD for quality, with tuned parameters above.
- Providers: local + cloud for ASR and summarization from day one.
- Speaker labeling: stream-based (mic -> Me, system -> Them).
