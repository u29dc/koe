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
    - [x] `util:format`, `util:lint`, `util:test`, and `util:check` scripts are configured (`package.json`).
        - Keep the scripts in this order so a failed format step does not mask lint or test failures. When you add new checks (e.g., `cargo audit`), extend `util:check` rather than introducing new top-level scripts. The goal is one command that runs all gates locally and in CI, so changes should be mirrored in CI configuration when added.
    - [x] Husky hooks configured for pre-commit + commit-msg (`.husky/pre-commit`, `.husky/commit-msg`).
        - Ensure hooks stay minimal and only call existing scripts, so failures are consistent in CI and local dev. If you add more hooks later, prefer small wrappers that call `bun run util:check` or commitlint rather than custom logic. Verify file execute permissions are preserved after edits.
    - [x] lint-staged configured to run `bun run util:check` (`lint-staged.config.js`).
        - This should remain a full gate, not a partial staged-file lint, to avoid hidden failures. If you later decide to optimize, add a separate `util:check:fast` but keep `util:check` as the authoritative full run. Ensure any new file types are still covered by the full gate.
    - [x] commitlint configuration present (`commitlint.config.js`).
        - Keep scopes aligned with repo domains; if you add new crates or packages, update the `scope-enum`. Commitlint should stay strict to enforce consistent history. When upgrading commitlint, confirm rule names still match.
    - [x] rustfmt configuration present (`rustfmt.toml`).
        - Maintain formatting rules here rather than inline editor settings so that `cargo fmt` is deterministic. If style changes are needed, document them in the plan or a style note. Ensure formatting does not fight the default Rust edition settings.
    - [ ] rustfmt and clippy are installed (`rustup component add rustfmt clippy`) (not verified).
        - These are required for `util:format` and `util:lint` to run in a clean environment. Verify by running `rustup component list --installed` or by executing the scripts. If this project is used in CI, make sure the CI image installs these components too.
- Smoke tests:
    - [ ] `bun run util:check` completes (not executed).
        - Run this once after any major change to confirm the full gate is healthy. If it fails, address the earliest failing step first to avoid cascading errors. Capture the output in CI logs for later debugging.

Phase 1: Audio capture + chunking

- Done criteria:
    - [x] ScreenCaptureKit stream starts and system audio callbacks fire (`crates/koe-core/src/capture/sck.rs`).
        - Confirm the stream is configured with `captures_audio=true` and a content filter that actually yields audio callbacks. The output handler should be registered on `SCStreamOutputType::Audio` and should be receiving buffers. Watch for errors at `start_capture()` for permission or display availability issues.
    - [ ] Microphone callbacks fire (microphone output handler not registered; `crates/koe-core/src/capture/sck.rs:48`).
        - Register the output handler for `SCStreamOutputType::Microphone` in addition to Audio. Ensure the handler routes microphone buffers into the mic ring buffer and that the mic ring is drained in `try_recv_mic`. Validate by speaking and watching mic counters increase separately from system audio.
    - [x] Audio processor emits VAD-gated chunks with overlap (`crates/koe-core/src/process/mod.rs`, `crates/koe-core/src/process/chunker.rs`).
        - The processor should read from the capture rings, resample to 16 kHz, run VAD, and feed the chunker with the correct sample rate. Confirm the overlap behavior by inspecting chunk sizes around the target and max sizes. Maintain the 2s/4s/6s/1s policy to keep downstream ASR costs predictable.
    - [ ] Drop metrics visible in status (frame drops not wired; handler drop count not surfaced).
        - Wire the handler drop counter into `CaptureStats` so the UI can display actual frame drops. This requires either passing `CaptureStats` into the handler or pulling drop counters periodically. The status bar should show both frames dropped and chunks dropped to separate capture overload from queue backpressure.
    - [ ] Drop policy is drop-oldest (currently drop-newest on full).
        - The plan specifies freshness over completeness, which means dropping the oldest pending chunk when the queue is full. Implement this by draining one item before enqueueing or by switching to a custom ring queue. Ensure you still count drops so the status bar reflects overload.
    - [ ] RT callback avoids locks/allocations (Mutex + Vec allocations in handler).
        - The capture callback should not block; remove mutex locking in the callback path or convert it to a lock-free or try-lock path that drops when contended. Avoid heap allocation for mono conversion in the callback; preallocate buffers or move the conversion to the consumer thread. This reduces priority inversion and avoids audio glitches under load.
    - [ ] PTS alignment is accurate for drained batches.
        - `drain_ring` currently uses the most recent PTS for all samples; instead, track PTS per buffer and compute start PTS for the batch. A safe approach is to record `(pts, len)` for each callback and derive a running start offset. Validate timestamps by comparing against known audio markers.
- Smoke tests:
    - [ ] Play system audio and see chunk counters increase.
        - Use a short audio clip and verify both frame and chunk counters increment. Confirm that system audio shows up only in the system pipeline and not the mic stream. This should be repeatable without restarting the process.
    - [ ] Speak into mic and see separate stream counters increase.
        - Verify mic data flows end-to-end by checking `try_recv_mic` returns frames and the mic pipeline emits chunks. If mic remains silent, re-check output handler registration and permissions. Keep system audio silent while testing mic to avoid cross-talk.
    - [ ] Artificially pause consumer to confirm drops increment.
        - Sleep the processor loop or temporarily stop chunk consumption to force queue backpressure. Confirm chunk drops increment in stats and that the UI reflects the drop count. This test verifies both the queue policy and the visibility of drops.

Phase 2: ASR + transcript ledger + TUI

- Done criteria:
    - [x] whisper-rs provider implemented (`crates/koe-core/src/asr/whisper.rs`).
        - Verify model loading path and error handling for missing or invalid models. Ensure the sample rate matches 16 kHz and the language is set intentionally. If you change sampling strategy or thread count, update performance expectations in this plan.
    - [x] Groq provider implemented (`crates/koe-core/src/asr/groq.rs`).
        - Add explicit request timeouts and retry policy so a stalled request does not block the ASR worker forever. Ensure the Groq API key is read from the environment and errors are surfaced clearly to the UI. Keep the WAV encoding format stable; Groq expects WAV audio with a valid header.
    - [x] Transcript ledger implemented (`crates/koe-core/src/transcript.rs`).
        - Keep the overlap window in sync with chunker overlap. If you adjust chunk overlap, adjust `OVERLAP_WINDOW_MS` to match. Maintain tests for dedupe behavior because this directly affects transcript quality.
    - [ ] ASR worker consumes chunks and emits transcript events to UI.
        - Add a worker thread or task that reads from `chunk_rx`, calls the selected provider, and emits a `TranscriptEvent` (or equivalent) on a channel to the UI. Keep this loop resilient: skip empty segments, keep running on transient failures, and surface errors as events. This is the main wiring step that makes the app functional.
    - [ ] CLI flags wire into ASR provider creation (`crates/koe-cli/src/main.rs`).
        - Use `--asr` and `--model-trn` to call `create_asr` and pass the provider into the runtime. If the provider fails to initialize, exit with a clear error that points to missing model or API key. Ensure defaults align with the plan.
    - [ ] Transcript renders in TUI (currently placeholder text).
        - Keep a local transcript buffer in the TUI state and update it when new events arrive. Render a limited window to keep UI responsive and avoid huge redraws. Make sure text wrapping is stable and that the UI does not flicker on updates.
    - [ ] Speaker labeling (mic = "Me", system = "Them").
        - Map `AudioChunk.source` to `TranscriptSegment.speaker` at the time of segment creation. If a mixed stream is used, set speaker to `None` or `Unknown`. This is also the hook to later integrate diarization if needed.
    - [ ] Provider switching without restart (command surface missing).
        - Define a command channel from UI to core and support hotkeys to switch providers. Recreate the provider in the ASR worker and emit a status event to update the UI. Ensure in-flight chunks are handled safely during switches.
    - [ ] Status bar shows ASR lag and active provider.
        - Track time spent per chunk in the ASR worker and emit rolling latency metrics. Display the active provider and last latency in the status bar alongside capture stats. This is essential for troubleshooting local vs cloud performance.
    - [ ] Mutable window corrections only affect last 15 s (not implemented).
        - Implement a mutable transcript window and finalize segments that are older than the window. When new ASR results overlap finalized segments, do not alter them. Keep this window consistent with the chunk overlap to prevent duplicate text.
- Smoke tests:
    - [ ] Short utterance appears within 4 s locally, faster on cloud.
        - Use a stopwatch and a single spoken phrase. Confirm the local model is within target latency and cloud is faster. If latency is too high, reduce chunk size or increase ASR thread resources.
    - [ ] Overlap does not duplicate text.
        - Speak continuous speech over multiple chunks and verify deduplication in the transcript ledger. If duplicates appear, adjust similarity threshold or merge policy. Document any changes to the similarity heuristic.
    - [ ] Switch providers without crash.
        - Switch providers repeatedly during active audio capture. Ensure the worker restarts cleanly and the UI status updates. This test should not leak threads or leave the provider in a half-initialized state.

Phase 3: Notes engine (patch-only)

- Done criteria:
    - [ ] Ollama provider emits NotesPatch.
        - Implement a summarizer provider that calls Ollama and parses a patch-only response. Keep the prompt and response schema stable to avoid parse failures. Treat network errors as non-fatal and try again on the next cycle.
    - [ ] OpenRouter provider emits NotesPatch.
        - Add a second provider with similar patch parsing, but with OpenRouter-specific authentication. Validate response shape and ensure timeouts and retries are consistent with Groq. The provider should be swappable at runtime like ASR.
    - [ ] Notes pane updates without full rewrites.
        - Apply patches to a persistent `MeetingState` so only incremental updates are rendered. Avoid full re-render of notes to keep the UI stable and reduce flicker. Ensure patches only add or update, never delete silently.
    - [ ] Stable IDs persist across updates.
        - IDs should be generated by the summarizer and reused across updates, not regenerated on each patch. This allows UI selection and references to remain stable. Validate by running multiple summaries and confirming IDs remain unchanged for the same item.
    - [ ] Summarizer uses finalized segments only.
        - Feed only `finalized` transcript segments into the summarizer to avoid churn. If you want provisional notes, mark them clearly and separate them from stable notes. This protects against edits to earlier transcript text.
- Smoke tests:
    - [ ] Decision phrasing triggers new decision item.
        - Use test phrases that match the summarizer prompt and confirm a new decision appears. Check that it references the correct transcript evidence IDs. Verify that repeated phrases do not create duplicates.
    - [ ] Action phrasing triggers new action item.
        - Use phrases with clear owner and due date language. Validate parsing of optional fields like owner and due date. Ensure the item shows up in the correct notes section.
    - [ ] No duplicates after multiple cycles.
        - Run the summarizer multiple times without new transcript segments. The notes list should remain stable with no new items. If duplicates appear, adjust prompt or patch logic to enforce idempotency.

Phase 4: Latency comparison + polish

- Done criteria:
    - [ ] Status bar shows ASR lag, drops, and provider.
        - Extend the status bar to include ASR latency, active provider, and capture drop metrics. Keep the layout fixed width to avoid jitter as values change. This should be updated from the same event stream as transcript updates.
    - [ ] Export transcript/notes on quit.
        - On shutdown, persist the transcript and notes to files (e.g., `transcript.md`, `notes.json`). Ensure the export path is configurable and errors are surfaced cleanly. Do not block UI shutdown indefinitely; use a bounded export timeout.
- Smoke tests:
    - [ ] Compare local vs cloud latency over a 3-minute session.
        - Run a controlled session with both providers and record average latency. Use the status bar metrics to compare and document the result. If results vary, capture network conditions for context.
    - [ ] Export produces valid transcript.md and notes.json.
        - Validate the output format with a simple parser or quick manual check. Ensure files include metadata like timestamps or session ID if desired. Confirm the export does not include partial or duplicated entries.

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
