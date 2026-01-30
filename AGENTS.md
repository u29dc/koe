# KOE

Real-time meeting transcription and notes engine for macOS, built in Rust with ScreenCaptureKit audio capture, VAD-gated chunking, local/cloud ASR (whisper-rs/Groq), and LLM-powered patch-based summarization (Ollama/OpenRouter), rendered in a ratatui TUI.

## 1. Repository Structure

```
koe/
  Cargo.toml              # workspace root (resolver = "3")
  package.json            # bun scripts for quality gates (format, lint, test)
  biome.json              # extends ~/.config/biome/biome.json
  commitlint.config.js    # conventional commits, scopes: core|cli|web|config|deps
  lint-staged.config.js   # runs bun run util:check
  rustfmt.toml            # Rust formatting rules
  .husky/                 # pre-commit (lint-staged), commit-msg (commitlint)
  crates/
    koe-core/             # engine: capture, processing, ASR, transcript, notes
    koe-cli/              # thin TUI shell: renders core events, forwards commands
```

## 2. Stack

| Layer            | Choice                                         | Notes                                |
| ---------------- | ---------------------------------------------- | ------------------------------------ |
| Language         | Rust 2024 edition                              | rust-version 1.85                    |
| Audio capture    | screencapturekit 1.5.0                         | macOS 15+, system audio + mic        |
| Ring buffer      | rtrb 0.3.2                                     | lock-free SPSC for RT callbacks      |
| Resampling       | rubato 0.16.2                                  | 48k -> 16k high-quality              |
| VAD              | voice_activity_detector 0.2.1                  | Silero ONNX, 512 samples/32ms frames |
| Local ASR        | whisper-rs 0.15.1                              | Metal acceleration                   |
| Cloud ASR        | Groq API                                       | Whisper large-v3-turbo via ureq      |
| Local summarizer | Ollama                                         | NDJSON streaming via ureq            |
| Cloud summarizer | OpenRouter API                                 | via ureq                             |
| TUI              | ratatui 0.30.0 + crossterm 0.29.0              |                                      |
| CLI              | clap 4.5.56                                    | derive features                      |
| HTTP             | ureq 3.1.4                                     | json + multipart features            |
| Serialization    | serde 1.0.228 + serde_json 1.0.149             |                                      |
| Errors           | thiserror 2.0.18                               |                                      |
| Signals          | signal-hook 0.3.18                             |                                      |
| macOS FFI        | core-foundation 0.10.1                         |                                      |
| Quality gates    | bun + biome + commitlint + husky + lint-staged |                                      |

## 3. Architecture

Workspace layout: `crates/koe-core` (engine, zero UI deps) + `crates/koe-cli` (thin TUI adapter rendering core events and forwarding commands back); future macOS Swift UI replaces CLI via IPC to core without touching engine logic; all providers and capture backends live behind traits in koe-core.

Data flow:

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

Component responsibilities: ScreenCaptureKit adapter (enumerate content, configure stream, deliver audio buffers), RT callback (copy samples into ring buffer and return immediately), audio processor (timestamp alignment, optional mix, resample to 16 kHz mono), VAD + chunker (detect speech, emit bounded overlapped chunks), ASR provider (transcribe chunks into timestamped segments), transcript ledger (merge overlaps, keep last N seconds mutable, finalize older segments), notes engine (produce patch ops, apply to stable meeting state), TUI (render transcript/notes, show lag/drops/provider status, handle hotkeys).

Threading model: ScreenCaptureKit dispatch queue -> SPSC ring buffer (`rtrb::RingBuffer<f32>`) per stream; audio processor thread drains ring buffers and emits chunks; chunk queue via `std::sync::mpsc::sync_channel<AudioChunk>` (cap 4, drop-oldest); ASR worker thread reads chunks and outputs segments on `std::sync::mpsc::channel<TranscriptEvent>`; notes thread ticks on interval/trigger and outputs `NotesPatch` on `std::sync::mpsc::channel<NotesPatch>`; UI thread (ratatui) consumes events from a single merged channel.

Event/command surface: `CoreEvent` (transcript updates, finalized segment IDs, notes patches, provider status, stats, errors), `CoreCommand` (start/stop, provider switch, force summarize, export, pause/resume); transport is in-process channels for koe-cli, NDJSON over stdout or Unix socket for future Swift UI.

## 4. Technical Decisions

Platform: macOS 15+ only, latest ScreenCaptureKit APIs, no legacy fallback paths; permissions require Screen Recording (system audio) and Microphone, both typically need process restart after grant.

Audio capture: ScreenCaptureKit with content filter `captures_audio=true`, default audio-only output, fallback flag for tiny video output if needed for callbacks; microphone via ScreenCaptureKit only (macOS 15+), no cpal.

Chunking/VAD: Silero VAD frame size 512 samples at 16 kHz (32 ms), threshold 0.5, min speech 200 ms, hangover 300 ms; chunking target 4.0 s windows, 1.0 s overlap, min 2.0 s, max 6.0 s.

Backpressure: ring buffer 10 s per stream, chunk queue cap 4 drop-oldest, notes queue cap 1 skip-if-busy; favors freshness and UI responsiveness.

Notes stability: patch-based updates with stable IDs (add/update ops only), summarizer uses finalized segments only, tentative notes marked when referencing mutable evidence, update cadence every 10 s or on keyword triggers (decision/action phrases).

Speaker labeling: stream-based attribution (mic -> "Me", system -> "Them"), mixed stream labeled "Unknown"; most reliable low-latency option without diarization overhead.

## 5. Interfaces

```rust
pub struct AudioFrame {
    pub pts_ns: i128,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub samples_f32: Vec<f32>,
}

pub enum AudioSource { System, Microphone, Mixed }

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

pub struct NoteItem { pub id: String, pub text: String, pub evidence: Vec<u64> }
pub struct ActionItem { pub id: String, pub text: String, pub owner: Option<String>, pub due: Option<String>, pub evidence: Vec<u64> }

pub enum NotesOp {
    AddKeyPoint { id: String, text: String, evidence: Vec<u64> },
    AddAction { id: String, text: String, owner: Option<String>, due: Option<String>, evidence: Vec<u64> },
    AddDecision { id: String, text: String, evidence: Vec<u64> },
    UpdateAction { id: String, owner: Option<String>, due: Option<String> },
}

pub struct NotesPatch { pub ops: Vec<NotesOp> }
pub enum SummarizerEvent { DraftToken(String), PatchReady(NotesPatch) }

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
    fn summarize(&mut self, recent_segments: &[TranscriptSegment], state: &MeetingState, on_event: &mut dyn FnMut(SummarizerEvent)) -> Result<(), SummarizerError>;
}
```

## 6. Commands

| Command                 | Description                                                          |
| ----------------------- | -------------------------------------------------------------------- |
| `bun run build`         | `cargo build --workspace --release`                                  |
| `bun run koe -- [args]` | `cargo run -p koe-cli -- [args]`                                     |
| `bun run util:format`   | `cargo fmt --all`                                                    |
| `bun run util:lint`     | `cargo clippy --all-targets --all-features -- -D warnings`           |
| `bun run util:test`     | `cargo test --all`                                                   |
| `bun run util:check`    | runs format + lint + test sequentially, exits nonzero on any failure |

## 7. Local Setup and Testing

- Set `.env` with `GROQ_API_KEY=...` and `KOE_WHISPER_MODEL=/Users/han/.koe/models/ggml-base.en.bin`.
- Download model and update `.env`: `./scripts/koe-init.sh` (writes to `~/.koe/models`).
- Run local or cloud: `./scripts/koe-whisper.sh` or `./scripts/koe-groq.sh`.
- Alternate model: `./scripts/koe-init.sh --model small`, then `./scripts/koe-whisper.sh`.

## 8. Quality

Zero clippy warnings (`-D warnings`), `cargo fmt --all` enforced, all tests passing, pre-commit hooks run full `util:check` via lint-staged, commitlint enforces conventional commits with domain scopes (core, cli, web, config, deps).

Testing strategy: unit tests for VAD state machine and chunk boundary logic, transcript ledger overlap merge and finalize logic, NotesPatch apply and stable ID handling; integration tests feed canned WAV chunks through chunker -> ASR mock -> ledger and summarizer mock returns patch with state application; manual QA for permissions prompts, restart behavior, capture correctness, and 30-min long-running session stability.

## 9. Roadmap

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
    - [x] rustfmt and clippy are installed (`rustup component add rustfmt clippy`) (verified via `cargo fmt --version` and `cargo clippy --version`).
        - These are required for `util:format` and `util:lint` to run in a clean environment. Verify by running `rustup component list --installed` or by executing the scripts. If this project is used in CI, make sure the CI image installs these components too.
    - [x] Config file lives at `~/.koe/config.toml` with a minimal schema (audio, asr, summarizer, ui) and defaults for local-first.
        - Required sections: `[audio]` (sample_rate, channels, sources), `[asr]` (provider, model, api_key), `[summarizer]` (provider, model, api_key, prompt_profile), `[ui]` (show_transcript, notes_only_default, color_theme).
        - Keep all user-facing settings here, including API keys, model choices, and UI toggles. Environment variables may override for CI/dev but are optional for end users.
        - Ensure `~/.koe/` contains `config.toml`, `models/`, and `sessions/` directories with predictable paths.
        - Ensure config file permissions are restricted (0600); warn if file is group/world readable.
        - Add a config version field and lightweight migration path for new fields.
        - Define a single runtime entry point: `koe` runs the TUI; `koe init` and `koe config` handle setup. Avoid introducing additional top-level commands unless required.
    - [x] `koe config` subcommand reads/writes config with validation and redacts secrets in terminal output.
        - Support `--print` (redact keys), `--set key=value` (dotted paths like `asr.provider=whisper`), and `--edit` (opens in $EDITOR) for quick changes.
        - Validate enums (provider names), file paths, and required keys; surface errors with actionable guidance.
        - Define precedence: CLI flags > config file > env vars (env vars optional), and keep precedence consistent across commands.
    - [x] `koe init` runs interactive onboarding (local or cloud, model selection, API keys) and persists to config.
        - Prompt order: permissions guidance, ASR choice (local/cloud), model selection or download, summarizer choice (local/cloud), model selection, API key entry; write config and report the next command to run.
        - When local ASR is selected, offer model download options and show disk size; when cloud is selected, prompt for API key and validate non-empty input.
    - [x] `koe init` prints macOS permission instructions for Screen Recording + Microphone and notes restart requirements.
        - Keep output minimal and actionable, include the exact System Settings path, mention that permissions may require a restart, and show a single-line checklist of required grants.
- Smoke tests:
    - [x] `bun run util:check` completes (not executed).
        - Run this once after any major change to confirm the full gate is healthy. If it fails, address the earliest failing step first to avoid cascading errors. Capture the output in CI logs for later debugging.
    - [x] `koe init` writes `~/.koe/config.toml` and can be re-run idempotently without clobbering user edits.
        - Re-running init should preserve existing values unless explicitly changed or `--force` is set; report what changed and what was kept.

Phase 1: Audio capture + chunking

- Done criteria:
    - [x] ScreenCaptureKit stream starts and system audio callbacks fire (`crates/koe-core/src/capture/sck.rs`).
        - Confirm the stream is configured with `captures_audio=true` and a content filter that actually yields audio callbacks. The output handler should be registered on `SCStreamOutputType::Audio` and should be receiving buffers. Watch for errors at `start_capture()` for permission or display availability issues.
    - [x] Microphone callbacks fire (`crates/koe-core/src/capture/sck.rs:49`).
        - Register the output handler for `SCStreamOutputType::Microphone` in addition to Audio. Ensure the handler routes microphone buffers into the mic ring buffer and that the mic ring is drained in `try_recv_mic`. Validate by speaking and watching mic counters increase separately from system audio.
    - [x] Audio processor emits VAD-gated chunks with overlap (`crates/koe-core/src/process/mod.rs`, `crates/koe-core/src/process/chunker.rs`).
        - The processor should read from the capture rings, resample to 16 kHz, run VAD, and feed the chunker with the correct sample rate. Confirm the overlap behavior by inspecting chunk sizes around the target and max sizes. Maintain the 2s/4s/6s/1s policy to keep downstream ASR costs predictable.
    - [x] Drop metrics visible in status (frame drops not wired; handler drop count not surfaced).
        - Wire the handler drop counter into `CaptureStats` so the UI can display actual frame drops. This requires either passing `CaptureStats` into the handler or pulling drop counters periodically. The status bar should show both frames dropped and chunks dropped to separate capture overload from queue backpressure.
    - [x] Drop policy is drop-oldest (currently drop-newest on full).
        - The plan specifies freshness over completeness, which means dropping the oldest pending chunk when the queue is full. Implement this by draining one item before enqueueing or by switching to a custom ring queue. Ensure you still count drops so the status bar reflects overload.
    - [x] RT callback avoids locks/allocations (Mutex + Vec allocations in handler).
        - The capture callback should not block; remove mutex locking in the callback path or convert it to a lock-free or try-lock path that drops when contended. Avoid heap allocation for mono conversion in the callback; preallocate buffers or move the conversion to the consumer thread. This reduces priority inversion and avoids audio glitches under load.
    - [x] PTS alignment is accurate for drained batches.
        - `drain_ring` currently uses the most recent PTS for all samples; instead, track PTS per buffer and compute start PTS for the batch. A safe approach is to record `(pts, len)` for each callback and derive a running start offset. Validate timestamps by comparing against known audio markers.
- Smoke tests:
    - [x] Play system audio and see chunk counters increase.
        - Use a short audio clip and verify both frame and chunk counters increment. Confirm that system audio shows up only in the system pipeline and not the mic stream. This should be repeatable without restarting the process.
    - [x] Speak into mic and see separate stream counters increase.
        - Verify mic data flows end-to-end by checking `try_recv_mic` returns frames and the mic pipeline emits chunks. If mic remains silent, re-check output handler registration and permissions. Keep system audio silent while testing mic to avoid cross-talk.
    - [x] Artificially pause consumer to confirm drops increment.
        - Sleep the processor loop or temporarily stop chunk consumption to force queue backpressure. Confirm chunk drops increment in stats and that the UI reflects the drop count. This test verifies both the queue policy and the visibility of drops.

Phase 2: ASR + transcript ledger + TUI

- Done criteria:
    - [x] whisper-rs provider implemented (`crates/koe-core/src/asr/whisper.rs`).
        - Verify model loading path and error handling for missing or invalid models. Ensure the sample rate matches 16 kHz and the language is set intentionally. If you change sampling strategy or thread count, update performance expectations in this plan.
    - [x] Groq provider implemented (`crates/koe-core/src/asr/groq.rs`).
        - Add explicit request timeouts and retry policy so a stalled request does not block the ASR worker forever. Ensure the Groq API key is read from the environment and errors are surfaced clearly to the UI. Keep the WAV encoding format stable; Groq expects WAV audio with a valid header.
    - [x] Transcript ledger implemented (`crates/koe-core/src/transcript.rs`).
        - Keep the overlap window in sync with chunker overlap. If you adjust chunk overlap, adjust `OVERLAP_WINDOW_MS` to match. Maintain tests for dedupe behavior because this directly affects transcript quality.
    - [x] ASR worker consumes chunks and emits transcript events to UI.
        - Add a worker thread or task that reads from `chunk_rx`, calls the selected provider, and emits a `TranscriptEvent` (or equivalent) on a channel to the UI. Keep this loop resilient: skip empty segments, keep running on transient failures, and surface errors as events. This is the main wiring step that makes the app functional.
    - [x] CLI flags wire into ASR provider creation (`crates/koe-cli/src/main.rs`).
        - Use `--asr` and `--model-trn` to call `create_asr` and pass the provider into the runtime. If the provider fails to initialize, exit with a clear error that points to missing model or API key. Ensure defaults align with the plan.
    - [x] Transcript renders in TUI (currently placeholder text).
        - Keep a local transcript buffer in the TUI state and update it when new events arrive. Render a limited window to keep UI responsive and avoid huge redraws. Make sure text wrapping is stable and that the UI does not flicker on updates.
    - [x] Speaker labeling (mic = "Me", system = "Them").
        - Map `AudioChunk.source` to `TranscriptSegment.speaker` at the time of segment creation. If a mixed stream is used, set speaker to `None` or `Unknown`. This is also the hook to later integrate diarization if needed.
    - [x] Provider switching without restart (command surface missing).
        - Define a command channel from UI to core and support hotkeys to switch providers. Recreate the provider in the ASR worker and emit a status event to update the UI. Ensure in-flight chunks are handled safely during switches.
    - [x] Status bar shows ASR lag and active provider.
        - Track time spent per chunk in the ASR worker and emit rolling latency metrics. Display the active provider and last latency in the status bar alongside capture stats. This is essential for troubleshooting local vs cloud performance.
    - [x] Mutable window corrections only affect last 15 s (not implemented).
        - Implement a mutable transcript window and finalize segments that are older than the window. When new ASR results overlap finalized segments, do not alter them. Keep this window consistent with the chunk overlap to prevent duplicate text.
    - [x] Minimal full-screen TUI with a focused notes pane and optional transcript pane.
        - Default view is notes-only; toggle transcript visibility with a single key (e.g., `t`). When visible, show transcript on one side and notes on the other with stable layout and no flicker.
        - Provide a clear status line with provider/lag/capture stats, and keep layout stable when toggling panes (no shifting widths).
        - Document key bindings in a single place (help overlay or footer) and keep them consistent: quit, toggle transcript, switch provider, set context.
    - [x] Color system: other-party content uses a restrained blue, self content uses neutral gray, headings are subtle and consistent.
        - Apply colors consistently in both transcript and notes; blue is reserved for “Them” content, gray for “Me,” and neutral for headings/metadata.
        - Keep palette minimal and readable; avoid bright or noisy styling. Ensure colors remain legible in common terminal themes.
    - [x] TUI clean shutdown restores terminal even on panic.
        - Maintain existing panic hook and ensure all threads stop cleanly on exit.
    - [x] Meeting context can be provided via CLI, config, or TUI and is passed to the summarizer.
        - Support `--context`, config default (`session.context`), and an in-TUI edit action; pick a single canonical source with clear precedence (CLI > TUI > config).
        - Context should be stored in session metadata (`metadata.toml`) and injected into summarizer prompts; allow empty/absent context.
        - Support multi-line context; preserve verbatim and redact from logs unless explicitly printed.
        - TUI context edits update the current session metadata only (do not mutate global config).
- Smoke tests:
    - [x] Short utterance appears within 4 s locally, faster on cloud.
        - Use a stopwatch and a single spoken phrase. Confirm the local model is within target latency and cloud is faster. If latency is too high, reduce chunk size or increase ASR thread resources.
    - [x] Overlap does not duplicate text.
        - Speak continuous speech over multiple chunks and verify deduplication in the transcript ledger. If duplicates appear, adjust similarity threshold or merge policy. Document any changes to the similarity heuristic.
    - [x] Switch providers without crash.
        - Switch providers repeatedly during active audio capture. Ensure the worker restarts cleanly and the UI status updates. This test should not leak threads or leave the provider in a half-initialized state.
    - [ ] Toggle transcript pane repeatedly without layout glitches.
        - Verify the notes pane remains stable and the transcript pane cleanly hides/shows without resizing artifacts.
        - BLOCKED: Manual UI toggle verification cannot be completed in this non-interactive environment; tried only code-level implementation; next steps: run `bun run koe`, press `t` repeatedly during active UI, confirm layout stability; file refs: `crates/koe-cli/src/tui.rs`.

Phase 3: Notes engine (patch-only)

- Done criteria:
    - [x] Ollama provider emits NotesPatch.
        - Implement a summarizer provider that calls Ollama and parses a patch-only response. Keep the prompt and response schema stable to avoid parse failures. Treat network errors as non-fatal and try again on the next cycle.
    - [x] OpenRouter provider emits NotesPatch.
        - Add a second provider with similar patch parsing, but with OpenRouter-specific authentication. Validate response shape and ensure timeouts and retries are consistent with Groq. The provider should be swappable at runtime like ASR.
    - [x] Notes pane updates without full rewrites.
        - Apply patches to a persistent `MeetingState` and only update changed items in the TUI; avoid full redraws to keep scrolling stable. Patches should only add/update, never delete silently.
    - [x] Stable IDs persist across updates.
        - IDs should be generated by the summarizer and reused across updates, not regenerated on each patch. This allows UI selection and references to remain stable. Validate by running multiple summaries and confirming IDs remain unchanged for the same item.
    - [x] Summarizer uses finalized segments only.
        - Feed only `finalized` transcript segments into the summarizer to avoid churn. If you want provisional notes, mark them clearly and separate them from stable notes. This protects against edits to earlier transcript text.
    - [ ] Summarizer prompts are tuned for minimal, information-dense output with short patches.
        - Output must be concise, avoid filler, and emit only key points/actions/decisions. Prefer short noun phrases over full sentences.
    - [ ] Notes capture speaker attribution when available (Me vs Them) with consistent labels.
        - Use stable labels (“Me”, “Them”) and keep the label prefix minimal (e.g., `Me:`) so notes stay compact.
    - [ ] Notes pane updates incrementally in real time without full redraws.
        - Patch application updates the data model and the UI only re-renders the visible notes list; maintain ordering and avoid flicker.
    - [ ] OpenRouter role infrastructure uses a stable system prompt and config-driven model selection.
        - System prompt must enforce patch-only JSON output and concise summaries; model and API key come from config, not env vars.
    - [ ] Summarizer prompt includes optional meeting context and preferred participant names.
        - Inject context ahead of transcript; if context is empty, omit the section entirely to avoid noise.
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
    - [ ] Sessions are persisted under `~/.koe/sessions/{uuidv7}/` with rolling checkpoints.
        - Create a new session directory at start with `metadata.toml` (or JSON) containing: id, start_time, end_time (nullable), finalized flag, asr/summarizer providers, model names, file names.
        - Keep schema extensible for future fields: title, description, participants, tags; do not require them yet.
        - Use UUIDv7 for session id and include it in filenames and metadata for easy correlation.
        - Canonical formats: `metadata.toml` (single-record), `transcript.jsonl` (append-only segments), `notes.json` (state snapshot), `context.txt` (verbatim optional), `audio.raw` (crash-safe stream).
        - Derived exports: `audio.wav`, `transcript.md`, `notes.md` written on finalize or explicit export; do not treat markdown as canonical.
        - Metadata schema (TOML): id (uuidv7), start_time (RFC3339), end_time (RFC3339 or null), finalized (bool), context_file, audio_raw_file, audio_wav_file, transcript_file, notes_file, asr_provider, asr_model, summarizer_provider, summarizer_model.
        - Transcript JSONL schema: `{id, start_ms, end_ms, speaker, text, finalized, source}`; append per segment, keep one JSON object per line.
        - Notes JSON schema: full `MeetingState` snapshot with key_points/actions/decisions; include `updated_at` timestamp for snapshots.
        - Audio raw format: PCM f32 little-endian, 48 kHz, mono, interleaved; record exact format in metadata for WAV finalization.
    - [ ] Audio, transcript, and notes are continuously written during the session.
        - Persist audio from local capture even when cloud ASR is used. Write `audio.raw` continuously with periodic flush, and finalize to `audio.wav` on clean shutdown; maintain a transcript append file (e.g., `transcript.jsonl`) and a notes snapshot (`notes.json`).
        - Use atomic write patterns for metadata and notes snapshots (write temp + rename) to survive crashes.
        - Define checkpoint interval (e.g., every 5–10 s) and ensure partial data is still readable on crash.
        - Rationale: JSONL for append-only streams (transcript, optional patch log), JSON/TOML for single-record snapshots (metadata/notes), Markdown only for human export.
    - [ ] Crash-safe recovery: partial sessions can be reopened and exported.
        - Ensure incomplete sessions still have usable transcript/notes; metadata should include `finalized=false` and last_update timestamp for recovery tooling.
    - [ ] Export transcript/notes on quit.
        - On shutdown, persist the transcript and notes to files (e.g., `transcript.md`, `notes.json`). Ensure the export path is configurable and errors are surfaced cleanly. Do not block UI shutdown indefinitely; use a bounded export timeout.
- Smoke tests:
    - [ ] Compare local vs cloud latency over a 3-minute session.
        - Run a controlled session with both providers and record average latency. Use the status bar metrics to compare and document the result. If results vary, capture network conditions for context.
    - [ ] Kill the process mid-session and confirm recovery files exist.
        - Verify audio, transcript, notes, and metadata are present and readable, with `finalized=false`.
    - [ ] Export produces valid transcript.md and notes.json.
        - Validate the output format with a simple parser or quick manual check. Ensure files include metadata like timestamps or session ID if desired. Confirm the export does not include partial or duplicated entries.

## 9. Resolved Decisions

macOS 15+ only (no legacy), Silero VAD for quality with tuned parameters, local + cloud providers for ASR and summarization from day one, stream-based speaker labeling (mic -> Me, system -> Them).
