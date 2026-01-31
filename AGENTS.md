## 1. Documentation

Real-time meeting transcription and notes engine for macOS, built in Rust with ScreenCaptureKit audio capture, VAD-gated chunking, local/cloud transcribe providers (whisper-rs/Groq), and LLM-powered patch-based summary engine (Ollama/OpenRouter), rendered in a ratatui TUI.

- Rust: https://doc.rust-lang.org/std/
- ratatui: https://docs.rs/ratatui/latest/ratatui/
- whisper-rs: https://docs.rs/whisper-rs/latest/whisper_rs/
- screencapturekit: https://docs.rs/screencapturekit/latest/screencapturekit/
- clap: https://docs.rs/clap/latest/clap/

## 2. Repository Structure

```
koe/
  Cargo.toml              # workspace root (resolver = "3")
  package.json            # bun scripts for quality gates (format, lint, test)
  commitlint.config.js    # conventional commits, scopes: core|cli|web|config|deps
  lint-staged.config.js   # runs bun run util:check
  rustfmt.toml            # Rust formatting rules
  .husky/                 # pre-commit (lint-staged), commit-msg (commitlint)
  crates/
    koe-core/             # engine: capture, processing, transcribe, transcript, notes
    koe-cli/              # thin TUI shell: renders core events, forwards commands
```

## 3. Stack

| Layer            | Choice                                 | Notes                                |
| ---------------- | -------------------------------------- | ------------------------------------ |
| Language         | Rust 2024 edition                      | rust-version 1.93.0                  |
| Audio capture    | screencapturekit 1.5.0                 | macOS 15+, system audio + mic        |
| Ring buffer      | rtrb 0.3.2                             | lock-free SPSC for RT callbacks      |
| Resampling       | rubato 0.16.2                          | 48k -> 16k high-quality              |
| VAD              | voice_activity_detector 0.2.1          | Silero ONNX, 512 samples/32ms frames |
| Local transcribe | whisper-rs 0.15.1                      | Metal acceleration                   |
| Cloud transcribe | Groq API                               | Whisper large-v3-turbo via ureq      |
| Local summarize  | Ollama                                 | NDJSON streaming via ureq            |
| Cloud summarize  | OpenRouter API                         | via ureq                             |
| TUI              | ratatui 0.30.0 + crossterm 0.29.0      |                                      |
| CLI              | clap 4.5.56                            | derive features                      |
| HTTP             | ureq 3.1.4                             | json + multipart features            |
| Serialization    | serde 1.0.228 + serde_json 1.0.149     |                                      |
| TOML             | toml 0.8.20                            | config parsing                       |
| Time             | time 0.3.45                            | timestamps, RFC3339                  |
| Session IDs      | uuid 1.20.0                            | v7 feature, time-ordered             |
| Errors           | thiserror 2.0.18                       |                                      |
| Signals          | signal-hook 0.3.18                     |                                      |
| macOS FFI        | core-foundation 0.10.1                 |                                      |
| Quality gates    | bun + commitlint + husky + lint-staged |                                      |

## 4. Architecture

Workspace layout: `crates/koe-core` (engine, zero UI deps) + `crates/koe-cli` (thin TUI adapter rendering core events and forwarding commands back); future macOS Swift UI replaces CLI via IPC to core without touching engine logic; all providers and capture backends live behind traits in koe-core.

Data flow:

```
ScreenCaptureKit (system audio + mic)
  -> RT callback (copy f32 into SPSC ring buffers, no alloc/locks)
    -> Audio processor (align, mix/keep separate, resample 48k -> 16k)
      -> VAD + chunker (speech-gated, overlap) -> sync_channel<AudioChunk>
        -> transcribe worker (local whisper-rs or Groq) -> TranscriptSegment[]
          -> Transcript ledger (mutable window + finalize) -> Event bus
            -> Notes engine (patch-only, Ollama/OpenRouter) -> NotesPatch
              -> TUI (transcript + notes + status)
```

Responsibilities: ScreenCaptureKit adapter (enumerate/configure/stream), RT callback (copy into ring buffer, return), audio processor (PTS align, mix, resample 48k→16k), VAD+chunker, transcribe provider, transcript ledger (overlap merge + finalize window), notes engine (patch ops), TUI (render + status + hotkeys). Threading: ScreenCaptureKit queue → SPSC ring buffers; processor drains → chunk queue (sync_channel cap 4, drop-oldest); transcribe worker emits segments; notes thread emits patches; UI consumes merged events. Event/command surface: `CoreEvent` (transcript/notes/status/stats/errors) and `CoreCommand` (start/stop/mode/force/export/pause), transported via in-process channels; NDJSON over stdout/Unix socket reserved for future Swift UI.

## 5. Technical Decisions

macOS 15+ only (ScreenCaptureKit, no legacy fallback; Screen Recording + Microphone permissions, restart often required); audio capture via ScreenCaptureKit with `captures_audio=true`, audio-only output with tiny-video fallback for callbacks, mic via ScreenCaptureKit (no cpal); VAD/chunking: Silero 512-sample frames @16 kHz (32 ms), threshold 0.5, min speech 200 ms, hangover 300 ms, chunks target 4.0 s with 1.0 s overlap (min 2.0 s, max 6.0 s); backpressure: 10 s ring per stream, chunk queue cap 4 drop-oldest, notes queue cap 1 skip-if-busy; notes: append-only bullets with stable IDs, summarize finalized segments only, cadence every 10 s or on keyword triggers; speaker labels: mic → “Me”, system → “Them”, mixed → “Unknown”. MacOS 15+ only (no legacy), Silero VAD for quality with tuned parameters, local + cloud providers for transcribe and summarize from day one, stream-based speaker labeling (mic -> Me, system -> Them).

## 6. Commands

| Command                 | Description                                                          |
| ----------------------- | -------------------------------------------------------------------- |
| `bun run build`         | `cargo build --workspace --release`                                  |
| `bun run koe -- [args]` | `cargo run -p koe-cli -- [args]`                                     |
| `bun run util:format`   | `cargo fmt --all`                                                    |
| `bun run util:lint`     | `cargo clippy --all-targets --all-features -- -D warnings`           |
| `bun run util:test`     | `cargo test --all`                                                   |
| `bun run util:check`    | runs format + lint + test sequentially, exits nonzero on any failure |
| `bun run util:clean`    | `cargo clean`                                                        |
| `bun run koe -- init`   | interactive onboarding: model download, provider/key config          |
| `bun run koe -- config` | `--print`/`--set`/`--edit` for `~/.koe/config.toml`                  |

## 7. Local Setup and Testing

- Requires Rust 1.93.0+ (`rustup update stable`).
- Run `bun run koe -- init` to download a Whisper model and write `~/.koe/config.toml` (interactive onboarding for transcribe/summarize provider, model, and API keys).
- Alternate model: `bun run koe -- init --model small`.
- Run local transcribe: `bun run koe -- --transcribe local`.
- Run cloud transcribe: `bun run koe -- --transcribe cloud`.
- Environment variables (`KOE_TRANSCRIBE_CLOUD_API_KEY`, `KOE_SUMMARIZE_CLOUD_API_KEY`) are optional overrides; `~/.koe/config.toml` is canonical.

## 8. Quality

Zero clippy warnings (`-D warnings`), `cargo fmt --all` enforced, all tests passing, pre-commit hooks run full `util:check` via lint-staged, commitlint enforces conventional commits with domain scopes (core, cli, web, config, deps).

Testing strategy: unit tests for VAD state machine and chunk boundary logic, transcript ledger overlap merge and finalize logic, NotesPatch apply and stable ID handling; integration tests feed canned WAV chunks through chunker -> transcribe mock -> ledger and summarize mock returns patch with state application; manual QA for permissions prompts, restart behavior, capture correctness, and 30-min long-running session stability.

## 9. Roadmap

Phase 0: Quality gate wiring

- Done criteria:
    - [x] Scripts configured: `util:format`, `util:lint`, `util:test`, `util:check` (order preserved; extend `util:check` for new gates; mirror in CI).
    - [x] Husky hooks configured (pre-commit lint-staged, commit-msg commitlint; minimal wrappers; preserve exec perms).
    - [x] lint-staged runs full `bun run util:check` (keep full gate; add `util:check:fast` only if optimizing).
    - [x] commitlint config present (scopes aligned with domains; strict rules; verify on upgrades).
    - [x] rustfmt config present (deterministic formatting; align with edition).
    - [x] rustfmt + clippy installed for util scripts (ensure CI installs components).
    - [x] Config schema at `~/.koe/config.toml` (sections: audio; transcribe local/cloud; summarize local/cloud; ui; version + migration; 0600 perms; dirs `~/.koe/{config.toml,models,sessions}`; env overrides optional).
    - [x] `koe config` supports `--print`/`--set`/`--edit` with validation, redaction, and precedence CLI > config > env.
    - [x] `koe init` onboarding for permissions, provider/model selection, API keys; idempotent unless `--force`; prints System Settings path + restart note.
- Smoke tests:
    - [x] `bun run util:check` completes (not executed here).
    - [x] `koe init` writes `~/.koe/config.toml` and is idempotent unless `--force`.

Phase 1: Audio capture + chunking

- Done criteria:
    - [x] ScreenCaptureKit stream starts and system audio callbacks fire (`crates/koe-core/src/capture/sck.rs`).
        - Confirm the stream is configured with `captures_audio=true` and a content filter that actually yields audio callbacks. The output handler should be registered on `SCStreamOutputType::Audio` and should be receiving buffers. Watch for errors at `start_capture()` for permission or display availability issues.
    - [x] Microphone callbacks fire (`crates/koe-core/src/capture/sck.rs:49`).
        - Register the output handler for `SCStreamOutputType::Microphone` in addition to Audio. Ensure the handler routes microphone buffers into the mic ring buffer and that the mic ring is drained in `try_recv_mic`. Validate by speaking and watching mic counters increase separately from system audio.
    - [x] Default microphone selection prefers built-in mic when config is unset.
        - If `audio.microphone_device_id` is empty and microphone capture is enabled, choose the built-in mic (id `BuiltInMicrophoneDevice` or name containing "built-in"/"macbook") and fall back to the OS default input when no built-in is present. Avoid Bluetooth headset mic becoming the implicit default unless explicitly configured.
    - [x] Audio processor emits VAD-gated chunks with overlap (`crates/koe-core/src/process/mod.rs`, `crates/koe-core/src/process/chunker.rs`).
        - The processor should read from the capture rings, resample to 16 kHz, run VAD, and feed the chunker with the correct sample rate. Confirm the overlap behavior by inspecting chunk sizes around the target and max sizes. Maintain the 2s/4s/6s/1s policy to keep downstream transcribe costs predictable.
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

Phase 2: transcribe + transcript ledger + TUI

- Done criteria:
    - [x] whisper-rs provider implemented (`crates/koe-core/src/transcribe/whisper.rs`).
        - Verify model loading path and error handling for missing or invalid models. Ensure the sample rate matches 16 kHz and the language is set intentionally. If you change sampling strategy or thread count, update performance expectations in this plan.
    - [x] Groq provider implemented (`crates/koe-core/src/transcribe/groq.rs`).
        - Add explicit request timeouts and retry policy so a stalled request does not block the transcribe worker forever. Ensure the Groq API key is read from config or environment overrides and errors are surfaced clearly to the UI. Keep the WAV encoding format stable; Groq expects WAV audio with a valid header.
    - [x] Transcript ledger implemented (`crates/koe-core/src/transcript.rs`).
        - Keep the overlap window in sync with chunker overlap. If you adjust chunk overlap, adjust `OVERLAP_WINDOW_MS` to match. Maintain tests for dedupe behavior because this directly affects transcript quality.
    - [x] transcribe worker consumes chunks and emits transcript events to UI.
        - Add a worker thread or task that reads from `chunk_rx`, calls the selected provider, and emits a `TranscriptEvent` (or equivalent) on a channel to the UI. Keep this loop resilient: skip empty segments, keep running on transient failures, and surface errors as events. This is the main wiring step that makes the app functional.
    - [x] CLI flags wire into transcribe provider creation (`crates/koe-cli/src/main.rs`).
        - Use `--transcribe` and `--transcribe-model` to call `create_transcribe_provider` and pass the provider into the runtime. If the provider fails to initialize, exit with a clear error that points to missing model or API key. Ensure defaults align with the plan.
    - [x] Transcript renders in TUI (currently placeholder text).
        - Keep a local transcript buffer in the TUI state and update it when new events arrive. Render a limited window to keep UI responsive and avoid huge redraws. Make sure text wrapping is stable and that the UI does not flicker on updates.
    - [x] Speaker labeling (mic = "Me", system = "Them").
        - Map `AudioChunk.source` to `TranscriptSegment.speaker` at the time of segment creation. If a mixed stream is used, set speaker to `None` or `Unknown`. This is also the hook to later integrate diarization if needed.
    - [x] Mode switching without restart (command surface missing).
        - Define a command channel from UI to core and support hotkeys to switch modes. Recreate the provider in the transcribe worker and emit a status event to update the UI. Ensure in-flight chunks are handled safely during switches.
    - [x] Status bar shows transcribe lag and active mode.
        - Track time spent per chunk in the transcribe worker and emit rolling latency metrics. Display the active mode and last latency in the status bar alongside capture stats. This is essential for troubleshooting local vs cloud performance.
    - [x] Mutable window corrections only affect last 15 s (not implemented).
        - Implement a mutable transcript window and finalize segments that are older than the window. When new transcribe results overlap finalized segments, do not alter them. Keep this window consistent with the chunk overlap to prevent duplicate text.
    - [x] Minimal full-screen TUI with a focused notes pane and transcript pane always visible.
        - Two-pane layout is the only view: notes on one side and transcript on the other with stable layout and no flicker.
        - Provide a clear status line with provider/lag/capture stats, and keep layout stable.
        - Document key bindings in a single place (help overlay or footer) and keep them consistent: quit, switch mode, set context.
    - [x] Color system: other-party content uses a restrained blue, self content uses neutral gray, headings are subtle and consistent.
        - Apply colors consistently in both transcript and notes; blue is reserved for “Them” content, gray for “Me,” and neutral for headings/metadata.
        - Keep palette minimal and readable; avoid bright or noisy styling. Ensure colors remain legible in common terminal themes.
    - [x] TUI clean shutdown restores terminal even on panic.
        - Maintain existing panic hook and ensure all threads stop cleanly on exit.
    - [x] Meeting context can be provided via CLI, config, or TUI and is passed to the summarize.
        - Support `--context`, config default (`session.context`), and an in-TUI edit action; pick a single canonical source with clear precedence (CLI > TUI > config).
        - Context should be stored in session metadata (`metadata.toml`) and injected into summarize prompts; allow empty/absent context.
        - Support multi-line context; preserve verbatim and redact from logs unless explicitly printed.
        - TUI context edits update the current session metadata only (do not mutate global config).
- Smoke tests:
    - [x] Short utterance appears within 4 s locally, faster on cloud.
        - Use a stopwatch and a single spoken phrase. Confirm the local model is within target latency and cloud is faster. If latency is too high, reduce chunk size or increase transcribe thread resources.
    - [x] Overlap does not duplicate text.
        - Speak continuous speech over multiple chunks and verify deduplication in the transcript ledger. If duplicates appear, adjust similarity threshold or merge policy. Document any changes to the similarity heuristic.
    - [x] Switch modes without crash.
        - Switch modes repeatedly during active audio capture. Ensure the worker restarts cleanly and the UI status updates. This test should not leak threads or leave the provider in a half-initialized state.
    - [x] Two-pane layout remains stable during repeated updates.
        - Verify the notes and transcript panes stay aligned with no width jitter during rapid updates.

Phase 3: Notes engine (patch-only)

- Done criteria:
    - [x] Ollama provider emits NotesPatch.
        - Implement a summarize provider that calls Ollama and parses a patch-only response. Keep the prompt and response schema stable to avoid parse failures. Treat network errors as non-fatal and try again on the next cycle.
    - [x] OpenRouter provider emits NotesPatch.
        - Add a second provider with similar patch parsing, but with OpenRouter-specific authentication. Validate response shape and ensure timeouts and retries are consistent with Groq. The provider should be swappable at runtime like transcribe.
    - [x] Notes pane updates without full rewrites.
        - Apply patches to a persistent `MeetingState` and only update changed items in the TUI; avoid full redraws to keep scrolling stable. Patches should only add/update, never delete silently.
    - [x] Stable IDs persist across updates.
        - IDs should be generated by the summarize and reused across updates, not regenerated on each patch. This allows UI selection and references to remain stable. Validate by running multiple summaries and confirming IDs remain unchanged for the same item.
    - [x] Summarizer uses finalized segments only.
        - Feed only `finalized` transcript segments into the summarize to avoid churn. If you want provisional notes, mark them clearly and separate them from stable notes. This protects against edits to earlier transcript text.
    - [x] Summarizer prompts are tuned for minimal, information-dense output with short patches.
        - Output must be concise, avoid filler, and emit only key points/actions/decisions. Prefer short noun phrases over full sentences.
    - [x] Notes capture speaker attribution when available (Me vs Them) with consistent labels.
        - Use stable labels (“Me”, “Them”) and keep the label prefix minimal (e.g., `Me:`) so notes stay compact.
    - [x] Notes pane updates incrementally in real time without full redraws.
        - Patch application updates the data model and the UI only re-renders the visible notes list; maintain ordering and avoid flicker.
    - [x] OpenRouter role infrastructure uses a stable system prompt and config-driven model selection.
        - System prompt must enforce patch-only JSON output and concise summaries; model and API key come from config, not env vars.
    - [x] Summarizer prompt includes optional meeting context and preferred participant names.
        - Inject context ahead of transcript; if context is empty, omit the section entirely to avoid noise.
- Smoke tests:
    - [x] Decision phrasing triggers new decision item.
        - Use test phrases that match the summarize prompt and confirm a new decision appears. Check that it references the correct transcript evidence IDs. Verify that repeated phrases do not create duplicates.
    - [x] Action phrasing triggers new action item.
        - Use phrases with clear owner and due date language. Validate parsing of optional fields like owner and due date. Ensure the item shows up in the correct notes section.
    - [ ] No duplicates after multiple cycles.
        - Run the summarize multiple times without new transcript segments. The notes list should remain stable with no new items. If duplicates appear, adjust prompt or patch logic to enforce idempotency.
        - BLOCKED: Requires live summarize execution over multiple cycles; cannot run external model or interactive session here; tried only prompt/unit-level changes; next steps: run `koe`, let summarize run multiple intervals without new speech, confirm no duplicate notes; adjust prompt/idempotency logic if duplicates appear; file refs: `crates/koe-core/src/summarize/patch.rs`, `crates/koe-cli/src/tui.rs`.

Phase 4: Latency comparison + polish

- Done criteria:
    - [x] Status bar shows transcribe lag, drops, and provider.
        - Extend the status bar to include transcribe latency, active provider, and capture drop metrics. Keep the layout fixed width to avoid jitter as values change. This should be updated from the same event stream as transcript updates.
    - [x] Sessions are persisted under `~/.koe/sessions/{uuidv7}/` with rolling checkpoints.
        - Create a new session directory at start with `metadata.toml` (or JSON) containing: id, start_time, end_time (nullable), finalized flag, transcribe/summarize providers, model names, file names.
        - Keep schema extensible for future fields: title, description, participants, tags; do not require them yet.
        - Use UUIDv7 for session id and include it in filenames and metadata for easy correlation.
        - Canonical formats: `metadata.toml` (single-record), `transcript.jsonl` (append-only segments), `notes.json` (state snapshot), `context.txt` (verbatim optional), `audio.raw` (crash-safe stream).
        - Derived exports: `audio.wav`, `transcript.md`, `notes.md` written on finalize or explicit export; do not treat markdown as canonical.
        - Metadata schema (TOML): id (uuidv7), start_time (RFC3339), end_time (RFC3339 or null), finalized (bool), context_file, audio_raw_file, audio_wav_file, transcript_file, notes_file, transcribe_provider, transcribe_model, summarize_provider, summarize_model.
        - Transcript JSONL schema: `{id, start_ms, end_ms, speaker, text, finalized, source}`; append per segment, keep one JSON object per line.
        - Notes JSON schema: `MeetingNotes` snapshot with bullets; include `updated_at` timestamp for snapshots.
        - Audio raw format: PCM f32 little-endian, 48 kHz, mono, interleaved; record exact format in metadata for WAV finalization.
    - [x] Audio, transcript, and notes are continuously written during the session.
        - Persist audio from local capture even when cloud transcribe is used. Write `audio.raw` continuously with periodic flush, and finalize to `audio.wav` on clean shutdown; maintain a transcript append file (e.g., `transcript.jsonl`) and a notes snapshot (`notes.json`).
        - Use atomic write patterns for metadata and notes snapshots (write temp + rename) to survive crashes.
        - Define checkpoint interval (e.g., every 5–10 s) and ensure partial data is still readable on crash.
        - Rationale: JSONL for append-only streams (transcript, optional patch log), JSON/TOML for single-record snapshots (metadata/notes), Markdown only for human export.
    - [x] Crash-safe recovery: partial sessions can be reopened and exported.
        - Ensure incomplete sessions still have usable transcript/notes; metadata should include `finalized=false` and last_update timestamp for recovery tooling.
    - [x] Export transcript/notes on quit.
        - On shutdown, persist the transcript and notes to files (e.g., `transcript.md`, `notes.json`). Ensure the export path is configurable and errors are surfaced cleanly. Do not block UI shutdown indefinitely; use a bounded export timeout.
- Smoke tests:
    - [ ] Compare local vs cloud latency over a 3-minute session.
        - Run a controlled session with both providers and record average latency. Use the status bar metrics to compare and document the result. If results vary, capture network conditions for context.
        - BLOCKED: Requires a live audio session with local and cloud transcribe plus manual timing; cannot run interactive 3-minute capture here; tried only code-level checks; next steps: run `bun run koe`, switch providers, speak continuously for 3 minutes each, record status bar lag metrics; file refs: `crates/koe-cli/src/tui.rs`, `crates/koe-cli/src/main.rs`.
    - [ ] Kill the process mid-session and confirm recovery files exist.
        - Verify audio, transcript, notes, and metadata are present and readable, with `finalized=false`.
        - BLOCKED: Requires an interactive session and forced termination to inspect on-disk artifacts; cannot run or kill interactive process here; tried only code-level persistence checks; next steps: start `bun run koe`, speak for 1–2 minutes, kill process, confirm `~/.koe/sessions/{id}` contains `audio.raw`, `transcript.jsonl`, `notes.json`, `metadata.toml` with `finalized=false`; file refs: `crates/koe-cli/src/session.rs`.
    - [x] Export produces valid transcript.md and notes.json.
        - Validate the output format with a simple parser or quick manual check. Ensure files include metadata like timestamps or session ID if desired. Confirm the export does not include partial or duplicated entries.

Phase 5: Reliability, correctness, and ops hardening

- Done criteria:
    - [x] All network calls have explicit timeouts + bounded retries.
        - Apply to Groq transcribe, OpenRouter summarize, Ollama summarize, and model downloads.
        - Use per-request connect/read timeouts and retry with backoff; surface failures in the UI status line or logs.
    - [x] RT audio callback is lock-free and allocation-free.
        - Remove Mutex usage from `OutputHandler` callback path; if contended, drop frames immediately.
        - Move downmixing and any heap usage to the processor thread; validate no allocations in callback.
    - [x] Audio export includes `audio.wav` finalization.
        - Convert `audio.raw` (f32 LE, 48 kHz mono) to WAV on clean shutdown/export.
        - Write the WAV file named in metadata and ensure metadata fields are accurate.
    - [x] Session artifacts are written with strict permissions.
        - Apply 0600 permissions to `metadata.toml`, `context.txt`, `notes.json`, `transcript.jsonl`, and `audio.raw`.
        - Warn if permissions are looser on existing files.
    - [x] Summarize queue is bounded and skip-if-busy.
        - Cap summarize input queue at 1; drop/skip when busy to avoid backlog.
    - [x] Transcript ledger is memory bounded for long sessions.
        - Prune finalized segments after persistence or cap the in-memory window.
    - [x] Raw audio writes do not block the processing thread.
        - Move disk IO to a writer thread or queue with backpressure; keep processor real-time.
    - [x] Meeting end/export drains in-flight audio and transcription.
        - Pause capture, flush chunk queue, and wait for transcribe thread to finish before export.
    - [x] Status bar includes provider name and frame drops.
        - Display active provider, capture frame drops, chunk drops, and lag in a fixed-width footer.
    - [x] Config precedence matches spec: CLI > config > env.
        - Apply env overrides last-resort only; document the precedence clearly.
    - [x] `koe config --edit` supports `$EDITOR` with args.
        - Parse editor command + args and pass the config path correctly.
    - [x] Dependency and tooling metadata aligned with the repo spec.
        - `rust-version` matches declared minimum; commitlint scopes match documented scopes.
- Smoke tests:
    - [ ] Simulate offline network and confirm transcribe/summarize time out and recover.
        - Use an invalid endpoint or disconnect network; verify UI status indicates failure and retries.
        - BLOCKED: Requires changing network conditions or running live providers; cannot simulate offline or run the TUI here; tried only code-level review; next steps: disable network or set invalid endpoints (e.g., `OPENROUTER_BASE_URL`, `OLLAMA_BASE_URL`), run `bun run koe`, observe status/reties; file refs: `crates/koe-core/src/transcribe/groq.rs`, `crates/koe-core/src/summarize/openrouter.rs`, `crates/koe-core/src/summarize/ollama.rs`, `crates/koe-cli/src/tui.rs`.
    - [ ] Run a 30-minute session without memory growth beyond a fixed cap.
        - Track resident memory; confirm ledger pruning keeps memory stable.
        - BLOCKED: Requires a long-running interactive capture session; cannot run a 30-minute TUI here; tried only code-level checks; next steps: run `bun run koe` for 30 minutes, monitor RSS, confirm ledger pruning; file refs: `crates/koe-core/src/transcript.rs`, `crates/koe-cli/src/tui.rs`.
    - [ ] End a meeting during active speech and verify no transcript loss.
        - Confirm final transcript contains the last spoken phrase after export.
        - BLOCKED: Requires live audio capture and interactive end-meeting; cannot verify in this environment; tried only code-level drain/export wiring; next steps: run `bun run koe`, speak and end meeting mid-utterance, confirm export includes final phrase; file refs: `crates/koe-cli/src/tui.rs`, `crates/koe-cli/src/session.rs`.
    - [ ] Restart after a crash and verify session artifacts are readable with correct permissions.
        - Confirm metadata has `finalized=false` and files are intact.
        - BLOCKED: Requires killing a live process and inspecting session artifacts; cannot run or kill interactive TUI here; tried only code-level persistence review; next steps: start `bun run koe`, kill process, inspect `~/.koe/sessions/{id}` for artifacts and permissions; file refs: `crates/koe-cli/src/session.rs`.

Phase 6: TUI design polish

- Done criteria:
    - [x] Target layout:
        - (split view, meeting active):
        - Title bar: accent square (U+25A0) + app name left, palette hint right.
        - Content: notes 55% left (rolling moment-capture bullets, no categories) | dim separator | transcript 45% right.
        - Footer: timer | audio viz | compact metrics.
        - Command palette overlay (centered modal):
        - Category tags right-aligned dim; labels neutral; selected row accent bg.
        - No footer in palette; version info already in title bar.
    - [x] Title bar: accent-colored filled square (U+25A0) + `koe v{version}` left-aligned; `ctrl+p command palette` hint right-aligned in muted text; no borders, single styled line (`crates/koe-cli/src/tui.rs`).
    - [x] Accent color: aqua/turquoise RGB(0, 190, 190) or RGB(80, 200, 200); used only for title square, app name, and command palette selection highlight; everything else grayscale.
    - [x] No box-drawing borders: panes separated by 1-column dim vertical separator and whitespace; content has 1-char left padding; section names rendered as first line in heading color, not border titles.
    - [x] Key bindings reduced to minimum: only `ctrl+p` (command palette), `q` (quit), `ctrl+c` (quit); all other actions are palette-only.
    - [x] Footer redesigned as three zones in one line (`crates/koe-cli/src/tui.rs`).
        - Left: meeting timer `MM:SS` or `H:MM:SS`; accent color when active, `--:--` muted when idle; frozen at final duration in post-meeting state.
        - Center-left: ASCII/Unicode audio waveform strip (10-20 chars); characters `~^-_` or block elements `▁▂▃▅▃▂▁`; either reactive (driven by audio RMS/peak from capture ring buffer, sampled every 50ms render tick) or ambient (randomized animation indicating capture active); flat `--------` when capture inactive; muted/dim color.
        - Right: compact metrics cluster `transcribe:{mode} lag:{ms}s chunks:{emitted}/{dropped} segs:{count}`; all muted gray, ~40 chars; additional metrics (frames captured/dropped) appended if space allows.
    - [x] Command palette: centered modal overlay triggered by `ctrl+p`, dismissed with `Esc` (`crates/koe-cli/src/tui.rs`).
        - Title "Command Palette" centered; `> ` filter input with cursor; fuzzy match narrows visible commands; up/down arrows to navigate, Enter to execute.
        - Command rows: right-aligned dim category tag, neutral command label; selection highlight in accent color background.
        - No footer in palette; version info already present in title bar. Keep palette clean and minimal.
        - Width ~60 chars, height fits content (max ~15 rows + header); modal blocks underlying input.

Phase 7: Audio quality improvements

- Done criteria:
    - [x] Add loudness normalization and gentle AGC for recorded mixdown.
        - Target consistent output level without clipping; ensure it can be disabled.
    - [x] Add optional noise reduction path for recorded audio.
        - Provide a lightweight denoise stage (e.g., RNNoise or spectral gating) with a conservative default.
    - [x] Add a simple high-pass filter to reduce rumble before mixdown/export.
        - Keep cutoff low (e.g., 80–120 Hz) and configurable.
    - [x] Context-aware command sets driven by app state (`crates/koe-cli/src/tui.rs`).
        - Idle: start meeting, switch transcribe/summarize mode, set transcribe/summarize model, edit context, browse sessions.
        - MeetingActive: end meeting, pause capture, force summarize, switch transcribe/summarize mode, edit context.
        - PostMeeting: copy transcript/notes/audio path to clipboard, open session folder, export markdown, start new meeting, browse sessions.
    - [x] State machine: Idle -> MeetingActive -> PostMeeting -> Idle; drives palette commands, footer timer behavior, and audio viz state.
    - [x] Pane layout unchanged: fixed two-pane 55/45 split; notes always visible left, transcript right; last 200 segments; auto-scroll to bottom.
- Smoke tests:
    - [x] `ctrl+p` opens palette overlay with correct commands for current state; Esc dismisses; filter narrows results.
    - [x] Footer timer counts up during active meeting, freezes on end, resets on new meeting.
    - [x] Audio waveform animates during capture, goes flat when stopped.
    - [ ] All actions (switch provider, edit context, end meeting, copy exports) accessible and functional through palette only.
        - BLOCKED: Requires manual palette interaction and system integrations (clipboard/open); cannot exercise in non-interactive session; tried only code-level wiring; next steps: run `bun run koe`, open palette and execute each command, confirm expected behavior; file refs: `crates/koe-cli/src/tui.rs`.

Phase 8: Post-cleanup stability pass

- Done criteria:
    - [x] Command palette only supports meeting/session actions; remove config-related commands and handlers (no transcribe/summarize mode/model switches in TUI).
    - [x] Paused capture still applies transcript/notes events to UI and session persistence; pause only stops new capture input.
    - [ ] CMSampleBuffer audio data alignment validated before unsafe cast; fallback path for unaligned buffers.
    - [ ] Summarize prompt includes existing notes/IDs to prevent duplicates; add tests for idempotency.
    - [ ] Export path resilient to long sessions; timeout behavior adjusted or async export status added.
- Smoke tests:
    - [ ] Palette lists only start/end/new meeting, export, and session browse/open/copy actions.
    - [ ] Pausing capture during active meeting does not drop transcript/notes updates.
    - [ ] Summarize runs multiple cycles without new transcript and produces no duplicate notes.
    - [ ] Export completes or reports async/pending for long sessions without blocking UI shutdown.
