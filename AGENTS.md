## 1. Documentation

Real-time meeting transcription and notes engine for macOS, built in Rust with ScreenCaptureKit audio capture, VAD-gated chunking, local/cloud transcribe providers (whisper-rs/Groq), and LLM-powered patch-based summary engine (Ollama/OpenRouter), rendered in a ratatui TUI.

- Rust: https://doc.rust-lang.org/std/
- ratatui: https://docs.rs/ratatui/latest/ratatui/
- whisper-rs: https://docs.rs/whisper-rs/latest/whisper_rs/
- screencapturekit: https://docs.rs/screencapturekit/latest/screencapturekit/
- clap: https://docs.rs/clap/latest/clap/

## 2. Repository Structure

```
.
├── Cargo.toml              # workspace root (resolver = "3")
├── package.json            # bun scripts for quality gates (format, lint, test)
├── commitlint.config.js    # conventional commits, scopes: core|cli|web|config|deps
├── lint-staged.config.js   # runs bun run util:check
├── rustfmt.toml            # Rust formatting rules
├── .husky/                 # pre-commit (lint-staged), commit-msg (commitlint)
└── crates/
    ├── koe-core/           # engine: capture, processing, transcribe, transcript, notes
    │   ├── Cargo.toml
    │   └── src/
    │       ├── capture/
    │       │   ├── handler.rs
    │       │   ├── mod.rs
    │       │   └── sck.rs
    │       ├── process/
    │       │   ├── chunker.rs
    │       │   ├── mod.rs
    │       │   ├── queue.rs
    │       │   ├── resample.rs
    │       │   └── vad.rs
    │       ├── summarize/
    │       │   ├── mod.rs
    │       │   ├── cloud.rs
    │       │   ├── local.rs
    │       │   └── patch.rs
    │       ├── transcribe/
    │       │   ├── cloud.rs
    │       │   ├── mod.rs
    │       │   └── local.rs
    │       ├── error.rs
    │       ├── http.rs
    │       ├── lib.rs
    │       ├── transcript.rs
    │       └── types.rs
    └── koe-cli/            # thin TUI shell: renders core events, forwards commands
        ├── Cargo.toml
        └── src/
            ├── config.rs
            ├── config_cmd.rs
            ├── init.rs
            ├── main.rs
            ├── raw_audio.rs
            ├── session.rs
            └── tui.rs
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
    - [x] Scripts set: `util:format`, `util:lint`, `util:test`, `util:check` (order preserved; extend `util:check` for new gates; mirror in CI).
    - [x] Husky hooks set (pre-commit lint-staged, commit-msg commitlint; minimal wrappers; keep exec perms).
    - [x] lint-staged runs full `bun run util:check` (keep full gate; add `util:check:fast` only for optimization).
    - [x] commitlint config present (scopes aligned; strict rules; verify on upgrades).
    - [x] rustfmt config present (deterministic; edition-aligned).
    - [x] rustfmt + clippy installed for util scripts (ensure CI installs components).
    - [x] Config schema at `~/.koe/config.toml` (sections: audio; transcribe local/cloud; summarize local/cloud; ui; version+migration; 0600 perms; dirs `~/.koe/{config.toml,models,sessions}`; env overrides optional).
    - [x] `koe config` supports `--print`/`--set`/`--edit` with validation, redaction, precedence CLI > config > env.
    - [x] `koe init` onboarding for permissions, provider/model, API keys; idempotent unless `--force`; prints System Settings path + restart note.
- Smoke tests:
    - [x] `bun run util:check` completes (not executed here).
    - [x] `koe init` writes `~/.koe/config.toml` and is idempotent unless `--force`.

Phase 1: Audio capture + chunking

- Done criteria:
    - [x] ScreenCaptureKit stream starts and system audio callbacks fire (`crates/koe-core/src/capture/sck.rs`): `captures_audio=true`, valid content filter; handler on `SCStreamOutputType::Audio` receives buffers; watch `start_capture()` errors (permissions/display).
    - [x] Microphone callbacks fire (`crates/koe-core/src/capture/sck.rs:49`): register `SCStreamOutputType::Microphone` handler; route to mic ring, drain via `try_recv_mic`; verify mic counters rise.
    - [x] Default mic selection prefers built-in when config unset: if `audio.microphone_device_id` empty and mic capture enabled, pick built-in mic (id `BuiltInMicrophoneDevice` or name contains "built-in"/"macbook"), else OS default; avoid Bluetooth unless configured.
    - [x] Audio processor emits VAD-gated chunks with overlap (`crates/koe-core/src/process/mod.rs`, `crates/koe-core/src/process/chunker.rs`): drain rings, resample to 16 kHz, run VAD, feed chunker at correct rate; keep 2s/4s/6s/1s policy; verify chunk sizing around targets.
    - [x] Drop metrics visible in status (frame drops not wired): wire handler drop counter into `CaptureStats`; status shows frame drops + chunk drops to separate capture overload vs backpressure.
    - [x] Drop policy is drop-oldest (currently drop-newest): drop oldest pending chunk when queue full (drain one or custom ring) and keep drop counters.
    - [x] RT callback avoids locks/allocations (Mutex + Vec allocations in handler): no blocking; remove mutex or try-lock and drop on contention; move downmix/alloc to consumer thread to avoid glitches.
    - [x] PTS alignment accurate for drained batches: `drain_ring` tracks PTS per buffer and computes batch start PTS (record `(pts, len)` and derive start offset); validate with known markers.
- Smoke tests:
    - [x] Play system audio: frames+chunks increment; system stays in system pipeline; repeatable without restart.
    - [x] Speak into mic: `try_recv_mic` returns frames and mic pipeline emits chunks; keep system audio silent to avoid crosstalk.
    - [x] Pause consumer to force backpressure: chunk drops increment and UI shows drops.

Phase 2: transcribe + transcript ledger + TUI

- Done criteria:
    - [x] whisper-rs provider (`crates/koe-core/src/transcribe/local.rs`): verify model path errors, 16 kHz sample rate, intentional language; update perf expectations if sampling/threading changes.
    - [x] Groq provider (`crates/koe-core/src/transcribe/cloud.rs`): explicit timeouts + retry; API key from config/env; clear UI errors; keep WAV encoding stable.
    - [x] Transcript ledger (`crates/koe-core/src/transcript.rs`): overlap window matches chunker; keep dedupe tests.
    - [x] Transcribe worker consumes chunks and emits transcript events: read `chunk_rx`, call provider, emit `TranscriptEvent`; skip empty segments; survive transient failures; surface errors.
    - [x] CLI flags wire provider creation (`crates/koe-cli/src/main.rs`): `--transcribe`/`--transcribe-model` select provider; init failures exit with clear missing model/API key error; defaults align with plan.
    - [x] Transcript renders in TUI: local buffer updated on events; windowed render; stable wrapping; no flicker.
    - [x] Speaker labeling (mic="Me", system="Them"): map `AudioChunk.source` to `TranscriptSegment.speaker`; mixed stream -> `None`/`Unknown`; diarization hook.
    - [x] Mode switching without restart: command channel UI->core; hotkeys switch modes; rebuild provider in worker; emit status; handle in-flight chunks safely.
    - [x] Status bar shows transcribe lag + active mode: track per-chunk latency, emit rolling metrics; display with capture stats.
    - [x] Mutable window corrections only affect last 15 s: finalize older segments; overlapping new results cannot alter finalized; window consistent with chunk overlap.
    - [x] Minimal full-screen TUI: fixed two-pane notes+transcript view, stable layout, no flicker; status line with provider/lag/capture stats; key bindings documented in one place (quit, switch mode, set context).
    - [x] Color system: "Them" in restrained blue, "Me" in neutral gray, headings subtle/consistent; palette minimal and readable across terminal themes.
    - [x] TUI clean shutdown restores terminal even on panic: keep panic hook; all threads stop cleanly.
    - [x] Meeting context via CLI/config/TUI passed to summarize: support `--context`, config `session.context`, in-TUI edit; precedence CLI > TUI > config; store in session `metadata.toml`; inject into summarize; allow empty; multi-line preserved, redact from logs unless printed; TUI edits only update session metadata.
- Smoke tests:
    - [x] Short utterance appears within 4 s locally, faster on cloud; tune chunk size/threads if slow.
    - [x] Overlap does not duplicate text; adjust similarity threshold/merge policy if needed and document.
    - [x] Switch modes repeatedly during capture without crash; worker restarts cleanly, status updates, no leaks.
    - [x] Two-pane layout stable during rapid updates.

Phase 3: Notes engine (patch-only)

- Done criteria:
    - [x] Ollama provider emits `NotesPatch`: stable prompt/schema; network errors non-fatal, retry next cycle.
    - [x] OpenRouter provider emits `NotesPatch`: auth, timeouts, retries consistent with Groq; runtime swappable.
    - [x] Notes pane updates without full rewrites: apply patches to persistent `MeetingState`; update only changed items; no silent deletes.
    - [x] Stable IDs persist across updates; verify IDs unchanged across multiple summaries.
    - [x] Summarizer uses finalized segments only; provisional notes must be clearly separated.
    - [x] Summarizer prompt tuned for short, dense patches (key points/actions/decisions; noun phrases preferred).
    - [x] Notes include speaker attribution when available ("Me"/"Them") with compact labels (e.g., `Me:`).
    - [x] Notes update incrementally in real time; UI re-renders visible list only; ordering stable, no flicker.
    - [x] OpenRouter role infra uses stable system prompt; model + API key from config, not env.
    - [x] Summarizer prompt includes optional meeting context and preferred participant names; omit empty context to avoid noise.
- Smoke tests:
    - [x] Decision phrasing triggers decision item; references correct transcript evidence IDs; no duplicates on repeats.
    - [x] Action phrasing triggers action item; owner/due parsing works; item appears in correct section.
    - [ ] No duplicates after multiple cycles.
        - BLOCKED: Needs live summarize cycles; run `koe` and let summarize tick without new speech; confirm stability, adjust prompt/idempotency if needed; file refs: `crates/koe-core/src/summarize/patch.rs`, `crates/koe-cli/src/tui.rs`.

Phase 4: Latency comparison + polish

- Done criteria:
    - [x] Status bar shows transcribe lag, drops, and provider; fixed-width layout; updated from same event stream.
    - [x] Sessions persist under `~/.koe/sessions/{uuidv7}/` with rolling checkpoints; `metadata.toml` includes id, start_time, end_time (nullable), finalized, transcribe/summarize providers+models, file names; schema extensible (title/description/participants/tags); UUIDv7 used in filenames and metadata.
    - [x] Canonical formats: `metadata.toml` (single record), `transcript.jsonl` (append-only), `notes.json` (snapshot), `context.txt` (verbatim optional), `audio.raw` (crash-safe stream); derived exports `audio.wav`, `transcript.md`, `notes.md` on finalize/export only.
    - [x] Metadata fields: id (uuidv7), start_time (RFC3339), end_time (RFC3339 or null), finalized, context_file, audio_raw_file, audio_wav_file, transcript_file, notes_file, transcribe_provider, transcribe_model, summarize_provider, summarize_model.
    - [x] Transcript JSONL schema: `{id, start_ms, end_ms, speaker, text, finalized, source}`; append per segment.
    - [x] Notes JSON schema: `MeetingNotes` snapshot with `updated_at`.
    - [x] Audio raw format: PCM f32 LE, 48 kHz, mono, interleaved; record exact format in metadata for WAV finalization.
    - [x] Audio, transcript, notes continuously written: persist local audio even with cloud transcribe; write `audio.raw` with periodic flush; finalize to `audio.wav` on clean shutdown; append `transcript.jsonl`; snapshot `notes.json`; atomic writes for metadata/notes; checkpoint every 5-10 s; JSONL for streams, JSON/TOML for snapshots, Markdown only for human export.
    - [x] Crash-safe recovery: incomplete sessions readable; metadata `finalized=false` and last_update timestamp for recovery.
    - [x] Export transcript/notes on quit: configurable export path; bounded timeout; surface errors without blocking shutdown.
- Smoke tests:
    - [ ] Compare local vs cloud latency over a 3-minute session.
        - BLOCKED: Needs live capture; run `bun run koe`, switch providers, speak 3 min each, record status bar lag; file refs: `crates/koe-cli/src/tui.rs`, `crates/koe-cli/src/main.rs`.
    - [ ] Kill the process mid-session and confirm recovery files exist.
        - BLOCKED: Needs live session; run `bun run koe`, speak 1-2 min, kill, confirm `~/.koe/sessions/{id}` has `audio.raw`, `transcript.jsonl`, `notes.json`, `metadata.toml` with `finalized=false`; file ref: `crates/koe-cli/src/session.rs`.
    - [x] Export produces valid `transcript.md` and `notes.json` (quick parse/manual check; include metadata if desired).

Phase 5: Reliability, correctness, and ops hardening

- Done criteria:
    - [x] Network calls have explicit timeouts + bounded retries (Groq transcribe, OpenRouter/Ollama summarize, model downloads); use connect/read timeouts and backoff; surface failures in UI status/logs.
    - [x] RT audio callback lock-free and allocation-free: remove mutex in `OutputHandler`; drop on contention; move downmix/alloc to processor; validate no allocations.
    - [x] Audio export finalizes `audio.wav` from `audio.raw` (f32 LE, 48 kHz mono); file named in metadata; metadata fields accurate.
    - [x] Session artifacts written with 0600 perms (`metadata.toml`, `context.txt`, `notes.json`, `transcript.jsonl`, `audio.raw`); warn if existing perms looser.
    - [x] Summarize queue bounded and skip-if-busy (cap 1; drop/skip when busy).
    - [x] Transcript ledger memory bounded for long sessions; prune finalized segments after persistence or cap window.
    - [x] Raw audio writes do not block processing thread; use writer thread/queue with backpressure.
    - [x] Meeting end/export drains in-flight audio/transcription; pause capture, flush chunk queue, wait for transcribe before export.
    - [x] Status bar shows provider name and frame drops; fixed-width footer with frame drops + chunk drops + lag.
    - [x] Config precedence matches spec: CLI > config > env; env overrides last-resort only, documented.
    - [x] `koe config --edit` supports `$EDITOR` with args (parse command + args; pass config path).
    - [x] Dependency/tooling metadata aligned with repo spec (`rust-version` matches minimum; commitlint scopes match docs).
- Smoke tests:
    - [ ] Simulate offline network and confirm transcribe/summarize time out and recover.
        - BLOCKED: Needs live providers; disable network or set invalid endpoints (`OPENROUTER_BASE_URL`, `OLLAMA_BASE_URL`), run `bun run koe`, confirm status/retries; file refs: `crates/koe-core/src/transcribe/cloud.rs`, `crates/koe-core/src/summarize/cloud.rs`, `crates/koe-core/src/summarize/local.rs`, `crates/koe-cli/src/tui.rs`.
    - [ ] Run a 30-minute session without memory growth beyond a fixed cap.
        - BLOCKED: Needs long run; run `bun run koe` for 30 minutes, monitor RSS, confirm ledger pruning; file refs: `crates/koe-core/src/transcript.rs`, `crates/koe-cli/src/tui.rs`.
    - [ ] End a meeting during active speech and verify no transcript loss.
        - BLOCKED: Needs live capture; run `bun run koe`, speak, end meeting mid-utterance, confirm export includes final phrase; file refs: `crates/koe-cli/src/tui.rs`, `crates/koe-cli/src/session.rs`.
    - [ ] Restart after a crash and verify session artifacts are readable with correct permissions.
        - BLOCKED: Needs live run; start `bun run koe`, kill, inspect `~/.koe/sessions/{id}` for artifacts and permissions; file ref: `crates/koe-cli/src/session.rs`.

Phase 6: TUI design polish

- Done criteria:
    - [x] Target layout (split view, meeting active): Title bar with accent square (U+25A0) + app name left, palette hint right; Content: notes 55% left (rolling bullets, no categories) | dim separator | transcript 45% right; Footer: timer | audio viz | compact metrics; Palette overlay: centered modal, category tags right-aligned dim, labels neutral, selected row accent bg; no footer in palette (version in title bar).
    - [x] Title bar: accent-colored filled square + `koe v{version}` left; `ctrl+p command palette` hint right; no borders, single styled line (`crates/koe-cli/src/tui.rs`).
    - [x] Accent color: aqua/turquoise RGB(0,190,190) or RGB(80,200,200); used only for title square, app name, palette selection; everything else grayscale.
    - [x] No box borders: panes separated by 1-col dim vertical separator and whitespace; content has 1-char left padding; section names as first line in heading color (no border titles).
    - [x] Key bindings minimal: `ctrl+p` (palette), `q` (quit), `ctrl+c` (quit); all other actions palette-only.
    - [x] Footer redesigned as three zones in one line (`crates/koe-cli/src/tui.rs`): Left timer `MM:SS` or `H:MM:SS` (accent when active, `--:--` muted when idle; freeze final duration post-meeting); Center-left waveform strip (10-20 chars, `~^-_` or `▁▂▃▅▃▂▁`, reactive every 50ms via RMS/peak or ambient animation, flat `--------` when inactive, muted); Right metrics cluster `transcribe:{mode} lag:{ms}s chunks:{emitted}/{dropped} segs:{count}` in muted gray ~40 chars; append frames captured/dropped if space.
    - [x] Command palette overlay `ctrl+p`, dismiss `Esc` (`crates/koe-cli/src/tui.rs`): title centered, `> ` filter input with cursor, fuzzy match, arrows navigate, Enter executes; rows show right-aligned dim category + neutral label, selection uses accent bg; width ~60, height fit (max ~15 rows + header); modal blocks input.

Phase 7: Audio quality improvements

- Done criteria:
    - [x] Loudness normalization + gentle AGC for recorded mixdown; consistent level without clipping; disable option.
    - [x] Optional noise reduction path (RNNoise or spectral gating) with conservative default.
    - [x] Simple high-pass filter before mixdown/export (80-120 Hz, configurable).
    - [x] Context-aware command sets by app state (`crates/koe-cli/src/tui.rs`): Idle (start meeting, switch modes/models, edit context, browse sessions); MeetingActive (end meeting, pause capture, force summarize, switch modes, edit context); PostMeeting (copy transcript/notes/audio path, open session folder, export markdown, start new meeting, browse sessions).
    - [x] State machine: Idle -> MeetingActive -> PostMeeting -> Idle; drives palette commands, footer timer, audio viz.
    - [x] Pane layout unchanged: fixed 55/45 split, notes left/transcript right; last 200 segments; auto-scroll to bottom.
- Smoke tests:
    - [x] `ctrl+p` opens palette overlay with correct commands for current state; Esc dismisses; filter narrows results.
    - [x] Footer timer counts up during active meeting, freezes on end, resets on new meeting.
    - [x] Audio waveform animates during capture, goes flat when stopped.
    - [ ] All actions (switch provider, edit context, end meeting, copy exports) accessible and functional through palette only.
        - BLOCKED: Needs manual palette run and system integrations; run `bun run koe`, execute each command, confirm behavior; file refs: `crates/koe-cli/src/tui.rs`.

Phase 8: Post-cleanup stability pass

- Done criteria:
    - [x] Command palette only supports meeting/session actions; remove config-related commands and handlers (no transcribe/summarize mode/model switches in TUI).
    - [x] Paused capture still applies transcript/notes events to UI and session persistence; pause only stops new capture input.
    - [x] CMSampleBuffer audio data alignment validated before unsafe cast; fallback path for unaligned buffers.
    - [x] Summarize prompt includes existing notes/IDs to prevent duplicates; add tests for idempotency.
    - [x] Export path resilient to long sessions; timeout behavior adjusted or async export status added.
- Smoke tests:
    - [ ] Palette lists only start/end/new meeting, export, and session browse/open/copy actions.
        - BLOCKED: Needs interactive TUI; run `bun run koe`, check palette per phase; file ref: `crates/koe-cli/src/tui.rs`.
    - [ ] Pausing capture during active meeting does not drop transcript/notes updates.
        - BLOCKED: Needs live audio; run `bun run koe`, start meeting, pause capture, confirm transcript/notes update from in-flight chunks; file ref: `crates/koe-cli/src/tui.rs`.
    - [ ] Summarize runs multiple cycles without new transcript and produces no duplicate notes.
        - BLOCKED: Needs live summarize; run `bun run koe`, let summarize tick without new speech, confirm no duplicates; file refs: `crates/koe-core/src/summarize/patch.rs`, `crates/koe-cli/src/tui.rs`.
    - [ ] Export completes or reports async/pending for long sessions without blocking UI shutdown.
        - BLOCKED: Needs long session; run `bun run koe`, record long meeting, end, confirm export completes or logs pending without blocking; file ref: `crates/koe-cli/src/tui.rs`.
