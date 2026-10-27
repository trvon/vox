# Vox — Agent Instructions

## Project Overview

Vox is a local voice MCP server in Rust. It provides text-to-speech (Kokoro) and speech-to-text (Moonshine Base) over MCP, plus long-form recording and file transcription with timestamped, exportable transcripts. It runs either over stdio (one process per client) or as an HTTP daemon (one shared process, models loaded once).

## Build & Test

```bash
cargo check          # type-check only
cargo test           # unit tests + tests/config_integration.rs + tests/invariants.rs (proptest)
cargo clippy --all-targets -- -D warnings  # lint — must pass with zero warnings
cargo fmt --check
cargo bench                  # Criterion benchmarks (DSP, TTS utils, pipeline)
scripts/profile.sh rtf       # end-to-end RTF + WER on a synthetic lecture (needs models)
scripts/profile.sh perf      # perf profile of `vox transcribe` (task-clock; hybrid-CPU safe)
```

The first four must pass cleanly before submitting changes. Run `PROPTEST_CASES=2560 cargo test --release --test invariants` after touching `audio.rs` or `transcript.rs`. See [BENCHMARKS.md](BENCHMARKS.md).

## Architecture

| Module | Purpose |
|--------|---------|
| `main.rs` | Entry point, config loading, model download, stdio/daemon startup, `vox transcribe` |
| `cli.rs` | Clap CLI parser: tool, daemon, config, transcribe, download-models subcommands |
| `server.rs` | MCP tool handlers (thin layer over `VoiceRuntime`) |
| `runtime.rs` | `VoiceRuntime`: shared voice state (TTS queue, inbox, background listener, recordings) |
| `recording.rs` | Long-form recording + file transcription jobs: segmenter (VAD + ring buffer padding), STT worker, durable store (manifest/JSONL/WAV), paging, export |
| `transcript.rs` | Pure transcript layout: paragraphs, sections, markdown/SRT/VTT/text, repetition cleanup, WER |
| `tool_client.rs` | `vox tool …` HTTP MCP client for the daemon |
| `calibrate.rs` | Genetic-algorithm DSP calibration |
| `tts.rs` | Kokoro TTS via the official `sherpa-onnx` crate (`generate_with_config` streaming callback), voice resolution, sentence splitting |
| `audio.rs` | cpal capture/playback, `StreamingResampler` (stateful, anti-aliased Lanczos-3), DSP (HPF, noise gate, normalize, bounded gain) |
| `stt.rs` | Moonshine Base (default) or Parakeet TDT v2/v3 via `sherpa_onnx::OfflineRecognizer`, optional timed tokens (1 thread by default; `VOX_STT_THREADS`) |
| `vad.rs` | Silero VAD wrapper: `VadOptions` (always sets `max_speech_secs`), absolute u64 segment start |
| `config.rs` | TOML config loading, env var overrides (`VOX_*` prefix), path resolution |
| `daemon.rs` | HTTP daemon lifecycle: daemonize, PID file, start/stop/status/log |
| `models.rs` | Model readiness checks and download/extraction |
| `lib.rs` | Public modules for benches/tests/examples (`audio`, `calibrate`, `config`, `error`, `stt`, `transcript`, `tts`, `vad`) |
| `error.rs` | `VoiceError` enum with `thiserror` derives |

### Transport Modes

- **Stdio** (default): `rmcp::transport::stdio()`. One process per MCP client.
- **Daemon** (`vox daemon start`): `StreamableHttpService` (rmcp 3.x). One process with models loaded once. **A single `VoiceRuntime` is shared by every session**: protocol `2026-07-28` is stateless (the factory may run once per request), and recordings and the TTS queue must survive reconnects, so never put state in `VoiceMcpServer`. Origin validation is enforced (browsers are rejected), because the daemon controls the microphone. On shutdown, running recordings are finalized.

### Streaming TTS

Uses `OfflineTts::generate_with_config` with a chunk callback (Kokoro emits one per internal chunk). The callback copies the chunk into a `Vec<f32>` and sends it through a `std::sync::mpsc` channel. A consumer task plays the chunks through a cpal output stream backed by an `Arc<Mutex<VecDeque<f32>>>` ring buffer.

### Long-form recording

Live capture uses 80 Hz HPF, no gate, a 4096-window channel and dropped-window counting. Audio goes to the `Segmenter` (VAD `for_lecture` plus a 90 s ring buffer for ±250 ms padding), then to an STT worker on `spawn_blocking`, then to `transcript.jsonl`. File jobs run the same `Segmenter` through `AudioSource` (hound for WAV, ffmpeg otherwise) and `StreamingResampler`.

**Invariants:**
- Timestamps are on the audio timeline. Absolute chunk positions locate dropped windows, including the tail; they are replaced with silence and recorded in `gaps`.
- Capture admission is process-wide across listening, recording and calibration. The permit is released only after joining capture.
- A job holds an OS ownership lock. Other processes observe it without rewriting its manifest. Terminal completion is retained for late subscribers.
- Persist snapshots serially; commit transcript counts only after successful appends. Storage/STT errors fail the job. The live STT queue is bounded.
- Job ids are validated, so they can't escape the recordings root.
- `stop` (finalize) and `cancel` (skip remaining STT) never delete audio.

**Callback convention**: the safe API's callback returns `true` to continue and `false` to stop (the C API underneath uses `1`/`0`).

**sherpa-onnx is pinned to `=1.13.8`.** Version 2.0 removed espeak-ng, so Kokoro drops out-of-lexicon words and numbers. Don't bump it without adding a G2P.

## Code Conventions

- **Edition 2024** — uses `let` chains (`if let Ok(x) = ... && let Ok(y) = ...`)
- **Visibility**: use `pub(crate)` to expose private functions for testing rather than making them fully `pub`
- **Error enum**: `VoiceError` carries forward-looking variants (`ModelNotFound`, `Config`, `Timeout`) that are `#[allow(dead_code)]` until wired up
- **Clippy**: treat all warnings as errors (`-D warnings`). Fix collapsible-if, dead code, etc. rather than suppressing broadly
- **Tests**: inline `#[cfg(test)] mod tests` per module, plus model-free config integration/property tests. Use `tempfile` for filesystem tests. Model-backed regressions are explicitly ignored by default.
- **`unsafe impl Send`**: `TtsEngine` and `CaptureHandle` have manual `Send` impls due to non-Send cpal/sherpa internals confined to dedicated threads

## Config Precedence

1. Compiled defaults (`Config::default()`)
2. TOML file (`$XDG_CONFIG_HOME/vox/config.toml`)
3. Environment variables (`VOX_SPEED`, `VOX_VOICE`, `VOX_MODEL_DIR`, `VOX_LOG_LEVEL`, `VOX_PORT`)

## Known Limitations

- **Audio contention in stdio mode**: each MCP client spawns its own vox process. Multiple stdio processes will fight over the mic/speaker. Use daemon mode (`--serve`) to share a single process across clients.
- **Model-backed tests are manual**: `examples/e2e_roundtrip.rs`, `examples/synth_lecture.rs` + `vox transcribe --reference`, and `scripts/profile.sh rtf` need the ~350 MB models; CI runs only model-free tests.
- **Long sessions**: sherpa-onnx 1.13 keeps VAD positions as int32 internally, so recreate the VAD before ~37 h of continuous audio.
- **Forced splits**: timestamp-bearing models use padded windows and midpoint word ownership at contiguous cuts. Moonshine has no word timestamps, so boundaries can still lose/repeat words. Validate Parakeet on real lectures before changing defaults.
