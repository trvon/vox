# Vox — Agent Instructions

## Project Overview

Vox is a lightweight voice MCP server (~2,000 lines of Rust) providing local text-to-speech (Kokoro) and speech-to-text (Moonshine Base) via MCP. Supports both stdio transport (one process per client) and HTTP daemon mode (shared process, models loaded once).

## Build & Test

```bash
cargo check          # type-check only
cargo test           # run all unit tests (103 tests across 12 modules)
cargo clippy -- -D warnings  # lint — must pass with zero warnings
cargo bench                  # run Criterion benchmarks (DSP + TTS utilities)
```

The first three must pass cleanly before submitting changes. See [BENCHMARKS.md](BENCHMARKS.md) for benchmark details.

## Architecture

| Module | Purpose |
|--------|---------|
| `main.rs` | Entry point, config loading, model download, stdio/daemon startup |
| `cli.rs` | Clap CLI parser: daemon, config, download-models subcommands |
| `server.rs` | MCP tool handlers (`say`, `listen`, `converse`), streaming TTS pipeline |
| `tts.rs` | Kokoro TTS via direct `sherpa-rs-sys` FFI, streaming callback, voice resolution, sentence splitting |
| `audio.rs` | cpal capture/playback (batch + streaming ring buffer), Lanczos-3 resampling, DSP (HPF, noise gate, normalize) |
| `stt.rs` | Moonshine Base STT engine wrapper |
| `vad.rs` | Voice activity detection (silero) |
| `config.rs` | TOML config loading, env var overrides (`VOX_*` prefix), path resolution |
| `daemon.rs` | HTTP daemon lifecycle: daemonize, PID file, start/stop/status/log |
| `models.rs` | Model readiness checks and download/extraction |
| `lib.rs` | Public re-exports for benchmarks (`audio`, `config`, `error`, `tts`) |
| `error.rs` | `VoiceError` enum with `thiserror` derives |

### Transport Modes

- **Stdio** (default): `rmcp::transport::stdio()`. One process per MCP client.
- **Daemon** (`--serve [port]`): `StreamableHttpService` via rmcp's `transport-streamable-http-server` feature. Single process, models loaded once, multiple clients connect over HTTP/SSE. Factory closure creates a `VoiceMcpServer` per session with shared `Arc<Mutex<TtsEngine>>` and `Arc<Mutex<SttEngine>>`.

### Streaming TTS

Uses `SherpaOnnxOfflineTtsGenerateWithCallbackWithArg` from `sherpa-rs-sys` (direct FFI, bypassing the `sherpa-rs` KokoroTts wrapper). A C callback fires with audio chunks during synthesis, copies them into a `Vec<f32>`, and sends them through a `std::sync::mpsc` channel. A consumer task plays chunks through a cpal output stream backed by an `Arc<Mutex<VecDeque<f32>>>` ring buffer. Audio starts playing within the first callback — no waiting for full synthesis to complete.

**Callback convention**: sherpa-onnx returns `1 = continue, 0 = stop` (opposite of typical C 0=success).

## Code Conventions

- **Edition 2024** — uses `let` chains (`if let Ok(x) = ... && let Ok(y) = ...`)
- **Visibility**: use `pub(crate)` to expose private functions for testing rather than making them fully `pub`
- **Error enum**: `VoiceError` carries forward-looking variants (`ModelNotFound`, `Config`, `Timeout`) that are `#[allow(dead_code)]` until wired up
- **Clippy**: treat all warnings as errors (`-D warnings`). Fix collapsible-if, dead code, etc. rather than suppressing broadly
- **Tests**: inline `#[cfg(test)] mod tests` per module. Use `tempfile` for filesystem tests. No integration tests yet (would need model files)
- **`unsafe impl Send`**: `TtsEngine` and `CaptureHandle` have manual `Send` impls due to non-Send cpal/sherpa internals confined to dedicated threads

## Config Precedence

1. Compiled defaults (`Config::default()`)
2. TOML file (`$XDG_CONFIG_HOME/vox/config.toml`)
3. Environment variables (`VOX_SPEED`, `VOX_VOICE`, `VOX_MODEL_DIR`, `VOX_LOG_LEVEL`, `VOX_PORT`)

## Known Limitations

- **Audio contention in stdio mode**: each MCP client spawns its own vox process. Multiple stdio processes will fight over the mic/speaker. Use daemon mode (`--serve`) to share a single process across clients.
- **No integration tests**: unit tests cover logic but not actual TTS/STT inference (requires ~1GB of model files).
