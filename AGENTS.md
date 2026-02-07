# Vox — Agent Instructions

## Project Overview

Vox is a lightweight voice MCP server (~1,200 lines of Rust) providing local text-to-speech (Kokoro) and speech-to-text (Whisper) via the MCP stdio transport. It runs as a subprocess per MCP client.

## Build & Test

```bash
cargo check          # type-check only
cargo test           # run all unit tests (47 tests across 6 modules)
cargo clippy -- -D warnings  # lint — must pass with zero warnings
```

All three must pass cleanly before submitting changes.

## Architecture

| Module | Purpose |
|--------|---------|
| `main.rs` | Entry point, config loading, model download, stdio/daemon startup |
| `server.rs` | MCP tool handlers (`say`, `listen`, `converse`), streaming TTS pipeline |
| `tts.rs` | Kokoro TTS engine wrapper, voice name → speaker ID resolution, sentence splitting |
| `audio.rs` | cpal-based mic capture and speaker playback, linear resampling |
| `stt.rs` | Whisper STT engine wrapper |
| `vad.rs` | Voice activity detection (silero) |
| `config.rs` | TOML config loading, env var overrides (`VOX_*` prefix), path resolution |
| `models.rs` | Model readiness checks and download/extraction |
| `error.rs` | `VoiceError` enum with `thiserror` derives |

### Transport Modes

- **Stdio** (default): `rmcp::transport::stdio()`. One process per MCP client.
- **Daemon** (`--serve [port]`): `StreamableHttpService` via rmcp's `transport-streamable-http-server` feature. Single process, models loaded once, multiple clients connect over HTTP/SSE. Factory closure creates a `VoiceMcpServer` per session with shared `Arc<Mutex<TtsEngine>>` and `Arc<Mutex<SttEngine>>`.

### Streaming TTS

Multi-sentence text is pipelined: a producer task synthesizes sentences sequentially and sends audio chunks through a channel, while a consumer task plays them back. This means the first sentence starts playing as soon as it's synthesized, while remaining sentences are synthesized in parallel with playback. Single-sentence text takes the simple path with no channel overhead.

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
3. Environment variables (`VOX_SPEED`, `VOX_VOICE`, `VOX_WHISPER_MODEL`, `VOX_MODEL_DIR`, `VOX_LOG_LEVEL`)

## Known Limitations

- **Audio contention in stdio mode**: each MCP client spawns its own vox process. Multiple stdio processes will fight over the mic/speaker. Use daemon mode (`--serve`) to share a single process across clients.
- **No integration tests**: unit tests cover logic but not actual TTS/STT inference (requires ~1GB of model files).
