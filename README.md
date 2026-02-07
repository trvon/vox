# Vox

Local voice MCP server with text-to-speech (Kokoro) and speech-to-text (Moonshine Base). All inference runs on-device — no API keys, no cloud.

## Tools

| Tool | Description |
|------|-------------|
| `say` | Speak text aloud through speakers |
| `listen` | Record from microphone and transcribe |
| `converse` | Speak then listen (bidirectional) |

## Install

```bash
git clone https://github.com/trvon/vox && cd vox
./setup.sh            # builds, installs to ~/.local/bin, downloads models (~1GB)
```

`setup.sh` builds a release binary, copies it and any shared libraries to `~/.local/bin`, downloads models, and on macOS installs a launchd plist so the daemon starts at login.

Or manually:

```bash
cargo install --path .
```

## MCP Configuration

### Stdio mode (default)

Each client spawns its own `vox` process. Simple, no setup.

```json
{
  "mcpServers": {
    "vox": {
      "command": "~/.local/bin/vox",
      "args": []
    }
  }
}
```

### Daemon mode

A single long-lived process serves multiple clients. Models loaded once, shared across sessions.

```bash
vox daemon start              # backgrounds, listens on 127.0.0.1:3030
vox daemon start -p 8080      # custom port
vox daemon start --foreground  # stay in foreground (useful for debugging)
vox daemon status              # check if running
vox daemon stop                # graceful shutdown via SIGTERM
vox daemon restart             # stop + start
vox daemon log                 # tail the daemon log
```

Point clients at the daemon:

```json
{
  "mcpServers": {
    "vox": {
      "url": "http://localhost:3030/mcp"
    }
  }
}
```

## Configuration

Settings are resolved in order: compiled defaults, then TOML file, then env vars.

| Env var | TOML key | Default | Description |
|---------|----------|---------|-------------|
| `VOX_VOICE` | `voice` | `af_heart` | Default TTS voice |
| `VOX_SPEED` | `speed` | `1.4` | Speech rate multiplier |
| `VOX_MODEL_DIR` | `model_dir` | `$XDG_DATA_HOME/vox/models` | Model storage path |
| `VOX_LOG_LEVEL` | `log_level` | `info` | Log filter |
| `VOX_PORT` | — | `3030` | Daemon mode port (env/CLI only) |

TOML config location: `$XDG_CONFIG_HOME/vox/config.toml` (e.g. `~/.config/vox/config.toml`).

Manage config from the CLI:

```bash
vox config get            # show all values
vox config get voice      # show a single value
vox config set speed 1.5  # persist to config.toml
vox config path           # print config file location
```

<details>
<summary>Available voices (26)</summary>

**American female** (`af_*`): `af_heart` (default), `af_alloy`, `af_aoede`, `af_bella`, `af_jessica`, `af_kore`, `af_nicole`, `af_nova`, `af_river`, `af_sarah`, `af_sky`

**American male** (`am_*`): `am_adam`, `am_echo`, `am_eric`, `am_liam`, `am_michael`, `am_onyx`, `am_puck`, `am_santa`

**British female** (`bf_*`): `bf_alice`, `bf_emma`, `bf_lily`

**British male** (`bm_*`): `bm_daniel`, `bm_fable`, `bm_george`, `bm_lewis`

</details>

## Audio Pipeline

### TTS (Streaming)

Synthesis uses the sherpa-onnx C callback API (`SherpaOnnxOfflineTtsGenerateWithCallbackWithArg`) to stream audio chunks directly to the speaker as they are generated. Audio starts playing during synthesis — no waiting for the full utterance to complete. A ring buffer bridges the C callback thread and the cpal output stream.

### STT (Capture)

Microphone input is processed through a DSP pipeline before transcription:

1. **High-pass filter** (200Hz Butterworth) — removes low-frequency speaker bleed and room rumble
2. **Noise gate** (RMS threshold) — zeros quiet windows to prevent false VAD triggers from ambient noise
3. **Resample** to 16kHz (Lanczos-3 sinc interpolation)
4. **Silero VAD** — detects speech boundaries, discards silence
5. **Peak normalization** — scales quiet audio to full range for better STT accuracy
6. **Moonshine Base** — on-device speech-to-text

## Architecture

See [AGENTS.md](AGENTS.md) for architecture details, code conventions, and developer docs.

See [BENCHMARKS.md](BENCHMARKS.md) for benchmark documentation and performance data.

## License

MIT
