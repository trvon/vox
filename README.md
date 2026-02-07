# Vox

Local voice MCP server with text-to-speech (Kokoro) and speech-to-text (Moonshine Base). All inference runs on-device — no API keys, no cloud.

## Tools

| Tool | Description |
|------|-------------|
| `say` | Speak text aloud through speakers |
| `listen` | Record from microphone and transcribe |
| `converse` | Speak then listen (bidirectional) |
| `start_listening` | Start background mic capture with VAD+STT (queues to inbox) |
| `check_inbox` | Drain transcribed messages from background listener |
| `stop_listening` | Stop background listener, return remaining messages |
| `calibrate` | Run DSP calibration via genetic algorithm on live audio |
| `reset_dsp` | Reset DSP parameters to defaults |

## Install

### npm (recommended)

```bash
npx vox-mcp
# or
npm install -g vox-mcp
```

### From source

```bash
cargo install --path .
```

## MCP Configuration

### npm (recommended)

```json
{
  "mcpServers": {
    "vox": {
      "command": "npx",
      "args": ["-y", "vox-mcp"]
    }
  }
}
```

### Stdio mode (manual install)

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
vox config reset-dsp      # reset DSP parameters to defaults
```

<details>
<summary>Available voices (26)</summary>

**American female** (`af_*`): `af_heart` (default), `af_alloy`, `af_aoede`, `af_bella`, `af_jessica`, `af_kore`, `af_nicole`, `af_nova`, `af_river`, `af_sarah`, `af_sky`

**American male** (`am_*`): `am_adam`, `am_echo`, `am_eric`, `am_liam`, `am_michael`, `am_onyx`, `am_puck`, `am_santa`

**British female** (`bf_*`): `bf_alice`, `bf_emma`, `bf_lily`

**British male** (`bm_*`): `bm_daniel`, `bm_fable`, `bm_george`, `bm_lewis`

</details>

## DSP Calibration

The DSP parameters (high-pass filter cutoff, noise gate threshold, gate window size, normalization threshold) can be auto-tuned to your mic and room using a genetic algorithm:

```bash
vox calibrate              # record speech + silence, optimize, save to config
vox calibrate --dry-run    # same but don't save (preview only)
vox config reset-dsp       # revert DSP params to defaults if calibration went wrong
```

The calibrator records ~10s of speech and ~5s of silence, then runs a GA (40 population x 30 generations) to maximize SNR while preserving speech quality. Results are saved to `config.toml`.

Calibration is also available as an MCP tool (`calibrate`), so an LLM can trigger it during a session. It defaults to `dry_run=true` for safety — set `dry_run=false` to persist results. Use `reset_dsp` to revert to defaults if calibration produces bad results.

| Flag | Default | Description |
|------|---------|-------------|
| `--speech-secs` | `10` | Seconds of speech to record |
| `--silence-secs` | `5` | Seconds of silence to record |
| `--population` | `40` | GA population size |
| `--generations` | `30` | Number of GA generations |
| `--dry-run` | off | Print results without saving |

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
