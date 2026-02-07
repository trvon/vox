# Vox

Local voice MCP server with text-to-speech (Kokoro) and speech-to-text (Whisper). All inference runs on-device — no API keys, no cloud.

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

Or manually:

```bash
cargo build --release
cp target/release/vox ~/.local/bin/
cp target/release/*.dylib ~/.local/bin/ 2>/dev/null  # macOS shared libs
./vox --download-models
```

## MCP Configuration

### Stdio mode (default)

Each client spawns its own process. Simple, no setup.

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

Start the daemon:

```bash
vox --serve           # listens on 127.0.0.1:3030
vox --serve 8080      # custom port
VOX_PORT=9000 vox --serve  # or via env var
```

Point clients at it:

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
| `VOX_SPEED` | `speed` | `1.0` | Speech rate multiplier |
| `VOX_WHISPER_MODEL` | `whisper_model` | `tiny` | Whisper size: `tiny`, `base`, `small` |
| `VOX_MODEL_DIR` | `model_dir` | `$XDG_DATA_HOME/vox/models` | Model storage path |
| `VOX_LOG_LEVEL` | `log_level` | `info` | Log filter |
| `VOX_PORT` | — | `3030` | Daemon mode port |

TOML config location: `$XDG_CONFIG_HOME/vox/config.toml` (e.g. `~/.config/vox/config.toml`).

<details>
<summary>Available voices</summary>

**American female** (`af_*`): `af_heart` (default), `af_alloy`, `af_aoede`, `af_bella`, `af_jessica`, `af_kore`, `af_nicole`, `af_nova`, `af_river`, `af_sarah`, `af_sky`

**American male** (`am_*`): `am_adam`, `am_echo`, `am_eric`, `am_liam`, `am_michael`, `am_onyx`, `am_puck`, `am_santa`

**British female** (`bf_*`): `bf_alice`, `bf_emma`, `bf_lily`

**British male** (`bm_*`): `bm_daniel`, `bm_fable`, `bm_george`, `bm_lewis`

</details>

## Architecture

See [AGENTS.md](AGENTS.md) for architecture details, code conventions, and developer docs.

## License

MIT
