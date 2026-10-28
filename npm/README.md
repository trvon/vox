# vox-mcp

Local voice MCP server with text-to-speech (Kokoro) and speech-to-text (Moonshine). All inference runs on-device — no API keys, no cloud.

## Install

```bash
npx vox-mcp
# or
npm install -g vox-mcp
```

On Linux, a persistent global install attempts to install and enable a systemd
**user** service at login. It runs the downloaded native binary directly (not a
Node wrapper), in the foreground, and restarts on failure. Run as your desktop
user, not with `sudo`. Local installs and temporary `npx` runs do not enable
services.

```bash
VOX_NO_SERVICE=1 npm install -g vox-mcp  # opt out of autostart
vox-mcp service install                # retry from a login session if needed
systemctl --user status vox.service
journalctl --user -u vox.service -f
vox-mcp service uninstall              # before removing the global package
```

Stop a manually launched daemon with `vox-mcp daemon stop` before installing the
service. If systemd/the user manager is unavailable, the binary installation still
succeeds with a warning. Initial daemon startup downloads missing models. Once
managed, use `systemctl --user` to start, stop, or restart it. Uninstalling the
service retains models and recordings.

For a shared daemon, point clients at `http://127.0.0.1:3030/mcp` instead of
starting a separate stdio server per client:

```json
{
  "mcpServers": {
    "vox": { "url": "http://127.0.0.1:3030/mcp" }
  }
}
```

## MCP Configuration (stdio, no shared daemon)

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

## Tools

| Tool | Description |
|------|-------------|
| `say` | Speak text aloud through speakers |
| `listen` | Record from microphone and transcribe |
| `converse` | Speak then listen (bidirectional) |
| `start_listening` | Start background mic capture |
| `check_inbox` | Drain transcribed messages from background listener |
| `stop_listening` | Stop background listener |

## More Info

See the [main repository](https://github.com/trvon/vox) for full documentation, configuration options, and daemon mode.
