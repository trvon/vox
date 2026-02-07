# vox-mcp

Local voice MCP server with text-to-speech (Kokoro) and speech-to-text (Moonshine). All inference runs on-device — no API keys, no cloud.

## Install

```bash
npx vox-mcp
# or
npm install -g vox-mcp
```

## MCP Configuration

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
