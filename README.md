# Vox

Local voice MCP server with text-to-speech (Kokoro) and speech-to-text (Moonshine Base). All inference runs on-device — no API keys, no cloud.

Default TTS model: `kokoro-int8-multi-lang-v1_1` (smaller footprint, good quality).

## Tools

| Tool | Description |
|------|-------------|
| `say` | Queue text for ordered asynchronous playback; returns a speech id (`enqueue_say` is a compatibility alias) |
| `tts_queue_status` | Inspect current and pending speech |
| `tts_queue_clear` | Clear pending speech without interrupting current playback |
| `listen` | Record from microphone and transcribe |
| `converse` | Speak then listen (bidirectional) |
| `start_listening` | Start background mic capture with VAD+STT (queues to inbox) |
| `check_inbox` | Drain transcribed messages from background listener |
| `stop_listening` | Stop background listener, return remaining messages |
| `calibrate` | Run DSP calibration via genetic algorithm on live audio |
| `reset_dsp` | Reset DSP parameters to defaults |
| `reload_config` | Reload config and synchronize the selected STT/TTS models |
| `start_recording` | Start a long-form recording (lecture/meeting), transcribed incrementally with timestamps |
| `mark_recording` | Insert a section marker ("Q&A", "Part 2") at the current position |
| `recording_status` | State, captured vs transcribed time, lag, progress, dropped-audio gaps, STT speed |
| `stop_recording` | Finalize (or `cancel` remaining STT); audio is never deleted |
| `transcribe_file` | Transcribe an existing audio/video file as a background job |
| `list_recordings` | Recordings and file jobs, newest first |
| `get_transcript` | Page through timestamped segments with a cursor (works while recording) |
| `export_transcript` | Markdown drafting scaffold for blog posts/notes, or text / SRT / VTT / JSON |

Tools are focused operations over shared services, not separate lecture, meeting,
or blog-writing pipelines. `converse` composes speech and listening;
`get_transcript` reads paginated data, while `export_transcript` materializes an
artifact. CLI `vox tool …` commands are HTTP MCP adapters; local
`vox transcribe` uses the same recording/transcription service without a daemon
or TTS model dependency.

Only one microphone capture can run at a time within a process. Listening,
recording, and calibration reject competing capture requests. Use the daemon
to share this policy across clients; separate stdio processes are not coordinated.
`stop_listening` waits for final capture/transcription instead of returning after
an arbitrary delay.

## Lectures, meetings and blog posts

Run vox as a daemon (`vox daemon start`) so a recording outlives the MCP client that started it. Then, from any agent harness:

1. `start_recording {title}` → `recording_id`. Audio goes to `audio.wav`; speech is segmented by VAD and transcribed in the background.
2. Optionally `mark_recording {label}` at topic changes, and `recording_status` to watch progress.
3. `stop_recording` waits for the remaining segments to be transcribed.
4. `export_transcript {format: "markdown"}` returns sections (from markers, else every ~5 min), paragraphs split at pauses, and a `[mm:ss]` stamp on every paragraph. Every quote can be traced back to the audio. The calling LLM turns this scaffold into prose. `get_transcript` pages through the raw segments for long recordings.

Existing recordings work the same way via `transcribe_file` (WAV natively; mp3/m4a/video via `ffmpeg`), or offline with no daemon:

```bash
vox transcribe lecture.m4a --format markdown > notes.md     # also: text, srt, vtt, json
vox transcribe lecture.wav --reference lecture.txt          # prints WER + real-time factor
```

**Storage** is under `$XDG_DATA_HOME/vox/recordings/<id>/`; change it with `recordings_dir` or `VOX_RECORDINGS_DIR`. Each recording directory holds:

- `manifest.json`, replaced atomically;
- `transcript.jsonl`, append-only with one segment per line;
- `markers.jsonl`;
- `audio.wav`, whose header is rewritten every 5 s, so a crash leaves a playable file;
- `owner.lock`: an OS-held liveness lock (released on completion or process exit);
- any exports.

Status reads do not rewrite another process's jobs. After owner exit, unfinished
jobs are reported as `interrupted` without altering their stored manifest.
Capture chunks carry absolute sample positions: overflow inserts silence at the
actual gap, including dropped tail windows. The live STT queue is bounded, and
storage/STT failures produce `failed` jobs rather than successful incomplete
transcripts. Already-saved audio is never deleted.

Moonshine Base remains the default. Optional `stt_model` values are
`parakeet-tdt-v2` and `parakeet-tdt-v3` (also `VOX_STT_MODEL`). Download the
selected model with `vox download-models`, then `vox tool reload-config` to
synchronize a running daemon. STT replacement is serialized with decoding;
segments already in progress finish on the previous engine. Parakeet timestamp
accuracy and forced-boundary quality still need real-lecture evaluation before
changing defaults.

If the process dies mid-job, the job reports `interrupted`. Everything transcribed so far stays readable, and the WAV can be re-run with `transcribe_file`.

## Install

### From source

#### Build prerequisites (Linux)

On Linux, `cpal` uses ALSA by default. You must have `pkg-config` and the ALSA development headers installed or builds will fail with an `alsa-sys` / `alsa.pc` error.

Debian/Ubuntu:

```bash
sudo apt update
sudo apt install -y pkg-config libasound2-dev
```

Fedora:

```bash
sudo dnf install -y pkgconf-pkg-config alsa-lib-devel
```

Arch:

```bash
sudo pacman -S --needed pkgconf alsa-lib
```

Alpine:

```bash
sudo apk add pkgconf alsa-lib-dev
```

#### sherpa-onnx linking

Vox uses the official k2-fsa [`sherpa-onnx`](https://crates.io/crates/sherpa-onnx) crate. Its default `static` feature downloads a prebuilt archive at build time and links sherpa-onnx and onnxruntime into the `vox` binary, so there are no shared libraries to ship. For offline builds, point `SHERPA_ONNX_ARCHIVE_DIR` or `SHERPA_ONNX_LIB_DIR` at a local copy. The crate is pinned to **1.13.x**: version 2.0 removed espeak-ng, which makes Kokoro silently skip numbers and words missing from its lexicon.

```bash
cargo install --path . --locked
# Linux: enable autostart after a manual cargo install
vox service install
```

#### Optional: guided setup

`setup.sh` builds a release binary, installs it to `~/.local/bin` by default,
downloads models, and enables a per-user daemon: systemd on Linux, launchd on
macOS. Run it as your desktop user, not with `sudo`.

```bash
./setup.sh
# Custom permanent binary location:
./setup.sh "$HOME/.local/bin"
# Binary/models only (containers, CI, or manual lifecycle management):
VOX_NO_SERVICE=1 ./setup.sh
```

On Linux without an available systemd user manager, setup leaves the binary and
models installed and prints the command to enable autostart from a login session.
Persistent global npm installs (`npm install -g vox-mcp`) also attempt to enable
the Linux service; `VOX_NO_SERVICE=1` opts out. Local npm installs and temporary
`npx` invocations never install a login service.

### Model download options

```bash
vox download-models                              # downloads configured kokoro_model
vox download-models --kokoro-model int8-v1_1    # override to int8 for this run
vox download-models --kokoro-model fp32-v1_1    # larger fp32 variant
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

### Linux autostart (systemd user service)

```bash
# For an already-installed binary; enable at login and start/restart now:
vox service install
# Enable at login without starting now:
vox service install --no-start
systemctl --user status vox.service
journalctl --user -u vox.service -f
systemctl --user restart vox.service
systemctl --user stop vox.service
# Disable autostart and remove the generated unit; keep models/recordings:
vox service uninstall
```

If a daemon was started manually, stop it with `vox daemon stop` before handing
control to systemd. Once managed, use `systemctl --user` for lifecycle commands
and the journal for logs, rather than `vox daemon start/restart/log`.

The generated unit is `$XDG_CONFIG_HOME/systemd/user/vox.service` (normally
`~/.config/systemd/user/vox.service`). It runs the installing binary's absolute
path with `daemon start --foreground`, restarts on failure, and allows up to five
minutes for graceful recording/transcription shutdown. It starts with your user
manager at login, not as a privileged system service. No microphone capture starts
until a client requests it. Initial startup downloads missing models if needed.
`active` means the process is running, not that model initialization is finished;
wait for `Vox MCP daemon listening` in the journal before connecting a client.

Keep the binary at a permanent path. Re-run `vox service install` after moving
it; npm users should run `vox-mcp service uninstall` before uninstalling the
package. Custom or symlinked units are never overwritten or removed. Put overrides
(e.g. `Environment=VOX_PORT=8080` under `[Service]`) in
`systemctl --user edit vox.service`; the service reads your TOML config and user
manager environment, not the installer's temporary shell environment. Logout
normally stops the user manager; lingering is an explicit administrator/user
choice, not enabled by the installer.

### Pi harness

Use the shared daemon when more than one Pi session may use Vox. Linux installers
can enable it with `vox service install`; otherwise start it with
`vox daemon start`. The daemon remains available when a harness session
disconnects. Stop managed daemons with `systemctl --user stop vox.service`, or
manually started ones with `vox daemon stop`.

With Pi's built-in MCP support:

```bash
pi mcp add vox --url http://127.0.0.1:3030/mcp
pi mcp list
```

Run `/reload` in an existing session after changing configuration from the shell.
If Pi uses `pi-mcp-adapter` instead, register the running daemon from the session
without reloading:

```js
mcp({ action: "install", server: "vox", url: "http://127.0.0.1:3030/mcp" })
mcp({ server: "vox" })
mcp({ tool: "vox_list_recordings", args: {} })
```

This stores a user-level connection, not project/session artifacts. Listing jobs
and transcribing existing files do not activate the microphone or speakers.
`listen`, `start_recording`, and calibration capture audio; `say` plays audio.

## Configuration

Settings are resolved in order: compiled defaults, then TOML file, then env vars.

| Env var | TOML key | Default | Description |
|---------|----------|---------|-------------|
| `VOX_VOICE` | `voice` | `af_heart` | Default TTS voice |
| `VOX_SPEED` | `speed` | `1.4` | Speech rate multiplier |
| `VOX_MODEL_DIR` | `model_dir` | `$XDG_DATA_HOME/vox/models` | Model storage path |
| `VOX_KOKORO_MODEL` | `kokoro_model` | `int8-v1_1` | Kokoro model variant (`int8-v1_1` or `fp32-v1_1`) |
| `VOX_LOG_LEVEL` | `log_level` | `info` | Log filter |
| `VOX_PORT` | — | `3030` | Daemon mode port (env/CLI only) |

TOML config location: `$XDG_CONFIG_HOME/vox/config.toml` (e.g. `~/.config/vox/config.toml`).

Manage config from the CLI:

```bash
vox config get            # show all values
vox config get voice      # show a single value
vox config set kokoro_model fp32-v1_1
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

Synthesis uses sherpa-onnx's `generate_with_config` callback to stream audio chunks to the speaker as they are generated, so playback starts during synthesis. A ring buffer bridges the callback thread and the cpal output stream.

### STT (Capture)

Microphone input is processed through a DSP pipeline before transcription:

1. **High-pass filter**: a Butterworth filter at 200 Hz for interactive `listen`, or a gentle 80 Hz for recordings. It removes rumble and speaker bleed.
2. **Resample** to 16 kHz with a stateful, anti-aliased Lanczos-3 resampler:
   - the kernel is widened when downsampling, so 8–24 kHz content can't fold into the speech band;
   - integer phase tracking makes output identical however the audio is chunked, with no drift over hours.
3. **Noise gate** (RMS threshold): interactive `listen` only. Recordings keep the raw audio.
4. **Silero VAD**: detects speech boundaries. Segments are capped at 20–25 s, and recordings add 250 ms of padding to each side so word edges aren't clipped.
5. **Gain**: peak normalization for `listen`; bounded gain (at most +20 dB) for recordings.
6. **Moonshine Base**: on-device speech-to-text. Runaway repetitions (a known failure on noise) are collapsed.

## Profiling

```bash
scripts/profile.sh rtf      # end-to-end real-time factor + WER on a synthetic lecture fixture
scripts/profile.sh perf     # perf profile of `vox transcribe` (symbols via the `profiling` cargo profile)
scripts/profile.sh bench    # Criterion micro-benchmarks
```

## Architecture

See [AGENTS.md](AGENTS.md) for architecture details, code conventions, and developer docs.

See [BENCHMARKS.md](BENCHMARKS.md) for benchmark documentation and performance data.

## License

MIT

## Troubleshooting

### `alsa-sys` build failure on Linux

If you see errors like:

```
failed to run custom build command for `alsa-sys`
The system library `alsa` required by crate `alsa-sys` was not found.
The file `alsa.pc` needs to be installed
```

Install ALSA development headers + `pkg-config`:

Debian/Ubuntu:

```bash
sudo apt install -y pkg-config libasound2-dev
```

Fedora:

```bash
sudo dnf install -y pkgconf-pkg-config alsa-lib-devel
```

Arch:

```bash
sudo pacman -S --needed pkgconf alsa-lib
```

Alpine:

```bash
sudo apk add pkgconf alsa-lib-dev
```

Then verify:

```bash
pkg-config --libs --cflags alsa
```
