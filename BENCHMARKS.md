# Benchmarks

Vox uses [Criterion](https://bheisler.github.io/criterion.rs/) for micro-benchmarks of the audio DSP pipeline and TTS utilities.

## Running

```bash
# Run all benchmarks
cargo bench

# Run a specific group
cargo bench -- resample
cargo bench -- highpass_filter
cargo bench -- noise_gate
cargo bench -- peak_normalize

# Run TTS utility benchmarks (sentence splitting, voice resolution)
cargo bench --bench tts_bench
```

## Results

Measured on Apple M4 Max. All DSP is CPU-only, single-threaded.

### `audio_bench`

| Group | Benchmark | Time | Description |
|-------|-----------|------|-------------|
| `resample` | `48k_to_16k` | 166 µs | Lanczos-3 downsample 1s of 48kHz → 16kHz |
| `resample` | `8k_to_16k` | 170 µs | Lanczos-3 upsample 1s of 8kHz → 16kHz |
| `resample` | `passthrough_16k` | 1.2 µs | Same-rate passthrough (memcpy baseline) |
| `highpass_filter` | `1s_16khz` | 46 µs | 200Hz Butterworth HPF on 1s at 16kHz |
| `highpass_filter` | `1s_48khz` | 137 µs | 200Hz Butterworth HPF on 1s at 48kHz |
| `noise_gate` | `quiet_512` | 440 ns | RMS gate on 512-sample quiet window (triggers zeroing) |
| `noise_gate` | `loud_512` | 411 ns | RMS gate on 512-sample loud window (no-op path) |
| `peak_normalize` | `quiet_1s` | 5.8 µs | Normalize 1s of quiet audio (peak < 0.5) |
| `peak_normalize` | `loud_1s_noop` | 4.6 µs | Normalize 1s of loud audio (no-op, peak > 0.5) |
| `peak_normalize` | `quiet_5s` | 25 µs | Normalize 5s of quiet audio |

### `tts_bench`

| Group | Benchmark | Time | Description |
|-------|-----------|------|-------------|
| `split_sentences` | `1_sentence` | 23 ns | Single short sentence |
| `split_sentences` | `5_sentences` | 92 ns | Five sentences |
| `split_sentences` | `paragraph` | 175 ns | Long paragraph with 6 sentences |
| `voice_resolution` | `known_voice` | 3.0 ns | Look up first voice in table |
| `voice_resolution` | `last_voice` | 31 ns | Look up last voice (linear scan, worst case) |
| `voice_resolution` | `numeric_fallback` | 11 ns | Numeric string fallback |
| `voice_resolution` | `unknown_voice` | 11 ns | Unknown voice (full scan + parse failure) |

### `pipeline_bench` and end-to-end (2026-10-27)

Measured on a 16-thread Intel hybrid laptop (Linux) **under heavy background load (load average 14–20)**, so treat the absolute numbers as upper bounds.

| Benchmark | Time | Notes |
|-----------|------|-------|
| `streaming_resample_1s/48000_to_16k_10ms_chunks` | 2.8 ms | Stateful anti-aliased resampler (18 taps), fed in 10 ms callbacks: ~0.3% of one core |
| `streaming_resample_1s/44100_to_16k_10ms_chunks` | 2.9 ms | Same, non-integer ratio |
| `transcript_3h/paragraphs_sections_markdown` | 179 µs | 2,000 segments, 9 markers: layout + markdown for a 3-hour lecture |
| `transcript_3h/srt` | 1.8 ms | SRT for 2,000 segments |
| `wer_2800_words` | 31 ms | O(n·m) word Levenshtein |

The old `resample/48k_to_16k` figure (166 µs, M4 Max) was for a 6-tap kernel with no anti-aliasing. The anti-aliased kernel widens to 18 taps when downsampling 3:1. On the machine above it now measures 2.5 ms.

**End-to-end** (`scripts/profile.sh rtf`, 76.5 s synthetic lecture, release build):

| Configuration | WER | STT RTF | Notes |
|---|---|---|---|
| Moonshine Base, fixed VAD, 1 STT thread | 6.2% | ~0.05 | Default since this change |
| Same, VAD `max_speech_duration` 0.5 s (old effective default) | 7.75% | — | ~10% of speech dropped; duplicated words at forced splits |
| 4 STT threads | 6.2% | ~0.06–0.09 | 10–15× more CPU time; ~80% of samples in ORT `SpinPause` |

### Follow-up correctness smoke test (2026-10-27)

The lifecycle/storage hardening was also exercised with a **freshly generated**
76.5 s lecture from `examples/synth_lecture.rs`, using the debug build under
heavy host load (observed load average ~20.5). Moonshine Base with one STT
thread produced 12 segments from 50.3 s of detected speech:

- WER: **8.53%** (11 edits / 129 reference words).
- File pipeline wall time: 14.51 s (whole-audio RTF 0.190).
- Reported STT time: 10.02 s (speech RTF 0.199).
- Audio SHA-256: `fbc29e7228d7d4867a2d8780122c2ef67254db6349f23d157f121b45b94ffa0e`.
- Reference SHA-256: `a65a46071a26a9b7ff8e9e6b1fcd15047208121cbc2908e6d221de7a60d4215d`.

This verifies the file pipeline, not a performance improvement or a reproduction
of the historical 6.2% row above. The earlier audio fixture was not available for
a same-waveform comparison, and debug/release timing is not comparable. Freeze
and retain the audio/reference pair before using WER as a regression gate.
The smoke test also verified silence handling with STT/VAD-only assets and
isolated cross-session CLI/MCP status, paging, stop, list, Origin/Host rejection,
and prompt daemon shutdown. No microphone capture or speaker playback was used.

## Profiling

- `scripts/profile.sh perf [file]` records `vox transcribe` with `perf -e task-clock`. The default `cycles` event fails on hybrid P/E-core CPUs. Call graphs need `kernel.perf_event_paranoid <= 1`; otherwise the script falls back to a flat profile.
- The `profiling` cargo profile keeps release optimizations with debug symbols and no stripping.
- `scripts/profile.sh samply [file]` gives a Firefox-profiler UI if `samply` is installed.
- Per-job STT speed is always available: `recording_status` / `list_recordings` report `stt_rtf`, `speech_ms` and `stt_compute_ms`.

## Bounded polyphase optimization (2026-10-28)

Measured on the same i7-1260P Linux host, with Rust 1.95 release settings and no
native-CPU flags. The baseline is commit `ea180ee`; the optimization is on the
separate `perf/polyphase-resampler` branch. These measurements compare against a
fresh retained baseline, **not** the historical M4 or heavily loaded numbers
above. CPU affinity/governor were not changed.

The resampler now caches raw f64 weights and their sums for each rational phase,
then uses contiguous input for interior dot products. Accumulation order, final
division, edge extension, sample positions, and scalar LUT fallback are retained.
No unsafe SIMD, model quantization changes, or sherpa/ONNX rebuild is involved.
The coefficient array is capped at 8,192 f64 values (64 KiB), plus per-phase sums;
unusual ratios fall back rather than allocate an unbounded phase table.

### Streaming cost

One second of audio in 10 ms chunks, including construction/flush/destruction.
Criterion means from 60 samples, 2 s warmup and 4 s measurement; the optimized
range covers two independent runs:

| Conversion | Before | After (two runs) | Speedup |
| --- | ---: | ---: | ---: |
| 48,000 → 16,000 Hz | 1.484 ms | 0.183–0.184 ms | 8.1× |
| 44,100 → 16,000 Hz | 1.489 ms | 0.199–0.222 ms | 6.7–7.5× |

The separate `resampler_setup_and_drop` group makes table-construction cost
visible: about 0.12 µs at 48/24 kHz input and 13.7 µs at 44.1 kHz, once per
resampler instance. The 44.1 → 16 kHz table uses 23.75 KiB for weights and sums.
Benchmark load averages were approximately 1.0 before and 2.8–3.7 afterward;
these are same-host measurements, not portable speed guarantees.

```bash
# Before modifying audio.rs, retain a named baseline:
cargo bench --bench pipeline_bench -- streaming_resample_1s \
  --save-baseline pre-polyphase-ea180ee --sample-size 60 --warm-up-time 2 --measurement-time 4
# After the change:
cargo bench --bench pipeline_bench -- streaming_resample_1s \
  --baseline pre-polyphase-ea180ee --sample-size 60 --warm-up-time 2 --measurement-time 4
cargo bench --bench pipeline_bench -- resampler_setup_and_drop
```

### Whole-file behavior

Five alternating baseline/optimized process pairs transcribed the **same retained
76.5 s fixture**, with Moonshine Base and one STT thread. The order reverses on
alternate pairs. Process wall time includes model loading; Python `wait4`
accounts for per-process CPU/RSS. Host load varied from 7.6 to 10.4 during these
runs.

- Every run produced the same 12 segments and identical `transcript.jsonl` bytes
  (SHA-256 `5de219a5a803628263c2d82e02b39f128c57e11f97ffc363a10dd7ce6aeb7131`).
- WER remained **8.53%**, 11 edits / 129 reference words. This is a correctness
  check on one synthetic fixture, not real-lecture acceptance evidence.
- Median process wall time: **3.354 s before, 3.324 s after**. Median paired wall
  change was only **−0.25%**, with pairs ranging from −1.94% to +2.97%:
  **no reliable end-to-end latency improvement is demonstrated**.
- Median process user CPU: 2.956 s before, 2.901 s after. These few runs do not
  establish a general battery or memory improvement.
- Separate one-thread profiles attributed approximately 2.03% of sampled CPU to
  resampler `compute` before, 0.27% afterward. FMA/AVX-VNNI neural kernels still
  dominate; thread spinning is no longer a dominant cost.

Validation: 368 model-free Rust tests passed (one opt-in model test ignored),
all 19 release invariant tests passed with `PROPTEST_CASES=2560`, and all 29 audio
regressions passed in release mode. Cached/scalar comparison tests cover 36 rate
pairs, five input lengths and three chunk sizes, plus uncommon-rate and
allocation-bound regressions.

Raw measurements, retained binaries/source, fixture copies, perf reports and
paired logs are under
`~/.local/state/vox/optimization/polyphase-20261028T001911Z/` on the measurement
host. Audio/reference hashes are the same as the correctness fixture above.

## Notes

- Results are saved to `target/criterion/` with HTML reports.
- The full DSP chain (HPF + noise gate + normalize) on 1s of 16kHz audio takes ~52 µs total — negligible compared to TTS synthesis time.
