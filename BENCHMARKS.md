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

## Notes

- Results are saved to `target/criterion/` with HTML reports.
- The full DSP chain (HPF + noise gate + normalize) on 1s of 16kHz audio takes ~52 µs total — negligible compared to TTS synthesis time.
