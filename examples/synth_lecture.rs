//! Synthesize a multi-paragraph "lecture" with Kokoro for end-to-end tests and
//! profiling of the long-form transcription pipeline.
//!
//! ```bash
//! cargo run --release --example synth_lecture -- /tmp/lecture.wav [repeat] [sentence_gap_secs] [paragraph_gap_secs]
//! vox transcribe /tmp/lecture.wav --format markdown
//! ```
//!
//! Writes `<out>.wav` (Kokoro's native rate, so the resampler is exercised)
//! and `<out>.txt` with the reference text for WER checks. Paragraphs are
//! separated by 3 s of silence, sentences by 0.5 s, with low-level noise so
//! the VAD sees a realistic floor.

use vox::config::Config;
use vox::tts::{TtsEngine, split_sentences};

const PARAGRAPHS: &[&str] = &[
    "Welcome to today's lecture on signal processing. We will start with sampling. \
     The sampling theorem tells us how fast we must measure a signal to capture it completely.",
    "Aliasing happens when we sample too slowly. High frequencies fold back and pretend to be low ones. \
     That is why every good resampler applies a low pass filter before it throws samples away.",
    "Next, let us talk about voice activity detection. A detector decides which parts of a recording contain speech. \
     Good detectors ignore background noise but keep quiet words at the start and end of a sentence.",
    "Finally, a note on transcription. Long recordings are split into short segments at natural pauses. \
     Each segment keeps its timestamp, so every quote in your notes can be traced back to the audio.",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let out = args.next().unwrap_or_else(|| "lecture.wav".into());
    let repeat: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    // Short gaps (e.g. 0.15 / 0.2) give continuous speech that forces VAD splits.
    let sentence_gap: f32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(0.5);
    let paragraph_gap: f32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(3.0);
    let out = std::path::Path::new(&out);

    let config = Config::load();
    let mut tts = TtsEngine::new(&config)?;
    let rate = tts.sample_rate();
    let silence = |secs: f32| vec![0.0f32; (secs * rate as f32) as usize];

    let mut audio: Vec<f32> = silence(1.0);
    let mut reference = Vec::new();
    let t0 = std::time::Instant::now();
    for r in 0..repeat {
        for (i, para) in PARAGRAPHS.iter().enumerate() {
            let voice = if (r + i) % 2 == 0 {
                "af_heart"
            } else {
                "am_michael"
            };
            for sentence in split_sentences(para) {
                let a = tts.synthesize(sentence, Some(voice), Some(1.0))?;
                audio.extend_from_slice(&a.samples);
                audio.extend(silence(sentence_gap));
            }
            reference.push(*para);
            audio.extend(silence(paragraph_gap));
        }
    }
    let synth_s = t0.elapsed().as_secs_f64();

    // Deterministic low-level noise floor (~ -50 dBFS).
    let mut state = 0x2545F4914F6CDD1Du64;
    for s in audio.iter_mut() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *s += ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.006;
    }

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(out, spec)?;
    for s in &audio {
        w.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
    }
    w.finalize()?;
    std::fs::write(out.with_extension("txt"), reference.join("\n\n"))?;

    let dur = audio.len() as f64 / rate as f64;
    eprintln!(
        "wrote {} ({dur:.1}s @ {rate} Hz); synthesis {synth_s:.2}s (RTF {:.3})",
        out.display(),
        synth_s / dur
    );
    Ok(())
}
