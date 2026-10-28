//! End-to-end TTS → VAD → STT round trip with real models.
//!
//! Run: `cargo run --release --example e2e_roundtrip [-- "custom sentence" ...]`
//!
//! Requires models in the configured model dir (`vox download-models`).
//! Synthesizes each sentence with Kokoro (batch and streaming), resamples to
//! 16 kHz, segments with Silero VAD, transcribes with Moonshine, and reports
//! synthesis/STT real-time factors and word error rate against the input.
//! Also serves as a profiling target (e.g. `perf record`, `heaptrack`).

use std::sync::mpsc;
use std::time::Instant;

use vox::audio;
use vox::config::Config;
use vox::stt::SttEngine;
use vox::tts::{TtsEngine, resolve_voice_id};
use vox::vad::{VadOptions, VadSession};

const STT_RATE: u32 = 16000;
const VAD_WINDOW: usize = 512;

const DEFAULT_SENTENCES: &[&str] = &[
    "The quick brown fox jumps over the lazy dog. Voice assistants should respond quickly and speak clearly.",
    "Please schedule a meeting with the design team for tomorrow afternoon and send everyone the agenda.",
    // Numbers and out-of-lexicon words: exercises text normalization / OOV handling.
    "Version 2 shipped on March 14 with 42 fixes for Kubernetes and GitHub users.",
    "In 2026 Hinnant ran vox on Kubernetes and measured pi as 3.14.",
];

fn normalize_words(s: &str) -> Vec<String> {
    s.split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric() || *c == '\'')
                .collect::<String>()
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// Word error rate: word-level Levenshtein distance / reference length.
fn wer(reference: &str, hypothesis: &str) -> (f64, usize, usize) {
    let r = normalize_words(reference);
    let h = normalize_words(hypothesis);
    let mut prev: Vec<usize> = (0..=h.len()).collect();
    for i in 1..=r.len() {
        let mut cur = vec![i; h.len() + 1];
        for j in 1..=h.len() {
            let sub = prev[j - 1] + usize::from(r[i - 1] != h[j - 1]);
            cur[j] = sub.min(prev[j] + 1).min(cur[j - 1] + 1);
        }
        prev = cur;
    }
    let edits = prev[h.len()];
    (edits as f64 / r.len().max(1) as f64, edits, r.len())
}

fn secs(samples: usize, rate: u32) -> f64 {
    samples as f64 / rate as f64
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let sentences: Vec<String> = if args.is_empty() {
        DEFAULT_SENTENCES.iter().map(|s| s.to_string()).collect()
    } else {
        args
    };

    let config = Config::load();
    println!("model dir: {}", config.model_dir.display());

    let t = Instant::now();
    let mut tts = TtsEngine::new(&config)?;
    let tts_init = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let mut stt = SttEngine::new(&config)?;
    let stt_init = t.elapsed().as_secs_f64();
    println!(
        "init: tts {tts_init:.2}s (incl. warmup), stt {stt_init:.2}s, tts rate {} Hz",
        tts.sample_rate()
    );

    let sid = resolve_voice_id(&config.voice);
    let speed = config.speed;

    let mut total_edits = 0usize;
    let mut total_words = 0usize;
    let (mut synth_time, mut synth_audio) = (0.0f64, 0.0f64);
    let (mut stt_time, mut stt_audio) = (0.0f64, 0.0f64);

    for (idx, text) in sentences.iter().enumerate() {
        println!("\n=== [{idx}] {text}");

        // --- Batch synthesis ---
        let t = Instant::now();
        let audio_out = tts.synthesize(text, None, None)?;
        let batch_secs = t.elapsed().as_secs_f64();
        let batch_dur = secs(audio_out.samples.len(), audio_out.sample_rate);
        println!(
            "tts batch:     {batch_secs:.3}s for {batch_dur:.2}s audio  RTF {:.3}",
            batch_secs / batch_dur
        );
        synth_time += batch_secs;
        synth_audio += batch_dur;

        // --- Streaming synthesis ---
        let (tx, rx) = mpsc::channel::<Vec<f32>>();
        let t = Instant::now();
        let rate = std::thread::scope(|s| {
            let h = s.spawn(|| tts.synthesize_sentences_streaming(text, sid, speed, tx));
            let mut first: Option<f64> = None;
            let mut chunks = 0usize;
            let mut streamed: Vec<f32> = Vec::new();
            for chunk in rx {
                first.get_or_insert_with(|| t.elapsed().as_secs_f64());
                chunks += 1;
                streamed.extend_from_slice(&chunk);
            }
            let rate = h.join().expect("tts thread panicked");
            (rate, first, chunks, streamed)
        });
        let (rate, first, chunks, streamed) = rate;
        let rate = rate?;
        let stream_secs = t.elapsed().as_secs_f64();
        let stream_dur = secs(streamed.len(), rate);
        println!(
            "tts streaming: {stream_secs:.3}s for {stream_dur:.2}s audio  RTF {:.3}  first-chunk {:.3}s  chunks {chunks}",
            stream_secs / stream_dur.max(1e-9),
            first.unwrap_or(f64::NAN)
        );

        // --- Resample + pad with silence so the VAD sees onset/offset ---
        let mut pcm = audio::resample(&audio_out.samples, audio_out.sample_rate, STT_RATE);
        let pad = vec![0.0f32; STT_RATE as usize / 2];
        pcm.splice(0..0, pad.iter().copied());
        pcm.extend_from_slice(&pad);

        // --- VAD ---
        let t = Instant::now();
        let mut vad = VadSession::new(&config, VadOptions::default())?;
        let mut segments = Vec::new();
        for chunk in pcm.chunks(VAD_WINDOW) {
            vad.accept_waveform(chunk);
            segments.extend(vad.collect_segments());
        }
        vad.flush();
        segments.extend(vad.collect_segments());
        let vad_secs = t.elapsed().as_secs_f64();
        let speech: Vec<f32> = segments
            .iter()
            .flat_map(|s| s.samples.iter().copied())
            .collect();
        println!(
            "vad: {} segment(s), {:.2}s speech of {:.2}s input in {vad_secs:.3}s; starts {:?}",
            segments.len(),
            secs(speech.len(), STT_RATE),
            secs(pcm.len(), STT_RATE),
            segments
                .iter()
                .map(|s| format!("{:.2}s", s.start_sample as f64 / STT_RATE as f64))
                .collect::<Vec<_>>()
        );

        // --- STT ---
        let input = if speech.is_empty() { &pcm } else { &speech };
        let t = Instant::now();
        let transcript = stt.transcribe(STT_RATE, input)?;
        let stt_secs = t.elapsed().as_secs_f64();
        let stt_dur = secs(input.len(), STT_RATE);
        stt_time += stt_secs;
        stt_audio += stt_dur;

        let (w, edits, n) = wer(text, &transcript);
        total_edits += edits;
        total_words += n;
        println!(
            "stt:           {stt_secs:.3}s for {stt_dur:.2}s audio  RTF {:.3}",
            stt_secs / stt_dur
        );
        println!("transcript: {transcript}");
        println!("WER: {:.1}% ({edits}/{n})", w * 100.0);
    }

    println!("\n=== summary");
    println!(
        "synthesis RTF {:.3} ({synth_time:.2}s / {synth_audio:.2}s audio)",
        synth_time / synth_audio
    );
    println!(
        "stt RTF       {:.3} ({stt_time:.2}s / {stt_audio:.2}s audio)",
        stt_time / stt_audio
    );
    println!(
        "overall WER   {:.1}% ({total_edits}/{total_words})",
        total_edits as f64 / total_words.max(1) as f64 * 100.0
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::wer;

    #[test]
    fn wer_basics() {
        assert_eq!(wer("Hello, world!", "hello world").1, 0);
        assert_eq!(wer("a b c", "a x c").1, 1);
        assert_eq!(wer("a b c", "a c").1, 1);
        assert_eq!(wer("a b", "a b c").1, 1);
    }
}
