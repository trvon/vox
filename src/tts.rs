use crate::config::{Config, KokoroModel};
use crate::error::{Result, VoiceError};
use sherpa_onnx::{
    GenerationConfig, OfflineTts, OfflineTtsConfig, OfflineTtsKokoroModelConfig,
    OfflineTtsModelConfig,
};
use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;

/// Voice name to speaker ID mapping for Kokoro
/// Full list at: https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/kokoro/voices.txt
const VOICE_MAP: &[(&str, i32)] = &[
    ("af_heart", 0),
    ("af_alloy", 1),
    ("af_aoede", 2),
    ("af_bella", 3),
    ("af_jessica", 4),
    ("af_kore", 5),
    ("af_nicole", 6),
    ("af_nova", 7),
    ("af_river", 8),
    ("af_sarah", 9),
    ("af_sky", 10),
    ("am_adam", 11),
    ("am_echo", 12),
    ("am_eric", 13),
    ("am_liam", 14),
    ("am_michael", 15),
    ("am_onyx", 16),
    ("am_puck", 17),
    ("am_santa", 18),
    ("bf_alice", 19),
    ("bf_emma", 20),
    ("bf_lily", 21),
    ("bm_daniel", 22),
    ("bm_fable", 23),
    ("bm_george", 24),
    ("bm_lewis", 25),
];

/// TTS audio output
#[derive(Debug)]
pub struct TtsAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

pub struct TtsEngine {
    tts: OfflineTts,
    sample_rate: u32,
    kokoro_model: KokoroModel,
    default_voice: String,
    default_speed: f32,
}

/// sherpa-onnx's safe API builds a `CString` from the text and panics on an
/// interior NUL, so strip them before handing text over.
fn sanitize_text(text: &str) -> Cow<'_, str> {
    if text.contains('\0') {
        Cow::Owned(text.replace('\0', " "))
    } else {
        Cow::Borrowed(text)
    }
}

fn generation_config(sid: i32, speed: f32) -> GenerationConfig {
    GenerationConfig {
        sid,
        speed,
        // 1.0 = keep model pauses as-is (matches the pre-2.0 `silence_scale: 1.0`).
        silence_scale: 1.0,
        ..Default::default()
    }
}

/// Forward one synthesized chunk to the consumer.
/// Returns `true` to continue synthesis, `false` to stop (receiver dropped).
/// (sherpa-onnx maps this to the C convention 1 = continue, 0 = stop.)
pub(crate) fn forward_chunk(tx: &Sender<Vec<f32>>, samples: &[f32]) -> bool {
    if samples.is_empty() {
        return true; // continue, just skip empty chunk
    }
    tx.send(samples.to_vec()).is_ok()
}

type NoCallback = fn(&[f32], f32) -> bool;

impl TtsEngine {
    pub fn new(config: &Config) -> Result<Self> {
        let kokoro_dir = config.kokoro_dir();
        let path = |name: &str| kokoro_dir.join(name).to_string_lossy().into_owned();

        let lexicon_parts: Vec<String> = ["lexicon-us-en.txt", "lexicon-zh.txt"]
            .into_iter()
            .filter(|name| kokoro_dir.join(name).exists())
            .map(path)
            .collect();

        // espeak-ng-data is the out-of-lexicon G2P fallback (numbers, names, rare
        // words). sherpa-onnx 2.x removed espeak-ng and silently *skips* OOV words,
        // which is why vox pins sherpa-onnx 1.13.x — see Cargo.toml.
        let espeak_dir = kokoro_dir.join("espeak-ng-data");
        if !espeak_dir.is_dir() {
            return Err(VoiceError::Tts(format!(
                "Kokoro espeak-ng-data not found at {} (re-run `vox download-models`)",
                espeak_dir.display()
            )));
        }

        let model_path = kokoro_dir.join(config.kokoro_model.model_file_name());
        if !model_path.exists() {
            return Err(VoiceError::Tts(format!(
                "Kokoro model not found at {}",
                model_path.display()
            )));
        }

        let dict_dir = kokoro_dir.join("dict");

        // Determine thread count: config > env > default
        let num_threads = config
            .tts_num_threads
            .or_else(|| {
                std::env::var("VOX_TTS_THREADS")
                    .ok()
                    .and_then(|s| s.parse().ok())
            })
            .unwrap_or(2); // Default 2 for better latency on most modern CPUs

        let tts_config = OfflineTtsConfig {
            model: OfflineTtsModelConfig {
                kokoro: OfflineTtsKokoroModelConfig {
                    model: Some(model_path.to_string_lossy().into_owned()),
                    voices: Some(path("voices.bin")),
                    tokens: Some(path("tokens.txt")),
                    data_dir: Some(espeak_dir.to_string_lossy().into_owned()),
                    length_scale: 1.0,
                    dict_dir: dict_dir
                        .exists()
                        .then(|| dict_dir.to_string_lossy().into_owned()),
                    lexicon: (!lexicon_parts.is_empty()).then(|| lexicon_parts.join(",")),
                    lang: None,
                },
                num_threads,
                debug: false,
                provider: Some("cpu".to_string()),
                ..Default::default()
            },
            rule_fsts: None,
            max_num_sentences: 1,
            rule_fars: None,
            silence_scale: 1.0,
        };

        let tts = OfflineTts::create(&tts_config)
            .ok_or_else(|| VoiceError::Tts("Failed to create TTS engine".to_string()))?;

        let sample_rate = tts.sample_rate() as u32;

        // Warm up with the actual voice/speed we'll use (batch + streaming path)
        // so the ONNX Runtime session is hot for the first real request.
        let gen_config = generation_config(resolve_voice_id(&config.voice), config.speed);
        let _ = tts.generate_with_config::<NoCallback>("hi", &gen_config, None);
        let (tx, _rx) = std::sync::mpsc::channel::<Vec<f32>>();
        let _ = tts.generate_with_config(
            "hi",
            &gen_config,
            Some(move |samples: &[f32], _progress: f32| forward_chunk(&tx, samples)),
        );

        tracing::debug!(
            sample_rate,
            num_threads,
            default_voice = %config.voice,
            default_speed = config.speed,
            "TTS engine initialized"
        );

        Ok(Self {
            tts,
            sample_rate,
            kokoro_model: config.kokoro_model,
            default_voice: config.voice.clone(),
            default_speed: config.speed,
        })
    }

    /// Get the native sample rate of the TTS engine.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn kokoro_model(&self) -> KokoroModel {
        self.kokoro_model
    }

    /// Synthesize text to audio samples (batch — waits for full synthesis).
    pub fn synthesize(
        &mut self,
        text: &str,
        voice: Option<&str>,
        speed: Option<f32>,
    ) -> Result<TtsAudio> {
        let voice_name = voice.unwrap_or(&self.default_voice);
        let speed = speed.unwrap_or(self.default_speed);
        let sid = resolve_voice_id(voice_name);

        tracing::debug!(
            voice = voice_name,
            sid,
            speed,
            text_len = text.len(),
            "Synthesizing speech"
        );

        let text = sanitize_text(text);
        let audio = self
            .tts
            .generate_with_config::<NoCallback>(&text, &generation_config(sid, speed), None)
            .ok_or_else(|| VoiceError::Tts("Synthesis returned null".to_string()))?;

        let samples = audio.samples();
        if samples.is_empty() {
            return Err(VoiceError::Tts("No audio samples generated".to_string()));
        }

        Ok(TtsAudio {
            samples: samples.to_vec(),
            sample_rate: audio.sample_rate() as u32,
        })
    }

    /// Synthesize text with streaming callback — audio chunks are sent through `tx`
    /// as they are generated by the C library. Returns the sample rate.
    ///
    /// sherpa-onnx splits the text into sentence-sized chunks internally and fires
    /// the callback once per chunk; synthesis stops early if the receiver is dropped.
    ///
    /// The callback runs on the same thread as the C synthesis call, so this method
    /// must be called from a blocking context (e.g. `spawn_blocking`).
    pub fn synthesize_streaming(
        &mut self,
        text: &str,
        sid: i32,
        speed: f32,
        tx: Sender<Vec<f32>>,
    ) -> Result<u32> {
        tracing::debug!(sid, speed, text_len = text.len(), "Streaming synthesis");
        self.stream_one(text, sid, speed, tx);
        Ok(self.sample_rate)
    }

    /// Run one streaming generation. Returns `false` if the receiver hung up
    /// (synthesis was stopped early).
    fn stream_one(&self, text: &str, sid: i32, speed: f32, tx: Sender<Vec<f32>>) -> bool {
        let text = sanitize_text(text);
        let alive = Arc::new(AtomicBool::new(true));
        let alive_cb = Arc::clone(&alive);
        // The returned full-utterance buffer is dropped immediately; chunks were
        // already delivered through `tx`. The closure (and `tx`) is dropped when
        // generation returns, which signals end-of-stream to the receiver.
        let _ = self.tts.generate_with_config(
            &text,
            &generation_config(sid, speed),
            Some(move |samples: &[f32], _progress: f32| {
                let ok = forward_chunk(&tx, samples);
                if !ok {
                    alive_cb.store(false, Ordering::Relaxed);
                }
                ok
            }),
        );
        alive.load(Ordering::Relaxed)
    }

    /// Synthesize sentences sequentially with streaming.
    /// Streams first sentence immediately, then subsequent sentences.
    /// Reduces time-to-first-audio for long prompts.
    pub fn synthesize_sentences_streaming(
        &mut self,
        text: &str,
        sid: i32,
        speed: f32,
        tx: Sender<Vec<f32>>,
    ) -> Result<u32> {
        let sentences = split_sentences(text);

        if sentences.len() <= 1 {
            // Fast path: single sentence or empty
            return self.synthesize_streaming(text, sid, speed, tx);
        }

        tracing::debug!(
            sentence_count = sentences.len(),
            total_chars = text.len(),
            "Multi-sentence streaming synthesis"
        );

        // 150 ms of silence between sentences (natural prosody)
        let silence_samples = (self.sample_rate as f32 * 0.15) as usize;

        for (i, sentence) in sentences.iter().enumerate() {
            if i == 0 {
                tracing::debug!("Streaming first sentence immediately");
            }

            if !self.stream_one(sentence, sid, speed, tx.clone()) {
                break; // receiver dropped — skip remaining sentences
            }

            if i < sentences.len() - 1 && tx.send(vec![0.0f32; silence_samples]).is_err() {
                break;
            }
        }

        drop(tx);
        Ok(self.sample_rate)
    }
}

pub fn resolve_voice_id(name: &str) -> i32 {
    VOICE_MAP
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, id)| *id)
        .unwrap_or_else(|| {
            // Try parsing as numeric ID
            name.parse::<i32>().unwrap_or(0)
        })
}

/// Split text into sentences on sentence-ending punctuation followed by whitespace.
/// Returns the original text as a single-element slice if no split points are found.
pub fn split_sentences(text: &str) -> Vec<&str> {
    let mut sentences = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();

    for (i, &b) in bytes.iter().enumerate() {
        if (b == b'.' || b == b'!' || b == b'?')
            && i + 1 < bytes.len()
            && bytes[i + 1].is_ascii_whitespace()
        {
            let sentence = text[start..=i].trim();
            if !sentence.is_empty() {
                sentences.push(sentence);
            }
            start = i + 1;
        }
    }

    // Remaining tail
    let tail = text[start..].trim();
    if !tail.is_empty() {
        sentences.push(tail);
    }

    if sentences.is_empty() {
        sentences.push(text);
    }

    sentences
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn all_named_voices_resolve_correctly() {
        for &(name, expected_id) in VOICE_MAP {
            assert_eq!(
                resolve_voice_id(name),
                expected_id,
                "Voice '{name}' should resolve to {expected_id}"
            );
        }
    }

    #[test]
    fn voice_map_has_26_entries() {
        assert_eq!(VOICE_MAP.len(), 26);
    }

    #[test]
    fn unknown_voice_falls_back_to_numeric() {
        assert_eq!(resolve_voice_id("5"), 5);
        assert_eq!(resolve_voice_id("25"), 25);
        assert_eq!(resolve_voice_id("0"), 0);
    }

    #[test]
    fn completely_unknown_voice_falls_back_to_zero() {
        assert_eq!(resolve_voice_id("nonexistent_voice"), 0);
        assert_eq!(resolve_voice_id(""), 0);
    }

    #[test]
    fn voice_map_has_no_duplicate_ids() {
        let ids: Vec<i32> = VOICE_MAP.iter().map(|(_, id)| *id).collect();
        let unique: HashSet<i32> = ids.iter().copied().collect();
        assert_eq!(ids.len(), unique.len(), "VOICE_MAP contains duplicate IDs");
    }

    #[test]
    fn voice_map_has_no_duplicate_names() {
        let names: Vec<&str> = VOICE_MAP.iter().map(|(n, _)| *n).collect();
        let unique: HashSet<&str> = names.iter().copied().collect();
        assert_eq!(
            names.len(),
            unique.len(),
            "VOICE_MAP contains duplicate names"
        );
    }

    #[test]
    fn voice_ids_are_sequential_0_to_25() {
        for (i, &(_, id)) in VOICE_MAP.iter().enumerate() {
            assert_eq!(
                id, i as i32,
                "Voice at index {i} has non-sequential ID {id}"
            );
        }
    }

    #[test]
    fn split_sentences_single() {
        let result = split_sentences("Hello world");
        assert_eq!(result, vec!["Hello world"]);
    }

    #[test]
    fn split_sentences_two() {
        let result = split_sentences("Hello world. How are you?");
        assert_eq!(result, vec!["Hello world.", "How are you?"]);
    }

    #[test]
    fn split_sentences_three() {
        let result = split_sentences("First. Second! Third? Tail");
        assert_eq!(result, vec!["First.", "Second!", "Third?", "Tail"]);
    }

    #[test]
    fn split_sentences_trailing_period_no_space() {
        let result = split_sentences("Hello world.");
        assert_eq!(result, vec!["Hello world."]);
    }

    #[test]
    fn split_sentences_empty() {
        let result = split_sentences("");
        assert_eq!(result, vec![""]);
    }

    #[test]
    fn split_sentences_abbreviation_no_split() {
        // "Dr.Smith" has no space after period, should not split
        let result = split_sentences("Dr.Smith is here");
        assert_eq!(result, vec!["Dr.Smith is here"]);
    }

    #[test]
    fn forward_chunk_sends_chunks() {
        let (tx, rx) = std::sync::mpsc::channel();
        let samples: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
        assert!(forward_chunk(&tx, &samples));
        let received: Vec<f32> = rx.recv().unwrap();
        assert_eq!(received, samples);
    }

    #[test]
    fn forward_chunk_stops_when_receiver_dropped() {
        let (tx, rx) = std::sync::mpsc::channel::<Vec<f32>>();
        drop(rx);
        assert!(!forward_chunk(&tx, &[0.1])); // should signal stop
    }

    #[test]
    fn forward_chunk_skips_empty() {
        let (tx, rx) = std::sync::mpsc::channel::<Vec<f32>>();
        assert!(forward_chunk(&tx, &[])); // continue, just skip
        drop(tx);
        assert!(rx.recv().is_err(), "empty chunk must not be forwarded");
    }

    #[test]
    fn sanitize_text_strips_nul() {
        assert_eq!(sanitize_text("a\0b"), "a b");
        assert!(matches!(sanitize_text("plain"), Cow::Borrowed("plain")));
    }

    #[test]
    fn generation_config_keeps_silence() {
        let g = generation_config(3, 1.25);
        assert_eq!(g.sid, 3);
        assert!((g.speed - 1.25).abs() < f32::EPSILON);
        assert!((g.silence_scale - 1.0).abs() < f32::EPSILON);
    }
}
