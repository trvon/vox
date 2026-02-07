use crate::config::Config;
use crate::error::{Result, VoiceError};
use sherpa_rs::tts::{KokoroTts, KokoroTtsConfig, TtsAudio};

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

pub struct TtsEngine {
    tts: KokoroTts,
    default_voice: String,
    default_speed: f32,
}

impl TtsEngine {
    pub fn new(config: &Config) -> Result<Self> {
        let kokoro_dir = config.kokoro_dir();

        let lexicon_us = kokoro_dir.join("lexicon-us-en.txt");
        let lexicon_zh = kokoro_dir.join("lexicon-zh.txt");

        let mut lexicon_parts = Vec::new();
        if lexicon_us.exists() {
            lexicon_parts.push(lexicon_us.to_string_lossy().to_string());
        }
        if lexicon_zh.exists() {
            lexicon_parts.push(lexicon_zh.to_string_lossy().to_string());
        }

        let tts_config = KokoroTtsConfig {
            model: kokoro_dir
                .join("model.onnx")
                .to_string_lossy()
                .to_string(),
            voices: kokoro_dir
                .join("voices.bin")
                .to_string_lossy()
                .to_string(),
            tokens: kokoro_dir
                .join("tokens.txt")
                .to_string_lossy()
                .to_string(),
            data_dir: kokoro_dir
                .join("espeak-ng-data")
                .to_string_lossy()
                .to_string(),
            dict_dir: kokoro_dir.join("dict").to_string_lossy().to_string(),
            lexicon: lexicon_parts.join(","),
            length_scale: 1.0,
            ..Default::default()
        };

        let tts = KokoroTts::new(tts_config);
        tracing::info!("TTS engine initialized");

        Ok(Self {
            tts,
            default_voice: config.voice.clone(),
            default_speed: config.speed,
        })
    }

    /// Synthesize text to audio samples
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

        self.tts
            .create(text, sid, speed)
            .map_err(|e| VoiceError::Tts(format!("Synthesis failed: {e}")))
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

// TtsEngine wraps a raw pointer but sherpa-rs declares Send+Sync on it
unsafe impl Send for TtsEngine {}

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
            assert_eq!(id, i as i32, "Voice at index {i} has non-sequential ID {id}");
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
}
