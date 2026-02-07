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

fn resolve_voice_id(name: &str) -> i32 {
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
