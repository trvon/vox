use crate::config::Config;
use crate::error::{Result, VoiceError};
use std::ffi::CString;
use std::ptr::null;

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

/// TTS audio output (matches sherpa-rs TtsAudio interface)
#[derive(Debug)]
pub struct TtsAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

pub struct TtsEngine {
    tts: *const sherpa_rs_sys::SherpaOnnxOfflineTts,
    sample_rate: u32,
    default_voice: String,
    default_speed: f32,
}

fn cstring(s: &str) -> CString {
    CString::new(s).unwrap_or_else(|_| CString::new("").unwrap())
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

        let model = cstring(&kokoro_dir.join("model.onnx").to_string_lossy());
        let voices = cstring(&kokoro_dir.join("voices.bin").to_string_lossy());
        let tokens = cstring(&kokoro_dir.join("tokens.txt").to_string_lossy());
        let data_dir = cstring(&kokoro_dir.join("espeak-ng-data").to_string_lossy());
        let dict_dir = cstring(&kokoro_dir.join("dict").to_string_lossy());
        let lexicon = cstring(&lexicon_parts.join(","));
        let provider = cstring("cpu");
        let lang = cstring("");

        let tts = unsafe {
            let config = sherpa_rs_sys::SherpaOnnxOfflineTtsConfig {
                model: sherpa_rs_sys::SherpaOnnxOfflineTtsModelConfig {
                    vits: std::mem::zeroed(),
                    num_threads: 1,
                    debug: 0,
                    provider: provider.as_ptr(),
                    matcha: std::mem::zeroed(),
                    kokoro: sherpa_rs_sys::SherpaOnnxOfflineTtsKokoroModelConfig {
                        model: model.as_ptr(),
                        voices: voices.as_ptr(),
                        tokens: tokens.as_ptr(),
                        data_dir: data_dir.as_ptr(),
                        length_scale: 1.0,
                        dict_dir: dict_dir.as_ptr(),
                        lexicon: lexicon.as_ptr(),
                        lang: lang.as_ptr(),
                    },
                    kitten: std::mem::zeroed(),
                },
                rule_fsts: null(),
                max_num_sentences: 1,
                rule_fars: null(),
                silence_scale: 1.0,
            };

            sherpa_rs_sys::SherpaOnnxCreateOfflineTts(&config)
        };

        if tts.is_null() {
            return Err(VoiceError::Tts("Failed to create TTS engine".to_string()));
        }

        let sample_rate = unsafe { sherpa_rs_sys::SherpaOnnxOfflineTtsSampleRate(tts) } as u32;

        tracing::info!(sample_rate, "TTS engine initialized");

        Ok(Self {
            tts,
            sample_rate,
            default_voice: config.voice.clone(),
            default_speed: config.speed,
        })
    }

    /// Get the native sample rate of the TTS engine.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
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

        let c_text = cstring(text);

        unsafe {
            let audio_ptr =
                sherpa_rs_sys::SherpaOnnxOfflineTtsGenerate(self.tts, c_text.as_ptr(), sid, speed);

            if audio_ptr.is_null() {
                return Err(VoiceError::Tts("Synthesis returned null".to_string()));
            }

            let audio = audio_ptr.read();

            if audio.n <= 0 || audio.samples.is_null() {
                sherpa_rs_sys::SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio_ptr);
                return Err(VoiceError::Tts("No audio samples generated".to_string()));
            }

            let samples = std::slice::from_raw_parts(audio.samples, audio.n as usize).to_vec();
            let sample_rate = audio.sample_rate as u32;

            sherpa_rs_sys::SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio_ptr);

            Ok(TtsAudio {
                samples,
                sample_rate,
            })
        }
    }

    /// Synthesize text with streaming callback — audio chunks are sent through `tx`
    /// as they are generated by the C library. Returns the sample rate.
    ///
    /// The callback runs on the same thread as the C synthesis call, so this method
    /// must be called from a blocking context (e.g. `spawn_blocking`).
    pub fn synthesize_streaming(
        &mut self,
        text: &str,
        sid: i32,
        speed: f32,
        tx: std::sync::mpsc::Sender<Vec<f32>>,
    ) -> Result<u32> {
        tracing::debug!(sid, speed, text_len = text.len(), "Streaming synthesis");

        let c_text = cstring(text);

        unsafe {
            let arg = &tx as *const std::sync::mpsc::Sender<Vec<f32>> as *mut std::ffi::c_void;

            let audio_ptr = sherpa_rs_sys::SherpaOnnxOfflineTtsGenerateWithCallbackWithArg(
                self.tts,
                c_text.as_ptr(),
                sid,
                speed,
                Some(streaming_callback),
                arg,
            );

            // The callback has already streamed chunks through tx.
            // We still get the full audio back — just clean it up.
            if !audio_ptr.is_null() {
                sherpa_rs_sys::SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio_ptr);
            }
        }

        // Drop the sender so the receiver knows we're done
        drop(tx);

        Ok(self.sample_rate)
    }
}

/// C callback for streaming TTS. Copies samples and sends them through the channel.
/// sherpa-onnx convention: return 1 to continue synthesis, 0 to stop.
unsafe extern "C" fn streaming_callback(
    samples: *const f32,
    n: i32,
    arg: *mut std::ffi::c_void,
) -> i32 {
    if samples.is_null() || n <= 0 {
        return 1; // continue, just skip empty chunk
    }
    unsafe {
        let tx = &*(arg as *const std::sync::mpsc::Sender<Vec<f32>>);
        let chunk = std::slice::from_raw_parts(samples, n as usize).to_vec();
        if tx.send(chunk).is_ok() {
            1 // continue synthesis
        } else {
            0 // receiver dropped, stop synthesis
        }
    }
}

impl Drop for TtsEngine {
    fn drop(&mut self) {
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOfflineTts(self.tts);
        }
    }
}

// TtsEngine wraps a raw pointer but the C library is thread-safe for single-writer access
unsafe impl Send for TtsEngine {}

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
    fn streaming_callback_sends_chunks() {
        let (tx, rx) = std::sync::mpsc::channel::<Vec<f32>>();
        let samples = [1.0f32, 2.0, 3.0, 4.0];
        let arg = &tx as *const std::sync::mpsc::Sender<Vec<f32>> as *mut std::ffi::c_void;

        let ret = unsafe { streaming_callback(samples.as_ptr(), 4, arg) };
        assert_eq!(ret, 1); // 1 = continue synthesis

        drop(tx);
        let chunk = rx.recv().unwrap();
        assert_eq!(chunk, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn streaming_callback_stops_when_receiver_dropped() {
        let (tx, _rx) = std::sync::mpsc::channel::<Vec<f32>>();
        let samples = [1.0f32];
        let arg = &tx as *const std::sync::mpsc::Sender<Vec<f32>> as *mut std::ffi::c_void;

        // Drop receiver first
        drop(_rx);

        let ret = unsafe { streaming_callback(samples.as_ptr(), 1, arg) };
        assert_eq!(ret, 0); // 0 = stop synthesis
    }

    #[test]
    fn streaming_callback_handles_null_samples() {
        let (tx, _rx) = std::sync::mpsc::channel::<Vec<f32>>();
        let arg = &tx as *const std::sync::mpsc::Sender<Vec<f32>> as *mut std::ffi::c_void;

        let ret = unsafe { streaming_callback(std::ptr::null(), 0, arg) };
        assert_eq!(ret, 1); // 1 = continue (skip empty)
    }

    #[test]
    fn streaming_callback_handles_zero_length() {
        let (tx, _rx) = std::sync::mpsc::channel::<Vec<f32>>();
        let samples = [1.0f32];
        let arg = &tx as *const std::sync::mpsc::Sender<Vec<f32>> as *mut std::ffi::c_void;

        let ret = unsafe { streaming_callback(samples.as_ptr(), 0, arg) };
        assert_eq!(ret, 1); // 1 = continue (skip empty)
    }
}
