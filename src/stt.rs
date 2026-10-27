use crate::config::{Config, SttModel};
use crate::error::{Result, VoiceError};
use sherpa_onnx::{
    OfflineMoonshineModelConfig, OfflineRecognizer, OfflineRecognizerConfig,
    OfflineTransducerModelConfig,
};

/// Default ONNX Runtime intra-op threads for Moonshine. Measured with
/// `scripts/profile.sh perf`: with more threads, ~80% of CPU goes to ORT's
/// spin-waiting thread pool (`SpinPause`). On a 76 s lecture, 4 threads used
/// 10-15x the CPU time of 1 thread for no reliable speedup, because VAD
/// segments are short. One thread suits always-on recording; raise it with
/// `VOX_STT_THREADS` for batch transcription of long files on an idle machine.
const DEFAULT_STT_THREADS: i32 = 1;

pub struct SttEngine {
    recognizer: OfflineRecognizer,
    model: SttModel,
    model_dir: std::path::PathBuf,
}

/// A recognized token and its start time relative to the start of the input.
#[derive(Debug, Clone, PartialEq)]
pub struct TimedToken {
    /// Token text as sherpa returns it; word-initial tokens carry a leading space.
    pub text: String,
    pub start_ms: u64,
}

/// Recognition result. `tokens` is empty when the model has no timestamps.
#[derive(Debug, Clone, Default)]
pub struct SttOutput {
    pub text: String,
    pub tokens: Vec<TimedToken>,
}

/// Pair sherpa tokens with timestamps (seconds). Mismatched lengths, as with
/// models that don't emit timestamps, yield no timed tokens.
pub(crate) fn zip_timestamps(tokens: &[String], timestamps: Option<&[f32]>) -> Vec<TimedToken> {
    match timestamps {
        Some(ts) if ts.len() == tokens.len() && !ts.is_empty() => tokens
            .iter()
            .zip(ts)
            .map(|(t, &s)| TimedToken {
                text: t.clone(),
                start_ms: (s.max(0.0) * 1000.0).round() as u64,
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// Thread count for STT: `VOX_STT_THREADS` env override, else
/// [`DEFAULT_STT_THREADS`].
pub(crate) fn stt_num_threads() -> i32 {
    if let Ok(s) = std::env::var("VOX_STT_THREADS")
        && let Ok(n) = s.trim().parse::<i32>()
        && n > 0
    {
        return n;
    }
    DEFAULT_STT_THREADS
}

impl SttEngine {
    pub fn new(config: &Config) -> Result<Self> {
        let model = config.stt_model;
        let dir = config.stt_dir();
        for name in model.required_files() {
            if !dir.join(name).exists() {
                return Err(VoiceError::Stt(format!(
                    "STT model file missing: {} (run `vox download-models`)",
                    dir.join(name).display()
                )));
            }
        }
        let path = |name: &str| Some(dir.join(name).to_string_lossy().into_owned());

        let num_threads = stt_num_threads();
        let mut rc = OfflineRecognizerConfig::default();
        match model {
            SttModel::MoonshineBase => {
                rc.model_config.moonshine = OfflineMoonshineModelConfig {
                    preprocessor: path("preprocess.onnx"),
                    encoder: path("encode.int8.onnx"),
                    uncached_decoder: path("uncached_decode.int8.onnx"),
                    cached_decoder: path("cached_decode.int8.onnx"),
                    merged_decoder: None,
                };
            }
            SttModel::ParakeetTdtV2 | SttModel::ParakeetTdtV3 => {
                rc.model_config.transducer = OfflineTransducerModelConfig {
                    encoder: path("encoder.int8.onnx"),
                    decoder: path("decoder.int8.onnx"),
                    joiner: path("joiner.int8.onnx"),
                };
                rc.model_config.model_type = Some("nemo_transducer".to_string());
            }
        }
        rc.model_config.tokens = path("tokens.txt");
        rc.model_config.num_threads = num_threads;
        rc.model_config.provider = Some("cpu".to_string());
        rc.decoding_method = Some("greedy_search".to_string());

        let recognizer = OfflineRecognizer::create(&rc)
            .ok_or_else(|| VoiceError::Stt(format!("Failed to initialize STT model {model}")))?;

        tracing::info!(num_threads, %model, "STT engine initialized");

        Ok(Self {
            recognizer,
            model,
            model_dir: dir,
        })
    }

    pub fn model(&self) -> SttModel {
        self.model
    }

    pub fn matches_config(&self, config: &Config) -> bool {
        self.model == config.stt_model && self.model_dir == config.stt_dir()
    }

    /// Transcribe audio samples to text
    pub fn transcribe(&mut self, sample_rate: u32, samples: &[f32]) -> Result<String> {
        Ok(self.transcribe_timed(sample_rate, samples)?.text)
    }

    /// Transcribe, keeping per-token timestamps when the model provides them.
    pub fn transcribe_timed(&mut self, sample_rate: u32, samples: &[f32]) -> Result<SttOutput> {
        tracing::debug!(
            sample_rate,
            num_samples = samples.len(),
            duration_secs = samples.len() as f32 / sample_rate as f32,
            "Transcribing audio"
        );

        if samples.is_empty() {
            return Ok(SttOutput::default());
        }

        let stream = self.recognizer.create_stream();
        stream.accept_waveform(sample_rate as i32, samples);
        self.recognizer.decode(&stream);
        let result = stream
            .get_result()
            .ok_or_else(|| VoiceError::Stt("Recognizer returned no result".to_string()))?;

        let out = SttOutput {
            text: result.text.trim().to_string(),
            tokens: zip_timestamps(&result.tokens, result.timestamps.as_deref()),
        };
        tracing::debug!(text = %out.text, tokens = out.tokens.len(), "Transcription complete");
        Ok(out)
    }
}

// OfflineRecognizer is Send in sherpa-onnx; kept explicit for the public contract.
unsafe impl Send for SttEngine {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn zip_timestamps_pairs_or_drops() {
        let toks = vec![" hello".to_string(), " wor".to_string(), "ld".to_string()];
        let timed = zip_timestamps(&toks, Some(&[0.0, 0.48, 0.64]));
        assert_eq!(timed.len(), 3);
        assert_eq!(timed[1].start_ms, 480);
        assert!(zip_timestamps(&toks, None).is_empty());
        assert!(zip_timestamps(&toks, Some(&[0.1])).is_empty());
    }

    #[test]
    fn stt_threads_bounded() {
        let n = stt_num_threads();
        assert!(n >= 1);
        if std::env::var("VOX_STT_THREADS").is_err() {
            assert_eq!(n, DEFAULT_STT_THREADS);
        }
    }

    #[test]
    fn stt_config_paths_correct() {
        let config = Config::default();
        let moonshine_dir = config.moonshine_dir();

        let preprocess = moonshine_dir.join("preprocess.onnx");
        let encoder = moonshine_dir.join("encode.int8.onnx");
        let uncached_decoder = moonshine_dir.join("uncached_decode.int8.onnx");
        let cached_decoder = moonshine_dir.join("cached_decode.int8.onnx");
        let tokens = moonshine_dir.join("tokens.txt");

        assert!(preprocess.to_string_lossy().ends_with("preprocess.onnx"));
        assert!(encoder.to_string_lossy().ends_with("encode.int8.onnx"));
        assert!(
            uncached_decoder
                .to_string_lossy()
                .ends_with("uncached_decode.int8.onnx")
        );
        assert!(
            cached_decoder
                .to_string_lossy()
                .ends_with("cached_decode.int8.onnx")
        );
        assert!(tokens.to_string_lossy().ends_with("tokens.txt"));
    }
}
