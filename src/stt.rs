use crate::config::Config;
use crate::error::{Result, VoiceError};
use sherpa_rs::whisper::{WhisperConfig, WhisperRecognizer};

pub struct SttEngine {
    recognizer: WhisperRecognizer,
}

impl SttEngine {
    pub fn new(config: &Config) -> Result<Self> {
        let whisper_dir = config.whisper_dir();
        let model = &config.whisper_model;

        let whisper_config = WhisperConfig {
            encoder: whisper_dir
                .join(format!("{model}-encoder.onnx"))
                .to_string_lossy()
                .to_string(),
            decoder: whisper_dir
                .join(format!("{model}-decoder.onnx"))
                .to_string_lossy()
                .to_string(),
            tokens: whisper_dir
                .join(format!("{model}-tokens.txt"))
                .to_string_lossy()
                .to_string(),
            language: "en".to_string(),
            ..Default::default()
        };

        let recognizer = WhisperRecognizer::new(whisper_config)
            .map_err(|e| VoiceError::Stt(format!("Failed to initialize Whisper: {e}")))?;

        tracing::info!(model = %model, "STT engine initialized");

        Ok(Self { recognizer })
    }

    /// Transcribe audio samples to text
    pub fn transcribe(&mut self, sample_rate: u32, samples: &[f32]) -> Result<String> {
        tracing::debug!(
            sample_rate,
            num_samples = samples.len(),
            duration_secs = samples.len() as f32 / sample_rate as f32,
            "Transcribing audio"
        );

        let result = self.recognizer.transcribe(sample_rate, samples);
        let text = result.text.trim().to_string();

        tracing::debug!(text = %text, "Transcription complete");
        Ok(text)
    }
}

// WhisperRecognizer wraps a raw pointer but sherpa-rs declares Send+Sync
unsafe impl Send for SttEngine {}
