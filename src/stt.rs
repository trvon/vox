use crate::config::Config;
use crate::error::{Result, VoiceError};
use sherpa_rs::moonshine::{MoonshineConfig, MoonshineRecognizer};

pub struct SttEngine {
    recognizer: MoonshineRecognizer,
}

impl SttEngine {
    pub fn new(config: &Config) -> Result<Self> {
        let moonshine_dir = config.moonshine_dir();

        let moonshine_config = MoonshineConfig {
            preprocessor: moonshine_dir
                .join("preprocess.onnx")
                .to_string_lossy()
                .to_string(),
            encoder: moonshine_dir
                .join("encode.int8.onnx")
                .to_string_lossy()
                .to_string(),
            uncached_decoder: moonshine_dir
                .join("uncached_decode.int8.onnx")
                .to_string_lossy()
                .to_string(),
            cached_decoder: moonshine_dir
                .join("cached_decode.int8.onnx")
                .to_string_lossy()
                .to_string(),
            tokens: moonshine_dir
                .join("tokens.txt")
                .to_string_lossy()
                .to_string(),
            ..Default::default()
        };

        let recognizer = MoonshineRecognizer::new(moonshine_config)
            .map_err(|e| VoiceError::Stt(format!("Failed to initialize Moonshine: {e}")))?;

        tracing::info!("STT engine initialized (Moonshine Base)");

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

// MoonshineRecognizer wraps a raw pointer but sherpa-rs declares Send+Sync
unsafe impl Send for SttEngine {}

#[cfg(test)]
mod tests {
    use crate::config::Config;

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
