use thiserror::Error;

#[derive(Error, Debug)]
#[allow(dead_code)]
pub enum VoiceError {
    #[error("TTS error: {0}")]
    Tts(String),

    #[error("STT error: {0}")]
    Stt(String),

    #[error("VAD error: {0}")]
    Vad(String),

    #[error("Audio error: {0}")]
    Audio(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Model download failed: {0}")]
    Download(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Timeout waiting for speech")]
    Timeout,

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Reqwest(#[from] reqwest::Error),
}

pub type Result<T> = std::result::Result<T, VoiceError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tts_error_display() {
        let err = VoiceError::Tts("synthesis failed".to_string());
        assert_eq!(err.to_string(), "TTS error: synthesis failed");
    }

    #[test]
    fn stt_error_display() {
        let err = VoiceError::Stt("decode failed".to_string());
        assert_eq!(err.to_string(), "STT error: decode failed");
    }

    #[test]
    fn vad_error_display() {
        let err = VoiceError::Vad("init failed".to_string());
        assert_eq!(err.to_string(), "VAD error: init failed");
    }

    #[test]
    fn audio_error_display() {
        let err = VoiceError::Audio("no device".to_string());
        assert_eq!(err.to_string(), "Audio error: no device");
    }

    #[test]
    fn model_not_found_display() {
        let err = VoiceError::ModelNotFound("whisper.onnx".to_string());
        assert_eq!(err.to_string(), "Model not found: whisper.onnx");
    }

    #[test]
    fn download_error_display() {
        let err = VoiceError::Download("HTTP 404".to_string());
        assert_eq!(err.to_string(), "Model download failed: HTTP 404");
    }

    #[test]
    fn config_error_display() {
        let err = VoiceError::Config("bad toml".to_string());
        assert_eq!(err.to_string(), "Configuration error: bad toml");
    }

    #[test]
    fn timeout_error_display() {
        let err = VoiceError::Timeout;
        assert_eq!(err.to_string(), "Timeout waiting for speech");
    }

    #[test]
    fn from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let voice_err: VoiceError = io_err.into();
        assert!(matches!(voice_err, VoiceError::Io(_)));
        assert!(voice_err.to_string().contains("file not found"));
    }
}
