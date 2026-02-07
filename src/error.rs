use thiserror::Error;

#[derive(Error, Debug)]
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
