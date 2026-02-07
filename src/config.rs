use serde::Deserialize;
use std::path::PathBuf;

const APP_NAME: &str = "vox";

#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
pub struct Config {
    pub model_dir: PathBuf,
    pub voice: String,
    pub speed: f32,
    pub whisper_model: WhisperModel,
    pub log_level: String,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum WhisperModel {
    Tiny,
    Base,
    Small,
}

impl std::fmt::Display for WhisperModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WhisperModel::Tiny => write!(f, "tiny"),
            WhisperModel::Base => write!(f, "base"),
            WhisperModel::Small => write!(f, "small"),
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model_dir: default_model_dir(),
            voice: "af_heart".to_string(),
            speed: 1.0,
            whisper_model: WhisperModel::Tiny,
            log_level: "info".to_string(),
        }
    }
}

/// Resolve XDG_DATA_HOME or platform default for model storage
fn default_model_dir() -> PathBuf {
    if let Ok(xdg) = std::env::var("XDG_DATA_HOME") {
        return PathBuf::from(xdg).join(APP_NAME).join("models");
    }
    dirs::data_local_dir()
        .unwrap_or_else(|| {
            dirs::home_dir()
                .map(|h| h.join(".local").join("share"))
                .unwrap_or_else(|| PathBuf::from("."))
        })
        .join(APP_NAME)
        .join("models")
}

/// Resolve XDG_CONFIG_HOME or platform default for config file
fn config_path() -> PathBuf {
    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
        return PathBuf::from(xdg).join(APP_NAME).join("config.toml");
    }
    dirs::config_dir()
        .unwrap_or_else(|| {
            dirs::home_dir()
                .map(|h| h.join(".config"))
                .unwrap_or_else(|| PathBuf::from("."))
        })
        .join(APP_NAME)
        .join("config.toml")
}

impl Config {
    pub fn load() -> Self {
        let mut config = Self::default();

        // Load from TOML file if it exists
        let path = config_path();
        if path.exists() {
            if let Ok(contents) = std::fs::read_to_string(&path) {
                if let Ok(file_config) = toml::from_str::<Config>(&contents) {
                    config = file_config;
                } else {
                    tracing::warn!("Failed to parse config file at {}", path.display());
                }
            }
        }

        // Override with env vars (VOX_ prefix)
        if let Ok(val) = std::env::var("VOX_MODEL_DIR") {
            config.model_dir = PathBuf::from(val);
        }
        if let Ok(val) = std::env::var("VOX_VOICE") {
            config.voice = val;
        }
        if let Ok(val) = std::env::var("VOX_SPEED") {
            if let Ok(speed) = val.parse() {
                config.speed = speed;
            }
        }
        if let Ok(val) = std::env::var("VOX_WHISPER_MODEL") {
            match val.to_lowercase().as_str() {
                "tiny" => config.whisper_model = WhisperModel::Tiny,
                "base" => config.whisper_model = WhisperModel::Base,
                "small" => config.whisper_model = WhisperModel::Small,
                _ => tracing::warn!("Unknown whisper model: {val}"),
            }
        }
        if let Ok(val) = std::env::var("VOX_LOG_LEVEL") {
            config.log_level = val;
        }

        config
    }

    /// Resolve paths for the whisper model files
    pub fn whisper_dir(&self) -> PathBuf {
        self.model_dir
            .join(format!("sherpa-onnx-whisper-{}", self.whisper_model))
    }

    /// Resolve path for the kokoro model directory
    pub fn kokoro_dir(&self) -> PathBuf {
        self.model_dir.join("kokoro-multi-lang-v1_0")
    }

    /// Resolve path for the VAD model
    pub fn vad_model_path(&self) -> PathBuf {
        self.model_dir.join("silero_vad.onnx")
    }
}
