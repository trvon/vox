use serde::{Deserialize, Serialize};
use std::path::PathBuf;

const APP_NAME: &str = "vox";

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(default)]
pub struct Config {
    pub model_dir: PathBuf,
    pub voice: String,
    pub speed: f32,
    pub log_level: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model_dir: default_model_dir(),
            voice: "af_heart".to_string(),
            speed: 1.0,
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
        if path.exists()
            && let Ok(contents) = std::fs::read_to_string(&path)
        {
            if let Ok(file_config) = toml::from_str::<Config>(&contents) {
                config = file_config;
            } else {
                tracing::warn!("Failed to parse config file at {}", path.display());
            }
        }

        // Override with env vars (VOX_ prefix)
        if let Ok(val) = std::env::var("VOX_MODEL_DIR") {
            config.model_dir = PathBuf::from(val);
        }
        if let Ok(val) = std::env::var("VOX_VOICE") {
            config.voice = val;
        }
        if let Ok(val) = std::env::var("VOX_SPEED")
            && let Ok(speed) = val.parse()
        {
            config.speed = speed;
        }
        if let Ok(val) = std::env::var("VOX_LOG_LEVEL") {
            config.log_level = val;
        }

        config
    }

    /// Return the config file path
    pub fn config_path() -> PathBuf {
        config_path()
    }

    /// Get a config value by key name
    pub fn get_value(&self, key: &str) -> Option<String> {
        match key {
            "voice" => Some(self.voice.clone()),
            "speed" => Some(self.speed.to_string()),
            "model_dir" => Some(self.model_dir.to_string_lossy().to_string()),
            "log_level" => Some(self.log_level.clone()),
            _ => None,
        }
    }

    /// Set a config value by key and persist to disk
    pub fn set_value(key: &str, value: &str) -> std::result::Result<(), String> {
        let path = config_path();

        // Read existing TOML table or start fresh
        let mut table = if path.exists() {
            let contents = std::fs::read_to_string(&path)
                .map_err(|e| format!("Failed to read config: {e}"))?;
            contents.parse::<toml::Table>()
                .map_err(|e| format!("Failed to parse config: {e}"))?
        } else {
            toml::Table::new()
        };

        // Validate and insert
        match key {
            "voice" => { table.insert(key.to_string(), toml::Value::String(value.to_string())); }
            "speed" => {
                let v: f32 = value.parse().map_err(|_| format!("Invalid speed: {value}"))?;
                table.insert(key.to_string(), toml::Value::Float(v as f64));
            }
            "model_dir" => { table.insert(key.to_string(), toml::Value::String(value.to_string())); }
            "log_level" => { table.insert(key.to_string(), toml::Value::String(value.to_string())); }
            _ => return Err(format!("Unknown config key: {key}\nValid keys: voice, speed, model_dir, log_level")),
        }

        // Write back
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create config dir: {e}"))?;
        }
        let out = toml::to_string_pretty(&table)
            .map_err(|e| format!("Failed to serialize config: {e}"))?;
        std::fs::write(&path, out)
            .map_err(|e| format!("Failed to write config: {e}"))?;

        Ok(())
    }

    /// Display all config values
    pub fn display_all(&self) -> String {
        format!(
            "voice = \"{}\"\nspeed = {}\nmodel_dir = \"{}\"\nlog_level = \"{}\"",
            self.voice,
            self.speed,
            self.model_dir.display(),
            self.log_level,
        )
    }

    /// Resolve path for the Moonshine model directory
    pub fn moonshine_dir(&self) -> PathBuf {
        self.model_dir
            .join("sherpa-onnx-moonshine-base-en-int8")
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_values() {
        let config = Config::default();
        assert_eq!(config.voice, "af_heart");
        assert!((config.speed - 1.0).abs() < f32::EPSILON);
        assert_eq!(config.log_level, "info");
    }

    #[test]
    fn moonshine_dir_path() {
        let config = Config::default();
        let dir = config.moonshine_dir();
        assert!(
            dir.to_string_lossy()
                .ends_with("sherpa-onnx-moonshine-base-en-int8"),
            "moonshine_dir should end with sherpa-onnx-moonshine-base-en-int8, got: {}",
            dir.display()
        );
    }

    #[test]
    fn kokoro_dir_path() {
        let config = Config::default();
        let dir = config.kokoro_dir();
        assert!(
            dir.to_string_lossy().ends_with("kokoro-multi-lang-v1_0"),
            "kokoro_dir should end with kokoro-multi-lang-v1_0, got: {}",
            dir.display()
        );
    }

    #[test]
    fn vad_model_path_ends_with_onnx() {
        let config = Config::default();
        let path = config.vad_model_path();
        assert!(
            path.to_string_lossy().ends_with("silero_vad.onnx"),
            "vad_model_path should end with silero_vad.onnx, got: {}",
            path.display()
        );
    }

    #[test]
    fn toml_deserialization_with_partial_fields() {
        let toml_str = r#"
            voice = "am_michael"
            speed = 1.5
        "#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.voice, "am_michael");
        assert!((config.speed - 1.5).abs() < f32::EPSILON);
        assert_eq!(config.log_level, "info");
    }

    #[test]
    fn toml_deserialization_empty_uses_all_defaults() {
        let config: Config = toml::from_str("").unwrap();
        assert_eq!(config.voice, "af_heart");
        assert!((config.speed - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn config_speed_tuning() {
        let toml_str = r#"speed = 1.5"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert!((config.speed - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn config_with_custom_model_dir() {
        let toml_str = r#"model_dir = "/tmp/vox-models""#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.model_dir, PathBuf::from("/tmp/vox-models"));
        // Derived paths should use the custom dir
        assert!(config.moonshine_dir().starts_with("/tmp/vox-models"));
        assert!(config.kokoro_dir().starts_with("/tmp/vox-models"));
        assert!(config.vad_model_path().starts_with("/tmp/vox-models"));
    }
}
