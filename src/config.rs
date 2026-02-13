use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

const APP_NAME: &str = "vox";

/// Thread-safe shared config: readable by many, writable on reload.
pub type SharedConfig = Arc<RwLock<Config>>;

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(default)]
pub struct DspConfig {
    pub hpf_cutoff_hz: f64,
    pub noise_gate_rms: f32,
    pub noise_gate_window: usize,
    pub normalize_threshold: f32,
}

impl Default for DspConfig {
    fn default() -> Self {
        Self {
            hpf_cutoff_hz: 200.0,
            noise_gate_rms: 0.01,
            noise_gate_window: 512,
            normalize_threshold: 0.5,
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(default)]
pub struct Config {
    pub model_dir: PathBuf,
    pub voice: String,
    pub speed: f32,
    pub log_level: String,
    pub tts_num_threads: Option<i32>,
    pub dsp: DspConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model_dir: default_model_dir(),
            voice: "af_heart".to_string(),
            speed: 1.4,
            log_level: "info".to_string(),
            tts_num_threads: None,
            dsp: DspConfig::default(),
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
    /// Wrap this config in a SharedConfig (Arc<RwLock<Config>>).
    pub fn into_shared(self) -> SharedConfig {
        Arc::new(RwLock::new(self))
    }

    /// Reload config from disk and swap it into the shared handle.
    pub fn reload_into(shared: &SharedConfig) -> Result<(), String> {
        let new = Config::load();
        *shared
            .write()
            .map_err(|e| format!("Config lock poisoned: {e}"))? = new;
        tracing::info!("Config reloaded from disk");
        Ok(())
    }

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
        if let Ok(val) = std::env::var("VOX_TTS_THREADS")
            && let Ok(threads) = val.parse()
        {
            config.tts_num_threads = Some(threads);
        }

        // DSP env overrides
        if let Ok(val) = std::env::var("VOX_DSP_HPF_CUTOFF_HZ")
            && let Ok(v) = val.parse()
        {
            config.dsp.hpf_cutoff_hz = v;
        }
        if let Ok(val) = std::env::var("VOX_DSP_NOISE_GATE_RMS")
            && let Ok(v) = val.parse()
        {
            config.dsp.noise_gate_rms = v;
        }
        if let Ok(val) = std::env::var("VOX_DSP_NOISE_GATE_WINDOW")
            && let Ok(v) = val.parse()
        {
            config.dsp.noise_gate_window = v;
        }
        if let Ok(val) = std::env::var("VOX_DSP_NORMALIZE_THRESHOLD")
            && let Ok(v) = val.parse()
        {
            config.dsp.normalize_threshold = v;
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
            "dsp.hpf_cutoff_hz" => Some(self.dsp.hpf_cutoff_hz.to_string()),
            "dsp.noise_gate_rms" => Some(self.dsp.noise_gate_rms.to_string()),
            "dsp.noise_gate_window" => Some(self.dsp.noise_gate_window.to_string()),
            "dsp.normalize_threshold" => Some(self.dsp.normalize_threshold.to_string()),
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
            contents
                .parse::<toml::Table>()
                .map_err(|e| format!("Failed to parse config: {e}"))?
        } else {
            toml::Table::new()
        };

        // Validate and insert
        match key {
            "voice" => {
                table.insert(key.to_string(), toml::Value::String(value.to_string()));
            }
            "speed" => {
                let v: f32 = value
                    .parse()
                    .map_err(|_| format!("Invalid speed: {value}"))?;
                table.insert(key.to_string(), toml::Value::Float(v as f64));
            }
            "model_dir" => {
                table.insert(key.to_string(), toml::Value::String(value.to_string()));
            }
            "log_level" => {
                table.insert(key.to_string(), toml::Value::String(value.to_string()));
            }
            k if k.starts_with("dsp.") => {
                let dsp_table = table
                    .entry("dsp")
                    .or_insert_with(|| toml::Value::Table(toml::Table::new()))
                    .as_table_mut()
                    .ok_or_else(|| "dsp is not a table".to_string())?;
                let field = &k[4..];
                match field {
                    "hpf_cutoff_hz" => {
                        let v: f64 = value
                            .parse()
                            .map_err(|_| format!("Invalid hpf_cutoff_hz: {value}"))?;
                        dsp_table.insert(field.to_string(), toml::Value::Float(v));
                    }
                    "noise_gate_rms" => {
                        let v: f64 = value
                            .parse::<f32>()
                            .map_err(|_| format!("Invalid noise_gate_rms: {value}"))?
                            as f64;
                        dsp_table.insert(field.to_string(), toml::Value::Float(v));
                    }
                    "noise_gate_window" => {
                        let v: i64 = value
                            .parse()
                            .map_err(|_| format!("Invalid noise_gate_window: {value}"))?;
                        dsp_table.insert(field.to_string(), toml::Value::Integer(v));
                    }
                    "normalize_threshold" => {
                        let v: f64 = value
                            .parse::<f32>()
                            .map_err(|_| format!("Invalid normalize_threshold: {value}"))?
                            as f64;
                        dsp_table.insert(field.to_string(), toml::Value::Float(v));
                    }
                    _ => {
                        return Err(format!(
                            "Unknown DSP key: {field}\nValid DSP keys: hpf_cutoff_hz, noise_gate_rms, noise_gate_window, normalize_threshold"
                        ));
                    }
                }
            }
            _ => {
                return Err(format!(
                    "Unknown config key: {key}\nValid keys: voice, speed, model_dir, log_level, dsp.hpf_cutoff_hz, dsp.noise_gate_rms, dsp.noise_gate_window, dsp.normalize_threshold"
                ));
            }
        }

        // Write back
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create config dir: {e}"))?;
        }
        let out = toml::to_string_pretty(&table)
            .map_err(|e| format!("Failed to serialize config: {e}"))?;
        std::fs::write(&path, out).map_err(|e| format!("Failed to write config: {e}"))?;

        Ok(())
    }

    /// Remove the `[dsp]` section from the config file, reverting to defaults on next load.
    pub fn reset_dsp() -> std::result::Result<(), String> {
        let path = config_path();
        if !path.exists() {
            return Ok(()); // no config file = already defaults
        }
        let contents =
            std::fs::read_to_string(&path).map_err(|e| format!("Failed to read config: {e}"))?;
        let mut table: toml::Table = contents
            .parse()
            .map_err(|e| format!("Failed to parse config: {e}"))?;
        table.remove("dsp");
        let new_contents = toml::to_string_pretty(&table)
            .map_err(|e| format!("Failed to serialize config: {e}"))?;
        std::fs::write(&path, new_contents).map_err(|e| format!("Failed to write config: {e}"))?;
        Ok(())
    }

    /// Display all config values
    pub fn display_all(&self) -> String {
        format!(
            "voice = \"{}\"\nspeed = {}\nmodel_dir = \"{}\"\nlog_level = \"{}\"\n\n[dsp]\nhpf_cutoff_hz = {}\nnoise_gate_rms = {}\nnoise_gate_window = {}\nnormalize_threshold = {}",
            self.voice,
            self.speed,
            self.model_dir.display(),
            self.log_level,
            self.dsp.hpf_cutoff_hz,
            self.dsp.noise_gate_rms,
            self.dsp.noise_gate_window,
            self.dsp.normalize_threshold,
        )
    }

    /// Resolve path for the Moonshine model directory
    pub fn moonshine_dir(&self) -> PathBuf {
        self.model_dir.join("sherpa-onnx-moonshine-base-en-int8")
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

/// Start a file watcher on the config TOML. Returns the watcher handle — caller
/// must keep it alive (drop stops watching). Returns `None` if the watcher
/// cannot be created (e.g. parent dir doesn't exist yet).
pub fn start_config_watcher(shared: SharedConfig) -> Option<notify::RecommendedWatcher> {
    use notify::{EventKind, RecursiveMode, Watcher};

    let config_file = config_path();
    let parent = config_file.parent()?.to_path_buf();
    let file_name = config_file.file_name()?.to_os_string();

    let (tx, rx) = std::sync::mpsc::channel();

    let mut watcher = notify::recommended_watcher(move |res: Result<notify::Event, _>| {
        if let Ok(event) = res {
            match event.kind {
                EventKind::Create(_) | EventKind::Modify(_) => {
                    // Only care about our config file (editors may rename temp files in parent dir)
                    let dominated = event
                        .paths
                        .iter()
                        .any(|p| p.file_name().map(|n| n == file_name).unwrap_or(false));
                    if dominated {
                        let _ = tx.send(());
                    }
                }
                _ => {}
            }
        }
    })
    .ok()?;

    // Watch parent dir (handles atomic renames from editors like vim/emacs)
    watcher.watch(&parent, RecursiveMode::NonRecursive).ok()?;

    // Background thread: debounce events and reload
    std::thread::Builder::new()
        .name("config-watcher".into())
        .spawn(move || {
            loop {
                // Block until first event
                if rx.recv().is_err() {
                    break; // sender dropped (watcher dropped)
                }
                // Debounce: drain for 500ms
                let deadline = std::time::Instant::now() + std::time::Duration::from_millis(500);
                loop {
                    let remaining = deadline.saturating_duration_since(std::time::Instant::now());
                    if remaining.is_zero() {
                        break;
                    }
                    if rx.recv_timeout(remaining).is_err() {
                        break; // timeout or sender dropped
                    }
                }
                // Reload
                if let Err(e) = Config::reload_into(&shared) {
                    tracing::error!("Config hot-reload failed: {e}");
                }
            }
        })
        .ok()?;

    Some(watcher)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_values() {
        let config = Config::default();
        assert_eq!(config.voice, "af_heart");
        assert!((config.speed - 1.4).abs() < f32::EPSILON);
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
        assert!((config.speed - 1.4).abs() < f32::EPSILON);
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

    // --- DspConfig tests ---

    #[test]
    fn dsp_config_defaults() {
        let dsp = DspConfig::default();
        assert!((dsp.hpf_cutoff_hz - 200.0).abs() < f64::EPSILON);
        assert!((dsp.noise_gate_rms - 0.01).abs() < f32::EPSILON);
        assert_eq!(dsp.noise_gate_window, 512);
        assert!((dsp.normalize_threshold - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn toml_with_dsp_section() {
        let toml_str = r#"
            [dsp]
            hpf_cutoff_hz = 300.0
            noise_gate_rms = 0.02
        "#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert!((config.dsp.hpf_cutoff_hz - 300.0).abs() < f64::EPSILON);
        assert!((config.dsp.noise_gate_rms - 0.02).abs() < f32::EPSILON);
        // Unset fields get defaults
        assert_eq!(config.dsp.noise_gate_window, 512);
        assert!((config.dsp.normalize_threshold - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn toml_without_dsp_section_uses_defaults() {
        let toml_str = r#"voice = "am_michael""#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert!((config.dsp.hpf_cutoff_hz - 200.0).abs() < f64::EPSILON);
        assert!((config.dsp.noise_gate_rms - 0.01).abs() < f32::EPSILON);
        assert_eq!(config.dsp.noise_gate_window, 512);
        assert!((config.dsp.normalize_threshold - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn get_value_dsp_keys() {
        let config = Config::default();
        assert_eq!(config.get_value("dsp.hpf_cutoff_hz").unwrap(), "200");
        assert_eq!(config.get_value("dsp.noise_gate_rms").unwrap(), "0.01");
        assert_eq!(config.get_value("dsp.noise_gate_window").unwrap(), "512");
        assert_eq!(config.get_value("dsp.normalize_threshold").unwrap(), "0.5");
    }

    #[test]
    fn get_value_unknown_returns_none() {
        let config = Config::default();
        assert!(config.get_value("dsp.nonexistent").is_none());
        assert!(config.get_value("nonexistent").is_none());
    }

    /// Verify that reload_into swaps config in a SharedConfig.
    #[test]
    fn reload_into_updates_shared_config() {
        let config = Config::default();
        let shared = config.into_shared();

        // Before reload: default voice
        assert_eq!(shared.read().unwrap().voice, "af_heart");

        // reload_into re-reads from disk (or defaults if no file), so the
        // SharedConfig should still have valid data after reload.
        Config::reload_into(&shared).unwrap();
        let reloaded = shared.read().unwrap();
        // Voice should still be valid (either from disk or default)
        assert!(!reloaded.voice.is_empty());
    }
}
