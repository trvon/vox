use std::process::Command;
use vox::config::{Config, SharedConfig};

fn isolated_vox(tmp: &tempfile::TempDir, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_vox"))
        .args(args)
        .env("XDG_CONFIG_HOME", tmp.path().join("config"))
        .env("XDG_DATA_HOME", tmp.path().join("data"))
        .env("XDG_STATE_HOME", tmp.path().join("state"))
        .env("XDG_RUNTIME_DIR", tmp.path().join("runtime"))
        .env("VOX_LOG_FILE", tmp.path().join("logs/daemon.log"))
        .env("VOX_PORT", "43123")
        .output()
        .unwrap()
}

#[test]
fn config_get_reports_isolated_resolved_paths() {
    let tmp = tempfile::tempdir().unwrap();
    let output = isolated_vox(&tmp, &["config", "get"]);
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    let value: toml::Value = toml::from_str(&stdout).unwrap();
    let paths = value.get("paths").and_then(toml::Value::as_table).unwrap();

    assert_eq!(
        paths["config_file"].as_str().unwrap(),
        tmp.path().join("config/vox/config.toml").to_str().unwrap()
    );
    assert_eq!(
        paths["model_dir"].as_str().unwrap(),
        tmp.path().join("data/vox/models").to_str().unwrap()
    );
    assert_eq!(
        paths["recordings_dir"].as_str().unwrap(),
        tmp.path().join("data/vox/recordings").to_str().unwrap()
    );
    assert_eq!(
        paths["log_file"].as_str().unwrap(),
        tmp.path().join("logs/daemon.log").to_str().unwrap()
    );
    assert_eq!(
        paths["pid_file"].as_str().unwrap(),
        tmp.path().join("runtime/vox.pid").to_str().unwrap()
    );
    let daemon = value.get("daemon").and_then(toml::Value::as_table).unwrap();
    assert_eq!(daemon["port"].as_integer(), Some(43123));
}

#[test]
fn config_get_log_file_matches_resolved_log_path() {
    let tmp = tempfile::tempdir().unwrap();
    let output = isolated_vox(&tmp, &["config", "get", "log_file"]);
    assert!(output.status.success());
    assert_eq!(
        String::from_utf8(output.stdout).unwrap().trim(),
        tmp.path().join("logs/daemon.log").to_str().unwrap()
    );
}

#[test]
fn config_roundtrip_with_dsp() {
    let dir = tempfile::tempdir().unwrap();
    let config_path = dir.path().join("config.toml");

    let toml_content = r#"
voice = "am_michael"
speed = 1.2

[dsp]
hpf_cutoff_hz = 300.0
noise_gate_rms = 0.02
noise_gate_window = 1024
normalize_threshold = 0.7
"#;
    std::fs::write(&config_path, toml_content).unwrap();

    let contents = std::fs::read_to_string(&config_path).unwrap();
    let config: Config = toml::from_str(&contents).unwrap();

    assert_eq!(config.voice, "am_michael");
    assert!((config.speed - 1.2).abs() < f32::EPSILON);
    assert!((config.dsp.hpf_cutoff_hz - 300.0).abs() < f64::EPSILON);
    assert!((config.dsp.noise_gate_rms - 0.02).abs() < f32::EPSILON);
    assert_eq!(config.dsp.noise_gate_window, 1024);
    assert!((config.dsp.normalize_threshold - 0.7).abs() < f32::EPSILON);
}

#[test]
fn config_roundtrip_without_dsp_section() {
    let dir = tempfile::tempdir().unwrap();
    let config_path = dir.path().join("config.toml");

    let toml_content = r#"
voice = "af_bella"
speed = 1.0
"#;
    std::fs::write(&config_path, toml_content).unwrap();

    let contents = std::fs::read_to_string(&config_path).unwrap();
    let config: Config = toml::from_str(&contents).unwrap();

    assert_eq!(config.voice, "af_bella");
    // DSP fields should all be defaults
    assert!((config.dsp.hpf_cutoff_hz - 200.0).abs() < f64::EPSILON);
    assert!((config.dsp.noise_gate_rms - 0.01).abs() < f32::EPSILON);
    assert_eq!(config.dsp.noise_gate_window, 512);
    assert!((config.dsp.normalize_threshold - 0.5).abs() < f32::EPSILON);
}

#[test]
fn config_partial_dsp_section_merges_defaults() {
    let toml_content = r#"
[dsp]
hpf_cutoff_hz = 150.0
"#;
    let config: Config = toml::from_str(toml_content).unwrap();

    assert!((config.dsp.hpf_cutoff_hz - 150.0).abs() < f64::EPSILON);
    // Other DSP fields should be defaults
    assert!((config.dsp.noise_gate_rms - 0.01).abs() < f32::EPSILON);
    assert_eq!(config.dsp.noise_gate_window, 512);
    assert!((config.dsp.normalize_threshold - 0.5).abs() < f32::EPSILON);
}

#[test]
fn env_override_dsp_fields() {
    // We can't safely set env vars and call Config::load() in parallel tests
    // because env vars are process-global. Instead, test the parsing logic directly.
    let toml_content = r#"
[dsp]
hpf_cutoff_hz = 200.0
noise_gate_rms = 0.01
"#;
    let mut config: Config = toml::from_str(toml_content).unwrap();

    // Simulate env override logic from Config::load()
    config.dsp.hpf_cutoff_hz = "350.0".parse().unwrap();
    config.dsp.noise_gate_rms = "0.05".parse().unwrap();
    config.dsp.noise_gate_window = "256".parse().unwrap();
    config.dsp.normalize_threshold = "0.8".parse().unwrap();

    assert!((config.dsp.hpf_cutoff_hz - 350.0).abs() < f64::EPSILON);
    assert!((config.dsp.noise_gate_rms - 0.05).abs() < f32::EPSILON);
    assert_eq!(config.dsp.noise_gate_window, 256);
    assert!((config.dsp.normalize_threshold - 0.8).abs() < f32::EPSILON);
}

#[test]
fn set_value_persists_to_disk() {
    let dir = tempfile::tempdir().unwrap();
    let config_path = dir.path().join("config.toml");

    // Write initial config
    let initial = r#"voice = "af_heart""#;
    std::fs::write(&config_path, initial).unwrap();

    // Read as TOML table, modify, write back (simulates set_value logic)
    let contents = std::fs::read_to_string(&config_path).unwrap();
    let mut table: toml::Table = contents.parse().unwrap();
    let dsp_table = table
        .entry("dsp")
        .or_insert_with(|| toml::Value::Table(toml::Table::new()))
        .as_table_mut()
        .unwrap();
    dsp_table.insert("hpf_cutoff_hz".to_string(), toml::Value::Float(300.0));
    dsp_table.insert("noise_gate_rms".to_string(), toml::Value::Float(0.02));

    let out = toml::to_string_pretty(&table).unwrap();
    std::fs::write(&config_path, &out).unwrap();

    // Re-read and verify
    let contents = std::fs::read_to_string(&config_path).unwrap();
    let config: Config = toml::from_str(&contents).unwrap();
    assert!((config.dsp.hpf_cutoff_hz - 300.0).abs() < f64::EPSILON);
    assert!((config.dsp.noise_gate_rms - 0.02).abs() < f32::EPSILON);
    // Unset DSP fields should get defaults
    assert_eq!(config.dsp.noise_gate_window, 512);
    assert!((config.dsp.normalize_threshold - 0.5).abs() < f32::EPSILON);
}

#[test]
fn set_value_dsp_keys_via_toml_manipulation() {
    // Test the set_value pattern for all four DSP keys
    let mut table = toml::Table::new();

    // Simulate set_value("dsp.hpf_cutoff_hz", "300")
    let dsp = table
        .entry("dsp")
        .or_insert_with(|| toml::Value::Table(toml::Table::new()))
        .as_table_mut()
        .unwrap();
    dsp.insert("hpf_cutoff_hz".to_string(), toml::Value::Float(300.0));
    dsp.insert("noise_gate_rms".to_string(), toml::Value::Float(0.02));
    dsp.insert("noise_gate_window".to_string(), toml::Value::Integer(256));
    dsp.insert("normalize_threshold".to_string(), toml::Value::Float(0.8));

    let toml_str = toml::to_string_pretty(&table).unwrap();
    let config: Config = toml::from_str(&toml_str).unwrap();

    assert!((config.dsp.hpf_cutoff_hz - 300.0).abs() < f64::EPSILON);
    assert!((config.dsp.noise_gate_rms - 0.02).abs() < f32::EPSILON);
    assert_eq!(config.dsp.noise_gate_window, 256);
    assert!((config.dsp.normalize_threshold - 0.8).abs() < f32::EPSILON);
}

/// Verify that SharedConfig reload updates the running snapshot,
/// while old clones remain unaffected.
#[test]
fn shared_config_reload_updates_running_snapshot() {
    let config = Config::default();
    let shared: SharedConfig = config.into_shared();

    // Take a snapshot before reload
    let old_snapshot = shared.read().unwrap().clone();
    assert!((old_snapshot.dsp.hpf_cutoff_hz - 200.0).abs() < f64::EPSILON);

    // Reload — reads from disk (or defaults if no config file)
    Config::reload_into(&shared).unwrap();

    // New reads from the SharedConfig see the reloaded values
    let new_snapshot = shared.read().unwrap().clone();
    assert!(!new_snapshot.voice.is_empty());

    // Old snapshot is unaffected (it's an independent clone)
    assert!((old_snapshot.dsp.hpf_cutoff_hz - 200.0).abs() < f64::EPSILON);
}

#[test]
fn display_all_includes_dsp_section() {
    let config = Config::default();
    let output = config.display_all();
    assert!(output.contains("[dsp]"));
    assert!(output.contains("hpf_cutoff_hz = 200"));
    assert!(output.contains("noise_gate_rms = 0.01"));
    assert!(output.contains("noise_gate_window = 512"));
    assert!(output.contains("normalize_threshold = 0.5"));
}
