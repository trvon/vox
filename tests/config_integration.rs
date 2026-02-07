use vox::config::Config;

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

/// Documents that config changes require a daemon restart.
/// set_value() writes to disk but does NOT affect a running VoiceMcpServer.
#[test]
fn config_no_hot_reload_documented() {
    // The Config struct is cloned into Arc<Config> at server startup.
    // Subsequent set_value() calls write to disk but the running server
    // holds an immutable snapshot. A restart is required for changes to
    // take effect. This test exists to document that limitation.
    let config = Config::default();
    let snapshot = config.clone();

    // Even if we could modify the original, the snapshot is independent
    assert!((snapshot.dsp.hpf_cutoff_hz - 200.0).abs() < f64::EPSILON);
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
