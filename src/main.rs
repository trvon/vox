// Items pub in lib.rs for benchmarks/tests may not be used in the binary
#![allow(dead_code)]

mod audio;
mod calibrate;
mod cli;
mod config;
mod daemon;
mod error;
mod models;
mod recording;
mod runtime;
mod server;
mod service;
mod stt;
mod tool_client;
mod transcript;
mod tts;
mod vad;

use clap::Parser;
use cli::{Cli, Command, ConfigAction, DaemonAction, ServiceAction, ToolAction};
use config::Config;
use rmcp::ServiceExt;

fn main() -> eyre::Result<()> {
    // Parse CLI first so --help/--version exit immediately
    let cli = Cli::parse();

    // Daemonize by re-exec with --foreground (parent returns immediately)
    if let Some(Command::Daemon {
        action:
            DaemonAction::Start { foreground, port } | DaemonAction::Restart { foreground, port },
    }) = &cli.command
        && !foreground
    {
        // For restart, stop the old daemon first
        if matches!(
            &cli.command,
            Some(Command::Daemon {
                action: DaemonAction::Restart { .. }
            })
        ) {
            let _ = daemon::stop_daemon();
        }
        daemon::daemonize(*port)?;
        return Ok(());
    }

    // Build the async runtime for the actual work
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(async_main(cli))
}

async fn async_main(cli: Cli) -> eyre::Result<()> {
    let config = Config::load();

    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| config.log_level.parse().unwrap_or_default()),
        )
        .with_writer(std::io::stderr)
        .init();

    match cli.command {
        Some(Command::Service { action }) => match action {
            ServiceAction::Install { no_start } => service::install(no_start)?,
            ServiceAction::Uninstall => service::uninstall()?,
        },
        Some(Command::Config { action }) => {
            match action {
                ConfigAction::Get { key: Some(key) } => match config.get_value(&key) {
                    Some(val) => println!("{val}"),
                    None => {
                        eprintln!(
                            "Unknown key: {key}\nValid keys: voice, speed, model_dir, kokoro_model, log_level"
                        );
                        std::process::exit(1);
                    }
                },
                ConfigAction::Get { key: None } => {
                    println!("{}", config.display_all());
                }
                ConfigAction::Set { key, value } => {
                    Config::set_value(&key, &value).map_err(|e| eyre::eyre!("{e}"))?;
                    // Reload and show the new value
                    let config = Config::load();
                    if let Some(val) = config.get_value(&key) {
                        println!("{key} = {val}");
                    }
                }
                ConfigAction::Path => {
                    println!("{}", Config::config_path().display());
                }
                ConfigAction::ResetDsp => {
                    Config::reset_dsp().map_err(|e| eyre::eyre!("{e}"))?;
                    let defaults = config::DspConfig::default();
                    eprintln!("DSP config reset to defaults:");
                    eprintln!("  hpf_cutoff_hz:       {}", defaults.hpf_cutoff_hz);
                    eprintln!("  noise_gate_rms:      {}", defaults.noise_gate_rms);
                    eprintln!("  noise_gate_window:   {}", defaults.noise_gate_window);
                    eprintln!("  normalize_threshold: {}", defaults.normalize_threshold);
                    eprintln!("\nRestart daemon to apply.");
                }
            }
        }
        Some(Command::DownloadModels { kokoro_model }) => {
            eprintln!("Downloading models...");
            models::download_models_with_variant(
                &config,
                kokoro_model.unwrap_or(config.kokoro_model),
            )
            .await?;
            eprintln!("All models downloaded successfully.");
        }
        Some(Command::Tool { url, action }) => {
            let (tool_name, args) = tool_action_request(&action);
            let result = match tool_client::call_tool(url.as_deref(), tool_name, args).await {
                Ok(ok) => tool_client::print_success(&ok),
                Err(err) => {
                    let cli_err = tool_client::map_error(tool_name, &err);
                    tool_client::print_error(&cli_err)?;
                    std::process::exit(1);
                }
            };
            result?;
        }
        Some(Command::Transcribe {
            path,
            format,
            out,
            title,
            reference,
        }) => {
            if !models::transcription_models_ready(&config) {
                models::download_transcription_models(&config).await?;
            }
            transcribe_local(&config, path, format, out, title, reference).await?;
        }
        Some(Command::Calibrate {
            speech_secs,
            silence_secs,
            population,
            generations,
            dry_run,
        }) => {
            ensure_models(&config).await?;
            audio::init();
            calibrate::run_calibration(
                &config,
                speech_secs,
                silence_secs,
                population,
                generations,
                dry_run,
            )
            .await?;
        }
        Some(Command::Daemon { action }) => match action {
            DaemonAction::Start { port, .. } => {
                let port = daemon::resolve_port(port);

                if let Some(state) = daemon::read_state() {
                    eyre::bail!(
                        "Daemon already running (pid {}, port {})",
                        state.pid,
                        state.port
                    );
                }

                ensure_models(&config).await?;
                let (tts, stt) = init_engines(&config).await?;
                daemon::start(tts, stt, config, port).await?;
            }
            DaemonAction::Restart { port, .. } => {
                // Stop if running (ignore errors if not running)
                let _ = daemon::stop_daemon();
                let port = daemon::resolve_port(port);
                ensure_models(&config).await?;
                let (tts, stt) = init_engines(&config).await?;
                daemon::start(tts, stt, config, port).await?;
            }
            DaemonAction::Stop => {
                daemon::stop_daemon()?;
            }
            DaemonAction::Status => {
                let code = daemon::daemon_status();
                std::process::exit(code);
            }
            DaemonAction::Log => {
                daemon::daemon_log()?;
            }
        },
        None => {
            ensure_models(&config).await?;
            let (tts, stt) = init_engines(&config).await?;
            run_stdio(tts, stt, config).await?;
        }
    }

    Ok(())
}

fn tool_action_request(action: &ToolAction) -> (&'static str, serde_json::Value) {
    match action {
        ToolAction::Say {
            message,
            voice,
            speed,
        } => (
            "say",
            serde_json::json!({
                "message": message,
                "voice": voice,
                "speed": speed,
            }),
        ),
        ToolAction::EnqueueSay {
            message,
            voice,
            speed,
        } => (
            "enqueue_say",
            serde_json::json!({
                "message": message,
                "voice": voice,
                "speed": speed,
            }),
        ),
        ToolAction::TtsQueueStatus => ("tts_queue_status", serde_json::json!({})),
        ToolAction::TtsQueueClear => ("tts_queue_clear", serde_json::json!({})),
        ToolAction::Listen {
            min_speech_ms,
            silence_timeout_ms,
        } => (
            "listen",
            serde_json::json!({
                "min_speech_ms": min_speech_ms,
                "silence_timeout_ms": silence_timeout_ms,
            }),
        ),
        ToolAction::StartListening => ("start_listening", serde_json::json!({})),
        ToolAction::CheckInbox => ("check_inbox", serde_json::json!({})),
        ToolAction::StopListening => ("stop_listening", serde_json::json!({})),
        ToolAction::ResetDsp => ("reset_dsp", serde_json::json!({})),
        ToolAction::ReloadConfig => ("reload_config", serde_json::json!({})),
        ToolAction::Converse {
            message,
            voice,
            speed,
            wait_for_response,
            async_mode,
            min_speech_ms,
            silence_timeout_ms,
        } => (
            "converse",
            serde_json::json!({
                "message": message,
                "voice": voice,
                "speed": speed,
                "wait_for_response": wait_for_response,
                "async_mode": async_mode,
                "min_speech_ms": min_speech_ms,
                "silence_timeout_ms": silence_timeout_ms,
            }),
        ),
        ToolAction::Calibrate {
            speech_secs,
            silence_secs,
            save,
        } => (
            "calibrate",
            serde_json::json!({
                "speech_secs": speech_secs,
                "silence_secs": silence_secs,
                "dry_run": !save,
            }),
        ),
        ToolAction::StartRecording { title, no_audio } => (
            "start_recording",
            serde_json::json!({ "title": title, "save_audio": !no_audio }),
        ),
        ToolAction::StopRecording {
            id,
            cancel,
            no_wait,
        } => (
            "stop_recording",
            serde_json::json!({ "recording_id": id, "cancel": cancel, "wait": !no_wait }),
        ),
        ToolAction::RecordingStatus { id } => (
            "recording_status",
            serde_json::json!({ "recording_id": id }),
        ),
        ToolAction::MarkRecording { label, id } => (
            "mark_recording",
            serde_json::json!({ "label": label, "recording_id": id }),
        ),
        ToolAction::ListRecordings { limit } => {
            ("list_recordings", serde_json::json!({ "limit": limit }))
        }
        ToolAction::TranscribeFile { path, title, wait } => {
            // Resolve relative paths against the caller's cwd, not the daemon's.
            let path = std::path::absolute(path)
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|_| path.clone());
            (
                "transcribe_file",
                serde_json::json!({ "path": path, "title": title, "wait": wait }),
            )
        }
        ToolAction::GetTranscript { id, cursor, limit } => (
            "get_transcript",
            serde_json::json!({ "recording_id": id, "cursor": cursor, "limit": limit }),
        ),
        ToolAction::ExportTranscript {
            id,
            format,
            paragraph_gap_secs,
            section_minutes,
            max_chars,
        } => (
            "export_transcript",
            serde_json::json!({
                "recording_id": id,
                "format": format,
                "paragraph_gap_secs": paragraph_gap_secs,
                "section_minutes": section_minutes,
                "max_chars": max_chars,
            }),
        ),
    }
}

/// Download models if they aren't already present.
async fn ensure_models(config: &Config) -> eyre::Result<()> {
    if !models::models_ready(config) {
        eprintln!("Models not found. Downloading (this only happens once)...");
        models::download_models(config).await?;
        eprintln!("Models downloaded successfully.");
    }
    Ok(())
}

/// Initialize TTS and STT engines in parallel (blocking work on spawn_blocking).
async fn init_engines(config: &Config) -> eyre::Result<(tts::TtsEngine, stt::SttEngine)> {
    eprintln!("Initializing voice engines...");

    let c1 = config.clone();
    let c2 = config.clone();

    let (tts_result, stt_result) = tokio::try_join!(
        async {
            tokio::task::spawn_blocking(move || tts::TtsEngine::new(&c1))
                .await
                .map_err(|e| eyre::eyre!(e))
        },
        async {
            tokio::task::spawn_blocking(move || stt::SttEngine::new(&c2))
                .await
                .map_err(|e| eyre::eyre!(e))
        },
    )?;

    let tts_engine = tts_result?;
    let stt_engine = stt_result?;

    // Eagerly initialize the resampler kernel table
    audio::init();

    Ok((tts_engine, stt_engine))
}

/// `vox transcribe`: run a file job in-process and print the export.
async fn transcribe_local(
    config: &Config,
    path: std::path::PathBuf,
    format: recording::ExportFormat,
    out: Option<std::path::PathBuf>,
    title: Option<String>,
    reference: Option<std::path::PathBuf>,
) -> eyre::Result<()> {
    let t_load = std::time::Instant::now();
    let c = config.clone();
    let stt = tokio::task::spawn_blocking(move || stt::SttEngine::new(&c)).await??;
    let load_ms = t_load.elapsed().as_millis();
    let manager = std::sync::Arc::new(recording::RecordingManager::new(
        config.recordings_dir(),
        std::sync::Arc::new(std::sync::Mutex::new(stt)),
    ));

    let t0 = std::time::Instant::now();
    let (view, mut done) = manager
        .transcribe_file(config, path, title)
        .map_err(|e| eyre::eyre!(e))?;
    let id = view.manifest.id.clone();
    let _ = done.wait_for(|d| *d).await;
    let wall = t0.elapsed();

    let view = manager.view(&id).map_err(|e| eyre::eyre!(e))?;
    if let Some(err) = &view.manifest.error {
        eyre::bail!("transcription failed: {err}");
    }
    let (export_path, content) = manager
        .export(&id, format, &transcript::LayoutOptions::default())
        .map_err(|e| eyre::eyre!(e))?;
    match out {
        Some(p) => std::fs::write(&p, &content)?,
        None => print!("{content}"),
    }

    let m = &view.manifest;
    let audio_s = m.captured_ms as f64 / 1000.0;
    eprintln!(
        "\n{id}: {:.1}s audio, {} segments, {:.1}s speech | wall {:.2}s (RTF {:.3}) | STT {:.2}s (RTF {:.3}) | model load {load_ms} ms\nsaved: {}",
        audio_s,
        m.segment_count,
        m.speech_ms as f64 / 1000.0,
        wall.as_secs_f64(),
        wall.as_secs_f64() / audio_s.max(1e-9),
        m.stt_compute_ms as f64 / 1000.0,
        m.stt_rtf().unwrap_or(0.0),
        export_path.display(),
    );
    if let Some(reference) = reference {
        let reference = std::fs::read_to_string(&reference)?;
        let (text_path, hyp) = manager
            .export(
                &id,
                recording::ExportFormat::Text,
                &transcript::LayoutOptions::default(),
            )
            .map_err(|e| eyre::eyre!(e))?;
        let _ = text_path;
        let w = transcript::word_error_rate(&reference, &hyp);
        eprintln!(
            "WER {:.2}% ({} edits / {} reference words)",
            w.wer() * 100.0,
            w.edits,
            w.reference_words
        );
    }
    Ok(())
}

/// Run as a stdio MCP server (default, backward-compatible).
async fn run_stdio(tts: tts::TtsEngine, stt: stt::SttEngine, config: Config) -> eyre::Result<()> {
    eprintln!("Vox MCP server ready (stdio)");
    let config = config.into_shared();
    let _watcher = config::start_config_watcher(config.clone());
    let runtime = runtime::VoiceRuntime::new(tts, stt, config);
    let server = server::VoiceMcpServer::from_runtime(runtime.clone());
    let service = server.serve(rmcp::transport::stdio()).await?;
    service.waiting().await?;
    // The client went away; finalize any recording it left running.
    runtime.shutdown().await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_action_request_maps_say_arguments() {
        let action = ToolAction::Say {
            message: "hello".to_string(),
            voice: Some("af_heart".to_string()),
            speed: 1.2,
        };
        let (tool, args) = tool_action_request(&action);
        assert_eq!(tool, "say");
        assert_eq!(args["message"], "hello");
        assert_eq!(args["voice"], "af_heart");
        assert!((args["speed"].as_f64().unwrap() - 1.2).abs() < 1e-6);
    }

    #[test]
    fn tool_action_request_maps_calibrate_save_to_dry_run_false() {
        let action = ToolAction::Calibrate {
            speech_secs: Some(10),
            silence_secs: Some(5),
            save: true,
        };
        let (tool, args) = tool_action_request(&action);
        assert_eq!(tool, "calibrate");
        assert_eq!(args["dry_run"], false);
    }

    #[test]
    fn tool_action_request_matches_queue_tools() {
        assert_eq!(
            tool_action_request(&ToolAction::TtsQueueStatus).0,
            "tts_queue_status"
        );
        assert_eq!(
            tool_action_request(&ToolAction::TtsQueueClear).0,
            "tts_queue_clear"
        );
    }
}
