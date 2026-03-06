// Items pub in lib.rs for benchmarks/tests may not be used in the binary
#![allow(dead_code)]

mod audio;
mod calibrate;
mod cli;
mod config;
mod daemon;
mod error;
mod models;
mod runtime;
mod server;
mod stt;
mod tool_client;
mod tts;
mod vad;

use clap::Parser;
use cli::{Cli, Command, ConfigAction, DaemonAction, ToolAction};
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
            let tool_name = tool_action_name(&action);
            let result = match handle_tool_command(url.as_deref(), action).await {
                Ok(ok) => tool_client::print_success(&ok),
                Err(err) => {
                    let cli_err = tool_client::map_error(tool_name, &err);
                    tool_client::print_error(&cli_err)?;
                    std::process::exit(1);
                }
            };
            result?;
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

async fn handle_tool_command(
    url: Option<&str>,
    action: ToolAction,
) -> eyre::Result<tool_client::CliSuccess> {
    let (tool_name, args) = tool_action_request(&action);
    tool_client::call_tool(url, tool_name, args).await
}

fn tool_action_name(action: &ToolAction) -> &'static str {
    match action {
        ToolAction::Say { .. } => "say",
        ToolAction::EnqueueSay { .. } => "enqueue_say",
        ToolAction::TtsQueueStatus => "tts_queue_status",
        ToolAction::TtsQueueClear => "tts_queue_clear",
        ToolAction::Listen { .. } => "listen",
        ToolAction::StartListening => "start_listening",
        ToolAction::CheckInbox => "check_inbox",
        ToolAction::StopListening => "stop_listening",
        ToolAction::ResetDsp => "reset_dsp",
        ToolAction::ReloadConfig => "reload_config",
        ToolAction::Converse { .. } => "converse",
        ToolAction::Calibrate { .. } => "calibrate",
    }
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

/// Run as a stdio MCP server (default, backward-compatible).
async fn run_stdio(tts: tts::TtsEngine, stt: stt::SttEngine, config: Config) -> eyre::Result<()> {
    eprintln!("Vox MCP server ready (stdio)");
    let config = config.into_shared();
    let _watcher = config::start_config_watcher(config.clone());
    let server = server::VoiceMcpServer::new(tts, stt, config);
    let service = server.serve(rmcp::transport::stdio()).await?;
    service.waiting().await?;
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
    fn tool_action_name_matches_queue_tools() {
        assert_eq!(
            tool_action_name(&ToolAction::TtsQueueStatus),
            "tts_queue_status"
        );
        assert_eq!(
            tool_action_name(&ToolAction::TtsQueueClear),
            "tts_queue_clear"
        );
    }
}
