// Items pub in lib.rs for benchmarks/tests may not be used in the binary
#![allow(dead_code)]

mod audio;
mod calibrate;
mod cli;
mod config;
mod daemon;
mod error;
mod models;
mod server;
mod stt;
mod tts;
mod vad;

use clap::Parser;
use cli::{Cli, Command, ConfigAction, DaemonAction};
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
                            "Unknown key: {key}\nValid keys: voice, speed, model_dir, log_level"
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
        Some(Command::DownloadModels) => {
            eprintln!("Downloading models...");
            models::download_models(&config).await?;
            eprintln!("All models downloaded successfully.");
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
    let server = server::VoiceMcpServer::new(tts, stt, config);
    let service = server.serve(rmcp::transport::stdio()).await?;
    service.waiting().await?;
    Ok(())
}
