use clap::{Parser, Subcommand};

/// Lightweight voice MCP server with local Moonshine + Kokoro inference
#[derive(Parser)]
#[command(version, about)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Manage the HTTP daemon
    Daemon {
        #[command(subcommand)]
        action: DaemonAction,
    },
    /// Get or set configuration values
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
    /// Download models and exit
    DownloadModels,
}

#[derive(Debug, Subcommand)]
pub enum ConfigAction {
    /// Show a config value (or all values if no key given)
    Get {
        /// Config key (e.g. voice, speed, model_dir, log_level)
        key: Option<String>,
    },
    /// Set a config value
    Set {
        /// Config key (e.g. voice, speed, model_dir, log_level)
        key: String,
        /// Value to set
        value: String,
    },
    /// Show the config file path
    Path,
}

#[derive(Debug, Subcommand)]
pub enum DaemonAction {
    /// Start the HTTP daemon (backgrounds by default)
    Start {
        /// Port to listen on (default: 3030, or VOX_PORT env)
        #[arg(short, long)]
        port: Option<u16>,
        /// Run in foreground instead of daemonizing
        #[arg(short, long)]
        foreground: bool,
    },
    /// Stop and restart the daemon
    Restart {
        /// Port to listen on (default: 3030, or VOX_PORT env)
        #[arg(short, long)]
        port: Option<u16>,
        /// Run in foreground instead of daemonizing
        #[arg(short, long)]
        foreground: bool,
    },
    /// Stop a running daemon
    Stop,
    /// Check if the daemon is running
    Status,
    /// Tail the daemon log file
    Log,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn parse_no_args() {
        let cli = Cli::try_parse_from(["vox"]).unwrap();
        assert!(cli.command.is_none());
    }

    #[test]
    fn parse_daemon_start() {
        let cli = Cli::try_parse_from(["vox", "daemon", "start", "-p", "8080"]).unwrap();
        match cli.command {
            Some(Command::Daemon { action: DaemonAction::Start { port, foreground } }) => {
                assert_eq!(port, Some(8080));
                assert!(!foreground);
            }
            other => panic!("Expected Daemon Start, got {other:?}"),
        }
    }

    #[test]
    fn parse_daemon_start_foreground() {
        let cli = Cli::try_parse_from(["vox", "daemon", "start", "--foreground"]).unwrap();
        match cli.command {
            Some(Command::Daemon { action: DaemonAction::Start { port, foreground } }) => {
                assert!(port.is_none());
                assert!(foreground);
            }
            other => panic!("Expected Daemon Start, got {other:?}"),
        }
    }

    #[test]
    fn parse_download_models() {
        let cli = Cli::try_parse_from(["vox", "download-models"]).unwrap();
        assert!(matches!(cli.command, Some(Command::DownloadModels)));
    }

    #[test]
    fn parse_daemon_stop() {
        let cli = Cli::try_parse_from(["vox", "daemon", "stop"]).unwrap();
        match cli.command {
            Some(Command::Daemon { action: DaemonAction::Stop }) => {}
            other => panic!("Expected Daemon Stop, got {other:?}"),
        }
    }

    #[test]
    fn parse_daemon_status() {
        let cli = Cli::try_parse_from(["vox", "daemon", "status"]).unwrap();
        match cli.command {
            Some(Command::Daemon { action: DaemonAction::Status }) => {}
            other => panic!("Expected Daemon Status, got {other:?}"),
        }
    }

    #[test]
    fn parse_daemon_log() {
        let cli = Cli::try_parse_from(["vox", "daemon", "log"]).unwrap();
        match cli.command {
            Some(Command::Daemon { action: DaemonAction::Log }) => {}
            other => panic!("Expected Daemon Log, got {other:?}"),
        }
    }

    #[test]
    fn parse_config_get_all() {
        let cli = Cli::try_parse_from(["vox", "config", "get"]).unwrap();
        match cli.command {
            Some(Command::Config { action: ConfigAction::Get { key } }) => {
                assert!(key.is_none());
            }
            other => panic!("Expected Config Get, got {other:?}"),
        }
    }

    #[test]
    fn parse_config_get_key() {
        let cli = Cli::try_parse_from(["vox", "config", "get", "voice"]).unwrap();
        match cli.command {
            Some(Command::Config { action: ConfigAction::Get { key } }) => {
                assert_eq!(key.as_deref(), Some("voice"));
            }
            other => panic!("Expected Config Get, got {other:?}"),
        }
    }

    #[test]
    fn parse_config_set() {
        let cli = Cli::try_parse_from(["vox", "config", "set", "speed", "1.3"]).unwrap();
        match cli.command {
            Some(Command::Config { action: ConfigAction::Set { key, value } }) => {
                assert_eq!(key, "speed");
                assert_eq!(value, "1.3");
            }
            other => panic!("Expected Config Set, got {other:?}"),
        }
    }

    #[test]
    fn parse_config_path() {
        let cli = Cli::try_parse_from(["vox", "config", "path"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Config { action: ConfigAction::Path })
        ));
    }

    #[test]
    fn parse_daemon_restart() {
        let cli = Cli::try_parse_from(["vox", "daemon", "restart"]).unwrap();
        match cli.command {
            Some(Command::Daemon {
                action: DaemonAction::Restart { port, foreground },
            }) => {
                assert!(port.is_none());
                assert!(!foreground);
            }
            other => panic!("Expected Daemon Restart, got {other:?}"),
        }
    }
}
