use crate::config::Config;
use crate::stt::SttEngine;
use crate::tts::TtsEngine;

use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

const DEFAULT_PORT: u16 = 3030;

/// Parsed PID file content: `<pid>:<port>`
#[derive(Debug, Clone, PartialEq)]
pub struct DaemonState {
    pub pid: u32,
    pub port: u16,
}

impl DaemonState {
    pub fn parse(s: &str) -> Option<Self> {
        let (pid_str, port_str) = s.trim().split_once(':')?;
        Some(Self {
            pid: pid_str.parse().ok()?,
            port: port_str.parse().ok()?,
        })
    }

    pub fn serialize(&self) -> String {
        format!("{}:{}", self.pid, self.port)
    }

    /// Check if the process is still alive via kill(pid, 0).
    pub fn is_running(&self) -> bool {
        // SAFETY: signal 0 doesn't send a signal, just checks if process exists
        unsafe { libc::kill(self.pid as i32, 0) == 0 }
    }
}

/// Resolve the PID file path.
/// Prefers `$XDG_RUNTIME_DIR/vox.pid`, falls back to `~/.local/share/vox/vox.pid`.
pub fn pid_file_path() -> PathBuf {
    if let Ok(runtime) = std::env::var("XDG_RUNTIME_DIR") {
        return PathBuf::from(runtime).join("vox.pid");
    }
    dirs::data_local_dir()
        .unwrap_or_else(|| {
            dirs::home_dir()
                .map(|h| h.join(".local").join("share"))
                .unwrap_or_else(|| PathBuf::from("."))
        })
        .join("vox")
        .join("vox.pid")
}

/// Read and validate the PID file. Removes stale files automatically.
pub fn read_state() -> Option<DaemonState> {
    let path = pid_file_path();
    let contents = fs::read_to_string(&path).ok()?;
    let state = DaemonState::parse(&contents)?;

    if state.is_running() {
        Some(state)
    } else {
        // Stale PID file — clean it up
        let _ = fs::remove_file(&path);
        None
    }
}

/// Write the PID file for the current process.
pub fn write_pid_file(port: u16) -> eyre::Result<()> {
    let path = pid_file_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let state = DaemonState {
        pid: std::process::id(),
        port,
    };
    fs::write(&path, state.serialize())?;
    Ok(())
}

/// Remove the PID file.
pub fn remove_pid_file() {
    let _ = fs::remove_file(pid_file_path());
}

/// Daemonize by re-executing the current binary as a detached child process.
///
/// This avoids `fork()`, which is unsafe on macOS when Objective-C frameworks
/// (CoreAudio via cpal) are in use — the ObjC runtime crashes if classes are
/// initialized in a forked child.
pub fn daemonize(port: Option<u16>) -> eyre::Result<()> {
    let exe = std::env::current_exe()?;
    let log_path = log_file_path();
    if let Some(parent) = log_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)?;

    let mut cmd = std::process::Command::new(exe);
    cmd.args(["daemon", "start", "--foreground"]);
    if let Some(p) = port {
        cmd.args(["-p", &p.to_string()]);
    }
    cmd.stdin(std::process::Stdio::null());
    cmd.stdout(std::process::Stdio::null());
    cmd.stderr(log_file);

    let child = cmd.spawn()?;
    eprintln!("Vox daemon started (pid {})", child.id());
    Ok(())
}

/// Resolve the daemon log file path.
pub fn log_file_path() -> PathBuf {
    let data_dir = if let Ok(xdg) = std::env::var("XDG_DATA_HOME") {
        PathBuf::from(xdg).join("vox")
    } else {
        dirs::data_local_dir()
            .unwrap_or_else(|| {
                dirs::home_dir()
                    .map(|h| h.join(".local").join("share"))
                    .unwrap_or_else(|| PathBuf::from("."))
            })
            .join("vox")
    };
    data_dir.join("vox-daemon.log")
}

/// Resolve port from CLI arg > VOX_PORT env > default 3030.
pub fn resolve_port(cli_port: Option<u16>) -> u16 {
    cli_port
        .or_else(|| {
            std::env::var("VOX_PORT")
                .ok()
                .and_then(|p| p.parse().ok())
        })
        .unwrap_or(DEFAULT_PORT)
}

/// Start the HTTP daemon.
pub async fn start(
    tts: TtsEngine,
    stt: SttEngine,
    config: Config,
    port: u16,
) -> eyre::Result<()> {
    use rmcp::transport::streamable_http_server::{
        session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
    };

    let tts = Arc::new(Mutex::new(tts));
    let stt = Arc::new(Mutex::new(stt));
    let config = Arc::new(config);

    let ct = tokio_util::sync::CancellationToken::new();

    let service = StreamableHttpService::new(
        {
            let tts = tts.clone();
            let stt = stt.clone();
            let config = config.clone();
            move || {
                Ok(crate::server::VoiceMcpServer::with_shared(
                    tts.clone(),
                    stt.clone(),
                    config.clone(),
                ))
            }
        },
        Arc::new(LocalSessionManager::default()),
        StreamableHttpServerConfig {
            stateful_mode: true,
            cancellation_token: ct.child_token(),
            ..Default::default()
        },
    );

    let router = axum::Router::new().nest_service("/mcp", service);
    let addr = format!("127.0.0.1:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    // Write PID file after successful bind
    write_pid_file(port)?;
    eprintln!("Vox MCP daemon listening on http://{addr}/mcp");

    let ct_shutdown = ct.clone();
    axum::serve(listener, router)
        .with_graceful_shutdown(async move {
            let mut sigterm =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("failed to install SIGTERM handler");
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {
                    eprintln!("\nShutting down (SIGINT)...");
                }
                _ = sigterm.recv() => {
                    eprintln!("\nShutting down (SIGTERM)...");
                }
            }
            ct_shutdown.cancel();
        })
        .await?;

    remove_pid_file();
    Ok(())
}

/// Stop a running daemon by sending SIGTERM.
pub fn stop_daemon() -> eyre::Result<()> {
    let path = pid_file_path();
    let contents = fs::read_to_string(&path).map_err(|_| eyre::eyre!("No PID file found — daemon is not running"))?;
    let state = DaemonState::parse(&contents)
        .ok_or_else(|| eyre::eyre!("Corrupt PID file"))?;

    if !state.is_running() {
        remove_pid_file();
        eprintln!("Daemon is not running (stale PID file cleaned up)");
        return Ok(());
    }

    eprintln!("Stopping vox daemon (pid {})...", state.pid);
    // SAFETY: sending SIGTERM to a known running process
    unsafe {
        libc::kill(state.pid as i32, libc::SIGTERM);
    }

    // Wait up to 5 seconds for the process to exit
    for _ in 0..50 {
        std::thread::sleep(std::time::Duration::from_millis(100));
        if !state.is_running() {
            remove_pid_file();
            eprintln!("Daemon stopped");
            return Ok(());
        }
    }

    eprintln!("Daemon did not stop within 5 seconds");
    remove_pid_file();
    Ok(())
}

/// Tail the daemon log file for live viewing.
pub fn daemon_log() -> eyre::Result<()> {
    let path = log_file_path();
    if !path.exists() {
        eyre::bail!("No log file found at {}", path.display());
    }
    eprintln!("Tailing {}", path.display());
    let status = std::process::Command::new("tail")
        .args(["-f", &path.to_string_lossy()])
        .status()
        .map_err(|e| eyre::eyre!("Failed to run tail: {e}"))?;
    if !status.success() {
        eyre::bail!("tail exited with status {status}");
    }
    Ok(())
}

/// Print daemon status and return exit code (0 = running, 1 = not running).
pub fn daemon_status() -> i32 {
    match read_state() {
        Some(state) => {
            eprintln!("Vox daemon is running (pid {}, port {})", state.pid, state.port);
            0
        }
        None => {
            eprintln!("Vox daemon is not running");
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn daemon_state_parse_valid() {
        let state = DaemonState::parse("12345:3030").unwrap();
        assert_eq!(state.pid, 12345);
        assert_eq!(state.port, 3030);
    }

    #[test]
    fn daemon_state_parse_with_whitespace() {
        let state = DaemonState::parse("  12345:3030  \n").unwrap();
        assert_eq!(state.pid, 12345);
        assert_eq!(state.port, 3030);
    }

    #[test]
    fn daemon_state_roundtrip() {
        let state = DaemonState {
            pid: 42,
            port: 8080,
        };
        let serialized = state.serialize();
        let parsed = DaemonState::parse(&serialized).unwrap();
        assert_eq!(state, parsed);
    }

    #[test]
    fn daemon_state_parse_invalid_no_colon() {
        assert!(DaemonState::parse("12345").is_none());
    }

    #[test]
    fn daemon_state_parse_invalid_not_numbers() {
        assert!(DaemonState::parse("abc:def").is_none());
    }

    #[test]
    fn daemon_state_parse_empty() {
        assert!(DaemonState::parse("").is_none());
    }

    #[test]
    fn resolve_port_cli_takes_precedence() {
        // SAFETY: test runs single-threaded via cargo test -- --test-threads=1
        // or env var is scoped to this test only
        unsafe { env::set_var("VOX_PORT", "9999") };
        assert_eq!(resolve_port(Some(4000)), 4000);
        unsafe { env::remove_var("VOX_PORT") };
    }

    #[test]
    fn resolve_port_env_fallback() {
        unsafe { env::set_var("VOX_PORT", "9999") };
        assert_eq!(resolve_port(None), 9999);
        unsafe { env::remove_var("VOX_PORT") };
    }

    #[test]
    fn resolve_port_default() {
        unsafe { env::remove_var("VOX_PORT") };
        assert_eq!(resolve_port(None), 3030);
    }

    #[test]
    fn pid_file_write_read_cleanup() {
        // Use a unique temp dir and write/read PID file directly to avoid
        // env var races with parallel tests.
        let tmp = tempfile::tempdir().unwrap();
        let pid_path = tmp.path().join("vox.pid");

        let state = DaemonState {
            pid: std::process::id(),
            port: 3030,
        };
        fs::write(&pid_path, state.serialize()).unwrap();
        assert!(pid_path.exists());

        // Read contents
        let contents = fs::read_to_string(&pid_path).unwrap();
        let parsed = DaemonState::parse(&contents).unwrap();
        assert_eq!(parsed.pid, std::process::id());
        assert_eq!(parsed.port, 3030);

        // Current process should be detected as running
        assert!(parsed.is_running());

        // Cleanup
        fs::remove_file(&pid_path).unwrap();
        assert!(!pid_path.exists());
    }

    #[test]
    fn stale_pid_file_detected() {
        // Write a PID file with a PID that definitely doesn't exist
        // and verify is_running returns false.
        let state = DaemonState {
            pid: 999_999_999,
            port: 3030,
        };
        assert!(!state.is_running());
    }

    #[test]
    fn pid_file_path_uses_xdg_runtime() {
        let tmp = tempfile::tempdir().unwrap();
        unsafe { env::set_var("XDG_RUNTIME_DIR", tmp.path()) };
        let path = pid_file_path();
        assert_eq!(path, tmp.path().join("vox.pid"));
        unsafe { env::remove_var("XDG_RUNTIME_DIR") };
    }
}
