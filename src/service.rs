//! Installation of a per-user systemd service. No audio engines are loaded here.
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const UNIT: &str = "vox.service";
const TEMPLATE: &str = include_str!("../systemd/vox.service.in");
const MANAGED_HEADER: &str = "# Managed by vox service install.";

fn unit_path() -> eyre::Result<PathBuf> {
    let config =
        dirs::config_dir().ok_or_else(|| eyre::eyre!("Cannot resolve user config directory"))?;
    Ok(config.join("systemd/user").join(UNIT))
}

fn systemctl(args: &[&str]) -> eyre::Result<()> {
    let status = Command::new("systemctl")
        .arg("--user")
        .args(args)
        .status()?;
    eyre::ensure!(
        status.success(),
        "systemctl --user {} failed: {status}",
        args.join(" ")
    );
    Ok(())
}

fn check_manager() -> eyre::Result<()> {
    eyre::ensure!(
        cfg!(target_os = "linux"),
        "Systemd user services are supported only on Linux"
    );
    let status = Command::new("systemctl")
        .args(["--user", "show", "--property=Version", "--value"])
        .stdout(Stdio::null())
        .status()
        .map_err(|err| {
            eyre::eyre!("Cannot run systemctl: {err}. Install systemd or run Vox manually.")
        })?;
    eyre::ensure!(
        status.success(),
        "No systemd user manager is available. Run this command from your login session."
    );
    Ok(())
}

fn render_unit(executable: &Path) -> eyre::Result<String> {
    eyre::ensure!(
        executable.is_absolute(),
        "Service executable must be an absolute path"
    );
    let executable = executable
        .to_str()
        .ok_or_else(|| eyre::eyre!("Executable path is not UTF-8"))?;
    // Quote the entire executable token, not a shell command. The template's
    // ':' prefix disables environment substitution; percent specifiers still
    // need escaping (including in the executable path itself).
    eyre::ensure!(
        !executable
            .chars()
            .any(|c| c.is_control() || matches!(c, '"' | '\'' | '\\')),
        "Executable path contains quotes, backslashes, or control characters unsupported by systemd"
    );
    let escaped = executable.replace('%', "%%");
    Ok(TEMPLATE.replace("@VOX_EXECUTABLE@", &format!("\"{escaped}\"")))
}

/// Refuse to overwrite/delete hand-written or symlinked units (e.g. dotfiles).
fn check_owned(path: &Path) -> eyre::Result<bool> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            eyre::ensure!(
                !metadata.file_type().is_symlink(),
                "Refusing to modify symlinked unit {}",
                path.display()
            );
            let content = fs::read_to_string(path)?;
            eyre::ensure!(
                content.starts_with(MANAGED_HEADER),
                "Refusing to modify unmanaged unit {}. Back it up and remove it first.",
                path.display()
            );
            Ok(true)
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(err) => Err(err.into()),
    }
}

pub fn install(no_start: bool) -> eyre::Result<()> {
    check_manager()?;
    let path = unit_path()?;
    check_owned(&path)?;
    if !no_start && crate::daemon::read_state().is_some() {
        let managed = Command::new("systemctl")
            .args(["--user", "is-active", "--quiet", UNIT])
            .status()?;
        eyre::ensure!(
            managed.success(),
            "An unmanaged Vox daemon is running. Stop it with `vox daemon stop`, then retry `vox service install`."
        );
    }
    let unit = render_unit(&std::env::current_exe()?)?;
    fs::create_dir_all(path.parent().expect("unit has parent directory"))?;
    fs::write(&path, unit)?;
    systemctl(&["daemon-reload"])?;
    systemctl(&["enable", UNIT])?;
    if !no_start {
        // Restart also starts inactive units and picks up a newly installed binary.
        systemctl(&["restart", UNIT])?;
    }
    eprintln!("Installed {} (enabled at user login)", path.display());
    if no_start {
        eprintln!("Start now with: systemctl --user start {UNIT}");
    } else {
        eprintln!("Check readiness: systemctl --user status {UNIT}");
    }
    eprintln!("Logs: journalctl --user -u {UNIT}");
    Ok(())
}

pub fn uninstall() -> eyre::Result<()> {
    check_manager()?;
    let path = unit_path()?;
    if !check_owned(&path)? {
        eprintln!("{} is not installed", path.display());
        return Ok(());
    }
    systemctl(&["disable", "--now", UNIT])?;
    fs::remove_file(&path)?;
    systemctl(&["daemon-reload"])?;
    eprintln!(
        "Removed {}. Models and recordings were retained.",
        path.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_runs_in_foreground_with_absolute_executable() {
        let unit = render_unit(Path::new("/opt/vox/bin/vox")).unwrap();
        assert!(unit.contains("ExecStart=:\"/opt/vox/bin/vox\" daemon start --foreground\n"));
        assert!(unit.contains("WantedBy=default.target"));
        assert!(unit.contains("Restart=on-failure"));
        assert!(!unit.contains("@VOX_EXECUTABLE@"));
    }

    #[test]
    fn unit_escapes_specifiers_and_preserves_literal_dollars() {
        let unit = render_unit(Path::new("/tmp/a b/%h/$HOME/vox")).unwrap();
        assert!(unit.contains("ExecStart=:\"/tmp/a b/%%h/$HOME/vox\""));
    }

    #[test]
    fn unit_rejects_relative_paths_and_control_characters() {
        assert!(render_unit(Path::new("vox")).is_err());
        for path in [
            "/tmp/bad\npath/vox",
            "/tmp/\"quoted\"/vox",
            "/tmp/a'b/vox",
            "/tmp/a\\b/vox",
        ] {
            assert!(render_unit(Path::new(path)).is_err());
        }
    }

    #[test]
    fn ownership_check_protects_custom_units() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join(UNIT);
        assert!(!check_owned(&path).unwrap());
        fs::write(&path, "[Service]\nExecStart=/bin/true\n").unwrap();
        assert!(check_owned(&path).is_err());
        fs::write(&path, render_unit(Path::new("/opt/vox")).unwrap()).unwrap();
        assert!(check_owned(&path).unwrap());
    }

    #[cfg(unix)]
    #[test]
    fn ownership_check_protects_symlinked_units() {
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("dotfiles.service");
        fs::write(&target, render_unit(Path::new("/opt/vox")).unwrap()).unwrap();
        let path = tmp.path().join(UNIT);
        std::os::unix::fs::symlink(&target, &path).unwrap();
        assert!(check_owned(&path).is_err());
    }
}
