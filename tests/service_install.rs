//! Model-free installer tests with an isolated HOME and fake systemctl.
#![cfg(target_os = "linux")]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

struct Sandbox {
    root: tempfile::TempDir,
}

impl Sandbox {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir_all(root.path().join("bin")).unwrap();
        fs::create_dir_all(root.path().join("run")).unwrap();
        let systemctl = root.path().join("bin/systemctl");
        fs::write(&systemctl, "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$SYSTEMCTL_LOG\"\n[ \"${FAIL_SYSTEMCTL:-0}\" = 1 ] && exit 1\n[ \"$2\" = is-active ] && exit 1\nexit 0\n").unwrap();
        fs::set_permissions(&systemctl, fs::Permissions::from_mode(0o755)).unwrap();
        Self { root }
    }

    fn unit(&self) -> PathBuf {
        self.root.path().join("config/systemd/user/vox.service")
    }

    fn command(&self, args: &[&str]) -> Command {
        self.command_with_binary(Path::new(env!("CARGO_BIN_EXE_vox")), args)
    }

    fn command_with_binary(&self, binary: &Path, args: &[&str]) -> Command {
        let mut command = Command::new(binary);
        command
            .env_clear()
            .env("HOME", self.root.path())
            .env("XDG_CONFIG_HOME", self.root.path().join("config"))
            .env("XDG_RUNTIME_DIR", self.root.path().join("run"))
            .env("PATH", self.root.path().join("bin"))
            .env("SYSTEMCTL_LOG", self.root.path().join("commands.log"))
            .args(["service"])
            .args(args);
        command
    }

    fn run(&self, args: &[&str]) -> Output {
        self.command(args).output().unwrap()
    }

    fn log(&self) -> String {
        fs::read_to_string(self.root.path().join("commands.log")).unwrap_or_default()
    }
}

fn assert_success(output: Output) {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn install_enables_foreground_unit_and_restarts_it() {
    let sandbox = Sandbox::new();
    assert_success(sandbox.run(&["install"]));
    let unit = fs::read_to_string(sandbox.unit()).unwrap();
    assert!(unit.contains(" daemon start --foreground\n"));
    assert!(unit.contains(env!("CARGO_BIN_EXE_vox")));
    let log = sandbox.log();
    assert!(
        log.contains(
            "--user daemon-reload\n--user enable vox.service\n--user restart vox.service\n"
        )
    );
    // Reinstall is idempotent and uninstall never touches model/recording data.
    assert_success(sandbox.run(&["install"]));
    let data = sandbox
        .root
        .path()
        .join(".local/share/vox/recordings/keep.txt");
    fs::create_dir_all(data.parent().unwrap()).unwrap();
    fs::write(&data, "retained").unwrap();
    assert_success(sandbox.run(&["uninstall"]));
    assert!(!sandbox.unit().exists());
    assert_eq!(fs::read_to_string(data).unwrap(), "retained");
    assert!(sandbox.log().contains("--user disable --now vox.service"));
    assert_success(sandbox.run(&["uninstall"]));
}

#[test]
fn no_start_enables_without_launching_audio_daemon() {
    let sandbox = Sandbox::new();
    assert_success(sandbox.run(&["install", "--no-start"]));
    assert!(sandbox.unit().exists());
    assert!(sandbox.log().contains("--user enable vox.service"));
    assert!(!sandbox.log().contains("restart"));
}

#[test]
fn no_user_manager_fails_without_writing_unit() {
    let sandbox = Sandbox::new();
    let output = sandbox
        .command(&["install"])
        .env("FAIL_SYSTEMCTL", "1")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(!sandbox.unit().exists());
    assert!(String::from_utf8_lossy(&output.stderr).contains("No systemd user manager"));
}

#[test]
fn missing_systemctl_fails_without_writing_unit() {
    let sandbox = Sandbox::new();
    let empty = sandbox.root.path().join("empty-bin");
    fs::create_dir(&empty).unwrap();
    let output = sandbox
        .command(&["install"])
        .env("PATH", empty)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(!sandbox.unit().exists());
    assert!(String::from_utf8_lossy(&output.stderr).contains("Cannot run systemctl"));
}

#[test]
fn unmanaged_units_are_not_overwritten_or_stopped() {
    let sandbox = Sandbox::new();
    fs::create_dir_all(sandbox.unit().parent().unwrap()).unwrap();
    let custom = "[Service]\nExecStart=/custom/vox\n";
    fs::write(sandbox.unit(), custom).unwrap();
    assert!(!sandbox.run(&["install"]).status.success());
    assert!(!sandbox.run(&["uninstall"]).status.success());
    assert_eq!(fs::read_to_string(sandbox.unit()).unwrap(), custom);
    assert!(!sandbox.log().contains("disable"));
    assert!(!sandbox.log().contains("restart"));
}

#[test]
fn systemd_parser_accepts_literal_special_characters_in_binary_path() {
    // Verification only: never connect to the real user manager or start audio.
    if Command::new("systemd-analyze")
        .arg("--version")
        .output()
        .is_err()
    {
        return;
    }
    let sandbox = Sandbox::new();
    let binary = sandbox.root.path().join("literal $HOME %h with spaces/vox");
    fs::create_dir_all(binary.parent().unwrap()).unwrap();
    fs::copy(env!("CARGO_BIN_EXE_vox"), &binary).unwrap();
    assert_success(
        sandbox
            .command_with_binary(&binary, &["install", "--no-start"])
            .output()
            .unwrap(),
    );
    let verify = Command::new("systemd-analyze")
        // CI/headless shells have no login runtime directory. Verification
        // creates its own offline manager; it must not depend on our session.
        .env("HOME", sandbox.root.path())
        .env("XDG_CONFIG_HOME", sandbox.root.path().join("config"))
        .env("XDG_RUNTIME_DIR", sandbox.root.path().join("run"))
        .env_remove("DBUS_SESSION_BUS_ADDRESS")
        .args(["--user", "verify"])
        .arg(sandbox.unit())
        .output()
        .unwrap();
    assert_success(verify);
}

#[test]
fn unmanaged_running_daemon_requires_explicit_handoff() {
    let sandbox = Sandbox::new();
    fs::write(
        sandbox.root.path().join("run/vox.pid"),
        format!("{}:3030", std::process::id()),
    )
    .unwrap();
    let output = sandbox.run(&["install"]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("vox daemon stop"));
    assert!(!sandbox.unit().exists());
    assert!(!sandbox.log().contains("restart"));
}
