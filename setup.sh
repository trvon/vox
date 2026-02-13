#!/usr/bin/env bash
set -euo pipefail

# Vox setup script
# Builds vox, installs binary + dylibs, downloads models, configures daemon

INSTALL_DIR="${1:-${HOME}/.local/bin}"
DATA_DIR="${XDG_DATA_HOME:-${HOME}/.local/share}/vox"
PLIST_LABEL="com.vox.daemon"
PLIST_DIR="${HOME}/Library/LaunchAgents"
PLIST_PATH="${PLIST_DIR}/${PLIST_LABEL}.plist"
DAEMON_PORT=3030

OS="$(uname)"

die() {
    # %b interprets backslash escapes in the message
    printf 'error: %b\n' "$*" >&2
    exit 1
}

info() {
    echo "==> $*"
}

if [[ "${OS}" == "Linux" ]]; then
    if ! command -v pkg-config >/dev/null 2>&1; then
        die "pkg-config is required on Linux. Install it (and ALSA dev headers) then rerun: \n\n  Debian/Ubuntu: sudo apt install -y pkg-config libasound2-dev\n  Fedora:        sudo dnf install -y pkgconf-pkg-config alsa-lib-devel\n  Arch:          sudo pacman -S --needed pkgconf alsa-lib\n  Alpine:        sudo apk add pkgconf alsa-lib-dev\n"
    fi

    if ! pkg-config --exists alsa >/dev/null 2>&1; then
        die "ALSA development files not found (missing alsa.pc). Install ALSA dev headers then rerun: \n\n  Debian/Ubuntu: sudo apt install -y libasound2-dev\n  Fedora:        sudo dnf install -y alsa-lib-devel\n  Arch:          sudo pacman -S --needed alsa-lib\n  Alpine:        sudo apk add alsa-lib-dev\n"
    fi
fi

info "Building vox (release)..."
cargo build --release

info "Installing binary to ${INSTALL_DIR}/"
mkdir -p "${INSTALL_DIR}"
cp target/release/vox "${INSTALL_DIR}/vox"

# Copy all required shared libraries next to binary
shopt -s nullglob
libs=(target/release/*.dylib target/release/*.so target/release/*.so.*)
for lib in "${libs[@]}"; do
    cp "${lib}" "${INSTALL_DIR}/"
    info "Installed $(basename "${lib}")"
done
shopt -u nullglob

info "Downloading models to ${DATA_DIR}/models/..."
"${INSTALL_DIR}/vox" download-models

# --- Daemon setup (macOS launchd) ---
if [[ "${OS}" == "Darwin" ]]; then
    info "Installing launchd plist → ${PLIST_PATH}"
    mkdir -p "${PLIST_DIR}" "${DATA_DIR}"

    # Unload existing plist if loaded
    launchctl bootout "gui/$(id -u)/${PLIST_LABEL}" 2>/dev/null || true

    cat > "${PLIST_PATH}" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${PLIST_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${INSTALL_DIR}/vox</string>
        <string>daemon</string>
        <string>start</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardErrorPath</key>
    <string>${DATA_DIR}/vox-daemon.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>DYLD_LIBRARY_PATH</key>
        <string>${INSTALL_DIR}</string>
    </dict>
</dict>
</plist>
PLIST

    info "Loading daemon..."
    launchctl bootstrap "gui/$(id -u)" "${PLIST_PATH}"

    # Wait briefly for the daemon to start
    sleep 2
    if command -v curl >/dev/null 2>&1 && curl -sf "http://localhost:${DAEMON_PORT}/mcp" -o /dev/null 2>/dev/null; then
        info "Daemon is running on http://localhost:${DAEMON_PORT}/mcp"
    else
        info "Daemon loaded (check ${DATA_DIR}/vox-daemon.log if it fails to start)"
    fi
fi

echo ""
echo "==> Done! MCP client configuration:"
echo ""
echo "  Claude Code (~/.claude.json → mcpServers):"
echo '    "vox": {'
echo "      \"url\": \"http://localhost:${DAEMON_PORT}/mcp\""
echo '    }'
echo ""
echo "  OpenCode (~/.config/opencode/opencode.json → mcp):"
echo '    "vox": {'
echo '      "type": "remote",'
echo "      \"url\": \"http://localhost:${DAEMON_PORT}/mcp\""
echo '    }'
echo ""

# Check if install dir is on PATH
if ! echo "${PATH}" | tr ':' '\n' | grep -qx "${INSTALL_DIR}"; then
    echo "Note: ${INSTALL_DIR} is not on your PATH."
    echo "  Add it:  export PATH=\"${INSTALL_DIR}:\${PATH}\""
fi

if [[ "${OS}" == "Linux" ]]; then
    echo ""
    echo "Linux note: setup.sh does not install a service manager."
    echo "  Start the daemon manually:  ${INSTALL_DIR}/vox daemon start"
fi
