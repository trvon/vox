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

echo "==> Building vox (release)..."
cargo build --release

echo "==> Installing binary to ${INSTALL_DIR}/"
mkdir -p "${INSTALL_DIR}"
cp target/release/vox "${INSTALL_DIR}/vox"

# Copy all required shared libraries next to binary
for lib in target/release/*.dylib target/release/*.so target/release/*.so.* 2>/dev/null; do
    [ -f "${lib}" ] && cp "${lib}" "${INSTALL_DIR}/" && echo "==> Installed $(basename "${lib}")"
done

echo "==> Downloading models to ${DATA_DIR}/models/..."
"${INSTALL_DIR}/vox" download-models

# --- Daemon setup (macOS launchd) ---
if [[ "$(uname)" == "Darwin" ]]; then
    echo "==> Installing launchd plist → ${PLIST_PATH}"
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

    echo "==> Loading daemon..."
    launchctl bootstrap "gui/$(id -u)" "${PLIST_PATH}"

    # Wait briefly for the daemon to start
    sleep 2
    if curl -sf "http://localhost:${DAEMON_PORT}/mcp" -o /dev/null 2>/dev/null; then
        echo "==> Daemon is running on http://localhost:${DAEMON_PORT}/mcp"
    else
        echo "==> Daemon loaded (check ${DATA_DIR}/vox-daemon.log if it fails to start)"
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
