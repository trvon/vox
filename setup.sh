#!/usr/bin/env bash
set -euo pipefail

# Vox setup script
# Builds vox and installs it to ~/.local/bin (or a custom prefix)

INSTALL_DIR="${1:-${HOME}/.local/bin}"
DATA_DIR="${XDG_DATA_HOME:-${HOME}/.local/share}/vox"

echo "==> Building vox (release)..."
cargo build --release

echo "==> Installing binary to ${INSTALL_DIR}/"
mkdir -p "${INSTALL_DIR}"
cp target/release/vox "${INSTALL_DIR}/vox"

# Copy ONNX Runtime shared library next to binary
ONNX_LIB=$(find target/release -maxdepth 1 -name "libonnxruntime*" -not -name "*.a" | head -1)
if [ -n "${ONNX_LIB}" ]; then
    cp "${ONNX_LIB}" "${INSTALL_DIR}/"
    echo "==> Installed ONNX Runtime library"
fi

echo "==> Downloading models to ${DATA_DIR}/models/..."
"${INSTALL_DIR}/vox" --download-models

echo ""
echo "==> Done! Add to Claude Code MCP config:"
echo ""
echo '  {'
echo '    "mcpServers": {'
echo '      "vox": {'
echo "        \"command\": \"${INSTALL_DIR}/vox\","
echo '        "args": []'
echo '      }'
echo '    }'
echo '  }'
echo ""

# Check if install dir is on PATH
if ! echo "${PATH}" | tr ':' '\n' | grep -qx "${INSTALL_DIR}"; then
    echo "Note: ${INSTALL_DIR} is not on your PATH."
    echo "  Add it:  export PATH=\"${INSTALL_DIR}:\${PATH}\""
fi
