"use strict";

const https = require("https");
const http = require("http");
const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");

const VERSION = require("./package.json").version;
const REPO = "trvon/vox";

const PLATFORM_MAP = {
  "darwin-arm64": "vox-aarch64-macos.tar.gz",
  "darwin-x64": "vox-x86_64-macos.tar.gz",
  "linux-arm64": "vox-aarch64-linux.tar.gz",
  "linux-x64": "vox-x86_64-linux.tar.gz",
};

function getArtifactName() {
  const key = `${process.platform}-${process.arch}`;
  const name = PLATFORM_MAP[key];
  if (!name) {
    console.error(
      `Unsupported platform: ${key}\nSupported: ${Object.keys(PLATFORM_MAP).join(", ")}`
    );
    process.exit(1);
  }
  return name;
}

function download(url) {
  return new Promise((resolve, reject) => {
    const client = url.startsWith("https") ? https : http;
    client
      .get(url, { headers: { "User-Agent": "vox-mcp-installer" } }, (res) => {
        // Follow redirects (GitHub releases redirect to S3)
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          return download(res.headers.location).then(resolve, reject);
        }
        if (res.statusCode !== 200) {
          return reject(new Error(`Download failed: HTTP ${res.statusCode} from ${url}`));
        }
        const chunks = [];
        res.on("data", (chunk) => chunks.push(chunk));
        res.on("end", () => resolve(Buffer.concat(chunks)));
        res.on("error", reject);
      })
      .on("error", reject);
  });
}

// Only persistent global installs opt into autostart. In particular, fetching
// this package for an npx invocation must not create a login service.
function maybeInstallService(voxBin, {
  platform = process.platform,
  env = process.env,
  run = execFileSync,
  warn = console.warn,
} = {}) {
  if (platform !== "linux" || env.npm_config_global !== "true" || env.VOX_NO_SERVICE === "1") {
    return;
  }
  try {
    run(voxBin, ["service", "install"], { stdio: "inherit" });
  } catch {
    // Headless/CI installs may have systemctl but no user manager. Do not turn
    // an optional service failure into a failed binary installation.
    warn("Vox binary installed, but autostart could not be enabled. From your login session run: vox-mcp service install (stop any manually started Vox daemon first).");
  }
}

function installArchive(tarball, binDir, run = execFileSync) {
  const staging = fs.mkdtempSync(path.join(path.dirname(binDir), ".vox-install-"));
  try {
    run("tar", ["xzf", tarball, "-C", staging], { stdio: "inherit" });
    const executable = path.join(staging, "vox");
    if (!fs.existsSync(executable)) throw new Error("Release archive is missing the vox binary");
    fs.chmodSync(executable, 0o755);
    fs.mkdirSync(binDir, { recursive: true });
    // Rename rather than truncate a binary/shared library a daemon has mapped.
    for (const name of fs.readdirSync(staging)) {
      fs.renameSync(path.join(staging, name), path.join(binDir, name));
    }
  } finally {
    fs.rmSync(staging, { recursive: true, force: true });
  }
}

async function main() {
  const artifact = getArtifactName();
  const url = `https://github.com/${REPO}/releases/download/v${VERSION}/${artifact}`;
  const binDir = path.join(__dirname, "bin");
  const tarball = path.join(__dirname, artifact);

  console.log(`Downloading vox v${VERSION} for ${process.platform}-${process.arch}...`);
  console.log(`  ${url}`);

  const data = await download(url);
  fs.writeFileSync(tarball, data);

  try {
    installArchive(tarball, binDir);
  } finally {
    fs.unlinkSync(tarball);
  }

  maybeInstallService(path.join(binDir, "vox"));
  console.log("vox installed successfully.");
}

if (require.main === module) {
  main().catch((err) => {
    console.error("Installation failed:", err.message);
    process.exit(1);
  });
}

module.exports = { installArchive, maybeInstallService };
