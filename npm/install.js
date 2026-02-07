"use strict";

const https = require("https");
const http = require("http");
const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");

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

async function main() {
  const artifact = getArtifactName();
  const url = `https://github.com/${REPO}/releases/download/v${VERSION}/${artifact}`;
  const binDir = path.join(__dirname, "bin");
  const tarball = path.join(__dirname, artifact);

  console.log(`Downloading vox v${VERSION} for ${process.platform}-${process.arch}...`);
  console.log(`  ${url}`);

  const data = await download(url);
  fs.writeFileSync(tarball, data);

  // Extract
  fs.mkdirSync(binDir, { recursive: true });
  execSync(`tar xzf "${tarball}" -C "${binDir}"`, { stdio: "inherit" });

  // Cleanup tarball
  fs.unlinkSync(tarball);

  // Make binary executable
  const voxBin = path.join(binDir, "vox");
  if (fs.existsSync(voxBin)) {
    fs.chmodSync(voxBin, 0o755);
  }

  console.log("vox installed successfully.");
}

main().catch((err) => {
  console.error("Installation failed:", err.message);
  process.exit(1);
});
