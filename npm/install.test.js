"use strict";

const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { installArchive, maybeInstallService } = require("./install.js");

test("archive installation atomically replaces an open binary", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "vox-npm-install-"));
  let descriptor;
  try {
    const bin = path.join(root, "bin");
    fs.mkdirSync(bin);
    const executable = path.join(bin, "vox");
    fs.writeFileSync(executable, "old binary");
    descriptor = fs.openSync(executable, "r");
    installArchive("archive with spaces.tar.gz", bin, (command, args) => {
      assert.equal(command, "tar");
      assert.equal(args[1], "archive with spaces.tar.gz");
      fs.writeFileSync(path.join(args[3], "vox"), "new binary");
    });
    assert.equal(fs.readFileSync(executable, "utf8"), "new binary");
    assert.equal(fs.readFileSync(descriptor, "utf8"), "old binary");
    assert.equal(fs.statSync(executable).mode & 0o777, 0o755);
    assert.deepEqual(fs.readdirSync(root), ["bin"]);
  } finally {
    if (descriptor !== undefined) fs.closeSync(descriptor);
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("bad archives preserve the installed binary and clean staging files", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "vox-npm-invalid-"));
  try {
    const bin = path.join(root, "bin");
    fs.mkdirSync(bin);
    fs.writeFileSync(path.join(bin, "vox"), "old binary");
    assert.throws(() => installArchive("bad.tar.gz", bin, () => {}), /missing the vox binary/);
    assert.equal(fs.readFileSync(path.join(bin, "vox"), "utf8"), "old binary");
    assert.deepEqual(fs.readdirSync(root), ["bin"]);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

function attempt(platform, env, fail = false) {
  const calls = [];
  const warnings = [];
  maybeInstallService("/tmp/vox with spaces/vox", {
    platform,
    env,
    run: (...args) => {
      calls.push(args);
      if (fail) throw new Error("no user manager");
    },
    warn: (message) => warnings.push(message),
  });
  return { calls, warnings };
}

test("persistent global Linux installation enables autostart", () => {
  const { calls, warnings } = attempt("linux", { npm_config_global: "true" });
  assert.deepEqual(calls, [["/tmp/vox with spaces/vox", ["service", "install"], { stdio: "inherit" }]]);
  assert.deepEqual(warnings, []);
});

test("npx and local installs never create services", () => {
  assert.deepEqual(attempt("linux", {}).calls, []);
  assert.deepEqual(attempt("linux", { npm_config_global: "false" }).calls, []);
});

test("explicit opt-out and non-Linux installs skip systemd", () => {
  assert.deepEqual(attempt("linux", { npm_config_global: "true", VOX_NO_SERVICE: "1" }).calls, []);
  assert.deepEqual(attempt("darwin", { npm_config_global: "true" }).calls, []);
});

test("unavailable service manager does not break binary installation", () => {
  const { calls, warnings } = attempt("linux", { npm_config_global: "true" }, true);
  assert.equal(calls.length, 1);
  assert.equal(warnings.length, 1);
  assert.match(warnings[0], /vox-mcp service install/);
});
