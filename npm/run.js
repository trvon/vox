#!/usr/bin/env node

"use strict";

const path = require("path");
const { execFileSync } = require("child_process");

const bin = path.join(__dirname, "bin", "vox");

try {
  execFileSync(bin, process.argv.slice(2), { stdio: "inherit" });
} catch (err) {
  if (err.status !== null) {
    process.exit(err.status);
  }
  throw err;
}
