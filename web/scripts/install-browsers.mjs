// Download the headless browser the PDF report renderer drives.
//
// PLAYWRIGHT_BROWSERS_PATH=0 puts it inside node_modules instead of the
// per-user cache. A Microsoft Store Python sees a virtualized AppData\Local,
// so a browser in that cache is invisible to the renderer the API spawns.
// Setting it here rather than in the npm script keeps it working on Windows,
// where npm runs scripts through cmd.exe and inline VAR=value is not a thing.

import { execFileSync } from "node:child_process";
import { createRequire } from "node:module";
import path from "node:path";

// cli.js is the package's bin entry but is not listed in "exports", so it is
// reached through the manifest rather than resolved directly.
const manifest = createRequire(import.meta.url).resolve("playwright/package.json");
const cli = path.join(path.dirname(manifest), "cli.js");

execFileSync(process.execPath, [cli, "install", "chromium"], {
  stdio: "inherit",
  env: { ...process.env, PLAYWRIGHT_BROWSERS_PATH: "0" },
});
