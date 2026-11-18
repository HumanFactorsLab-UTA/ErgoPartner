import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { once } from "node:events";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import { join } from "node:path";
import { createServer } from "vite";

process.env.PLAYWRIGHT_BROWSERS_PATH ??= "0";
const { chromium } = await import("playwright");
const web = fileURLToPath(new URL("../", import.meta.url));
const repo = fileURLToPath(new URL("../../", import.meta.url));
const localPython = join(
  repo,
  ".venv313",
  process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
);
const python =
  process.env.ERGOPARTNER_TEST_PYTHON || (existsSync(localPython) ? localPython : "python");
const script = `
import socket, sys, threading, tempfile
from pathlib import Path
import uvicorn
from api import assessments, main
from dashboard import data_loader
import cv2, numpy as np
with tempfile.TemporaryDirectory(prefix="ergopartner-browser-") as tmp:
    root = Path(tmp) / "studies"
    root.mkdir()
    assessments.WORKFLOW_ROOT = Path(tmp) / "workflow"
    main.STUDIES_ROOT = str(root)
    data_loader.RESULTS_DIR_STR = str(Path(tmp) / "results")
    imported = root / "study_20260212_125912"
    assessments.atomic_json(imported / "study_config.json", {"study_name":imported.name,"tasks":[{"name":"Completed reach","folder_name":"reach"}]})
    assessments.atomic_json(imported / "participant_A" / "participant_info.json", {"participant_id":"A"})
    raw = imported / "participant_A" / "reach" / "raw"
    raw.mkdir(parents=True)
    video = raw / "cam_0.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    for _ in range(10):
        writer.write(np.zeros((48,64,3), dtype=np.uint8))
    writer.release()
    result = Path(data_loader.RESULTS_DIR_STR) / "10fps" / imported.name / "A" / "reach"
    assessments.atomic_json(result / "run_config.json", {"video":{"target_fps":10}})
    assessments.atomic_json(result / "video_summary.json", {"duration_sec":1.0, "reba_summary":{"maximum_score":3}, "rula_summary":{"maximum_score":2}})
    (result / "frame_results.csv").write_text("frame_number,timestamp_sec")
    before = video.read_bytes()
    assessments.create(assessments.CreateDraft(), str(root))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        server = uvicorn.Server(uvicorn.Config(main.app, log_level="error"))
        thread = threading.Thread(target=server.run, kwargs={"sockets":[sock]})
        thread.start()
        print(f"PORT {sock.getsockname()[1]}", flush=True)
        sys.stdin.readline()
        server.should_exit = True
        thread.join()
    assert video.read_bytes() == before
`;
const backend = spawn(python, ["-B", "-u", "-c", script], {
  cwd: repo,
  windowsHide: true,
  stdio: ["pipe", "pipe", "inherit"],
});
const done = once(backend, "exit");
let vite, browser;
try {
  const output = createInterface({ input: backend.stdout });
  let port;
  for await (const line of output) {
    if (line.startsWith("PORT ")) {
      port = Number(line.slice(5));
      break;
    }
  }
  assert.ok(port, "isolated assessment API started");
  vite = await createServer({
    root: web,
    server: {
      host: "127.0.0.1",
      port: 0,
      strictPort: false,
      proxy: { "/api": { target: `http://127.0.0.1:${port}`, changeOrigin: false } },
    },
    logLevel: "error",
  });
  await vite.listen();
  const base = vite.resolvedUrls.local[0];
  browser = await chromium.launch();
  const page = await browser.newPage({ locale: "en-US" });
  const errors = [];
  page.setDefaultTimeout(12000);
  page.on("pageerror", (error) => {
    errors.push(error.message);
    console.error(error.message);
  });
  page.on("response", async (response) => {
    if (response.url().includes("/api/") && response.status() >= 400)
      console.error(response.status(), response.url(), await response.text());
  });
  const move = async (name, participant = false) => {
    const trigger = page.getByRole("button", { name: `More actions for ${name}`, exact: true });
    await trigger.focus();
    await page.keyboard.press("ArrowDown");
    await page
      .getByRole("button", {
        name: participant ? "Move participant to Trash" : "Move to Trash",
        exact: true,
      })
      .click();
    const dialog = page.getByRole("dialog");
    await dialog.getByRole("button", { name: "Move to Trash", exact: true }).click();
    await dialog.waitFor({ state: "hidden" });
  };
  await page.goto(base);
  await move("Untitled assessment");
  await page.getByRole("link", { name: "Open Trash", exact: true }).click();
  await page.getByRole("button", { name: "Restore Untitled assessment" }).click();
  await page.getByRole("heading", { name: "Trash is empty" }).waitFor();
  await page.getByRole("link", { name: "Back to assessments" }).click();
  await page.goto(base);
  const completed = page.getByRole("link", { name: /Completed reach.*Review/ });
  await completed.waitFor();
  const expectedStudy = await page.evaluate(() => "Study on " + new Intl.DateTimeFormat("en-US", {
    dateStyle: "medium", timeStyle: "short"
  }).format(new Date(2026, 1, 12, 12, 59)));
  await completed.getByText(expectedStudy, { exact: false }).waitFor();
  assert.match(await completed.getAttribute("href"), /^\/s\/.+\/reba$/);
  await move("Completed reach");
  await page.getByRole("link", { name: "Open Trash", exact: true }).click();
  await page.getByRole("button", { name: "Restore Completed reach" }).click();
  await page.getByRole("heading", { name: "Trash is empty" }).waitFor();
  await page.getByRole("link", { name: "Back to assessments" }).click();
  await page.getByRole("heading", { name: expectedStudy, exact: true }).waitFor();
  const originalGroup = page.getByRole("region", { name: "Participant A", exact: true });
  await originalGroup.getByRole("link", { name: "Review results" }).waitFor();
  await originalGroup.getByRole("link", { name: "Add task", exact: true }).click();
  await page
    .getByRole("option", { name: "Participant A", exact: true })
    .waitFor({ state: "attached" });
  assert.equal(
    await page.getByLabel("Participant", { exact: true }).locator("option:checked").textContent(),
    "Participant A",
  );
  await page.goto(base + "new?source=existing&participant=study_20260212_125912%3A%3AA&trial=reach");
  await page.getByRole("button", { name: "Create assessment" }).click();
  await page.getByRole("heading", { name: "Completed reach", exact: true }).waitFor();
  await page.goto(base + "assessments");
  const savedSetup = originalGroup
    .locator("article")
    .filter({ hasText: "Saved setup for an existing recording" });
  await savedSetup.getByRole("button", { name: "More actions for Completed reach" }).click();
  await savedSetup.getByRole("button", { name: "Move to Trash", exact: true }).click();
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Move to Trash", exact: true })
    .click();
  await savedSetup.waitFor({ state: "hidden" });
  await originalGroup.getByRole("link", { name: "Review results", exact: true }).waitFor();
  await page.goto(base + "new");
  await page.getByLabel("Study", { exact: true }).selectOption({ label: "Create a new study" });
  await page.getByLabel("Study name", { exact: true }).fill("Packing study");
  await page.getByLabel("Task name", { exact: true }).fill("Lift crate");
  await page.getByRole("button", { name: "Create assessment" }).click();
  await page
    .getByRole("heading", { name: "Lift crate", exact: true })
    .waitFor()
    .catch(async (error) => {
      console.error(page.url(), await page.locator("main").innerText());
      throw error;
    });
  await page.getByRole("link", { name: "Add another task for this participant" }).click();
  await page
    .getByRole("option", { name: "Participant 001", exact: true })
    .waitFor({ state: "attached" });
  assert.equal(
    await page.getByLabel("Participant", { exact: true }).locator("option:checked").textContent(),
    "Participant 001",
  );
  await page.getByLabel("Task name", { exact: true }).fill("Place crate");
  await page.getByRole("button", { name: "Create assessment" }).click();
  await page.getByRole("heading", { name: "Place crate", exact: true }).waitFor();
  await page.goto(base + "assessments");
  const person = page.getByRole("region", { name: "Participant 001", exact: true });
  await person.getByText("Lift crate", { exact: true }).waitFor();
  await person.getByText("Place crate", { exact: true }).waitFor();
  await page
    .getByRole("region", { name: "Packing study", exact: true })
    .getByRole("link", { name: "Add participant", exact: true })
    .click();
  await page.getByLabel("Task name", { exact: true }).fill("Carry tote");
  await page.getByRole("button", { name: "Create assessment" }).click();
  await page.getByRole("heading", { name: "Carry tote", exact: true }).waitFor();
  await page.goto(base + "assessments");
  await page.setViewportSize({ width: 390, height: 844 });
  await move("Lift crate");
  await person.getByText("Lift crate", { exact: true }).waitFor({ state: "hidden" });
  await person.getByText("Place crate", { exact: true }).waitFor();
  await move("Participant 001", true);
  await person.waitFor({ state: "hidden" });
  await page.getByRole("region", { name: "Participant 002", exact: true }).waitFor();
  await page.getByRole("link", { name: "Open Trash", exact: true }).click();
  await page.getByRole("button", { name: "Restore Participant 001", exact: true }).click();
  await page
    .getByRole("button", { name: "Restore Participant 001", exact: true })
    .waitFor({ state: "hidden" });
  await page.getByRole("button", { name: "Restore Lift crate", exact: true }).click();
  await page
    .getByRole("button", { name: "Restore Lift crate", exact: true })
    .waitFor({ state: "hidden" });
  await page.getByRole("link", { name: "Back to assessments" }).click();
  await person.getByText("Lift crate", { exact: true }).waitFor();
  await person.getByText("Place crate", { exact: true }).waitFor();
  assert.equal(
    await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
    true,
    "mobile library fits viewport",
  );
  await page.goto(base + "new?study=study_20260212_125912&participant=study_20260212_125912%3A%3Agone");
  await page.getByText("Choose an available participant in this study", { exact: false }).waitFor();
  await page.getByLabel("Task name", { exact: true }).fill("Extra task");
  assert.ok(await page.getByRole("button", { name: "Create assessment" }).isDisabled());
  assert.deepEqual(errors, []);
  console.log("Participant, task, keyboard actions and mobile Trash/restore checks passed.");
} finally {
  await browser?.close();
  await vite?.close();
  backend.stdin.end("\n");
  await done;
}
