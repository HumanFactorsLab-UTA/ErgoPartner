// Exercises the real setup screens with isolated draft and device responses.
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

process.env.PLAYWRIGHT_BROWSERS_PATH ??= "0";
const { chromium } = await import("playwright");
const root = fileURLToPath(new URL("../", import.meta.url));
const vite = await createServer({ root, server: { host: "127.0.0.1", strictPort: false }, logLevel: "error" });
await vite.listen();
const base = vite.resolvedUrls.local[0];
let browser;
try {
  browser = await chromium.launch();
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  const roles = ["front", "side", "iso"];
  let draft, lastCreation, capture, previewSources, placement;
  const resetCapture = () => ({assessmentId:null, preview_active:false, recording:false, cameras:{}});
  const clips = ["North.mp4","West.mp4"].map((name, i) => ({
    id: String(i), name, role: roles[i], fps: 30, durationSec: 3, width: 320, height: 240,
  }));
  const makeDraft = (source, cameraMode) => ({
    id:"browser-test", source, step:2, status:"draft", updatedAt:new Date().toISOString(),
    setup:{cameraMode, primaryMethod:"reba", samplingFps:10, taskName:"Move crate", studyName:"Test", participantId:"P1"},
    videos:[], availableVideos:source === "existing" ? clips : [], takes:[], selectedTake:null,
  });
  await page.route("**/api/**", async route => {
    const req = route.request(), url = new URL(req.url()), path = url.pathname;
    const reply = json => route.fulfill({json});
    if (path === "/api/capture/status") return reply(capture ?? resetCapture());
    if (path === "/api/capture/devices") return reply({devices:[2, 4]});
    if (path === "/api/participants") return reply({participants:[{key:"p", label:"P1", study:"Test", trials:[{folder:"t", name:"Move crate", cameras:2}]}]});
    if (path === "/api/assessment-options") return reply({studies:[], preferences:{}, profiles:{}});
    if (path === "/api/assessments" && req.method() === "POST") {
      lastCreation = req.postDataJSON();
      draft = makeDraft(lastCreation.source, lastCreation.cameraMode);
      return reply(draft);
    }
    if (path === "/api/assessments/browser-test") {
      if (req.method() === "POST") {
        const update = req.postDataJSON();
        draft.setup = update.setup;
        draft.step = update.step;
        if (update.videoIds) draft.videos = update.videoIds.map(id => draft.availableVideos.find(v => v.id === id));
        draft.videos = draft.videos.map((v,i) => ({...v, role:draft.setup.cameraMode === "single" ? "camera" : (update.roles?.[i] ?? roles[i])}));
      }
      return reply(draft);
    }
    if (path.endsWith("/capture/preview")) {
      previewSources = req.postDataJSON().cameras;
      capture = {...resetCapture(), assessmentId:draft.id, preview_active:true,
        cameras: Object.fromEntries(previewSources.map(({deviceId,role}) => [deviceId, {label: role, actual_fps:30, running:true, error:null}]))};
      placement = {preview:"rehearsal", cameras: Object.fromEntries(previewSources.map(({deviceId}) =>
        [deviceId, {id: "check-" + deviceId, state:"attention", samples:5, regions:[{name:"Hands",status:"partial"}],
                    issues:["Try another angle to see the hands."], visual:null, decision:null}]))};
      return reply(capture);
    }
    if (path.endsWith("/capture/placement")) return reply(placement);
    if (/\/capture\/placement\/\d+\/inspect$/.test(path)) {
      const camera = placement.cameras[path.split("/").at(-2)];
      assert.equal(req.postDataJSON().reviewId, camera.id);
      camera.visual = {state:"ready", checkId:"images", capturedAt:new Date().toISOString(), frames:[11,20,31],
        findings:[{label:"Grip contact area", visibility:"unclear", cue:"The hand is behind the tote.",
          suggestion:"Move sideways and repeat the grasp.", frames:[20]}]};
      return reply(placement);
    }
    if (/\/capture\/placement\/\d+\/decision$/.test(path)) {
      const camera = placement.cameras[path.split("/").at(-2)];
      assert.equal(req.postDataJSON().reviewId, camera.id);
      camera.decision = {value:req.postDataJSON().choice};
      return reply(placement);
    }
    if (path.endsWith("/videos") && req.method() === "POST") {
      const next = clips.find(v => v.name === url.searchParams.get("name"));
      draft.videos.push({...next, role:draft.setup.cameraMode === "single" ? "camera" : roles[draft.videos.length]});
      draft.availableVideos.push(next);
      return reply(draft);
    }
    if (path.endsWith("/status")) return reply({ready:false, state:"unavailable", detail:"Preview unavailable"});
    return reply({});
  });
  const nameNewTask = async () => {
    await page.getByRole("textbox", {name:"Study name", exact:true}).fill("Camera checks");
    await page.getByRole("textbox", {name:"Task name", exact:true}).fill("Move crate");
  };
  const mode = page.getByRole("combobox", {name:"Camera setup"});
  const uploads = page.locator("input[type=file]");
  const previewCards = page.locator(".setup-video-grid article");
  const sync = page.getByRole("checkbox", {name:/these views are synchronized/});
  const addClip = i => uploads.setInputFiles({name:clips[i].name, mimeType:"video/mp4", buffer:Buffer.from("mock video")});
  await page.goto(base + "new");
  await nameNewTask();
  assert.equal(await mode.inputValue(), "single");
  await page.getByRole("button", {name:"Create assessment"}).click();
  await uploads.waitFor();
  assert.equal(lastCreation.cameraMode, "single");
  assert.equal(await uploads.getAttribute("multiple"), null);
  await addClip(0);
  await previewCards.first().waitFor();
  assert.equal(await uploads.count(), 0);
  assert.equal(await sync.count(), 0);
  assert.equal(await page.getByRole("combobox",{name:/View for/}).count(), 0);
  await mode.selectOption("multi");
  await uploads.waitFor();
  assert.notEqual(await uploads.getAttribute("multiple"), null);
  await addClip(1);
  await sync.waitFor();
  assert.equal(await previewCards.count(), 2);
  assert.equal(await page.getByRole("combobox",{name:/View for/}).count(), 2);
  await mode.selectOption("single");
  await page.getByRole("combobox",{name:/Choose the video to keep/}).selectOption("1");
  await page.getByRole("combobox",{name:"Recording to analyze"}).waitFor();
  assert.equal(await previewCards.count(), 1);
  assert.match(await previewCards.textContent(), /West\.mp4/);
  assert.equal(await sync.count(), 0);
  await page.reload();
  await previewCards.first().waitFor();
  assert.equal(await mode.inputValue(), "single");
  assert.equal(await previewCards.count(), 1);
  await page.goto(base + "new?source=existing&participant=p&trial=t");
  await page.getByRole("button",{name:"Create assessment"}).click();
  await page.getByRole("combobox",{name:"Recording to analyze"}).selectOption("0");
  await previewCards.first().waitFor();
  assert.equal(lastCreation.participantKey, "p");
  assert.equal(lastCreation.taskFolder, "t");
  assert.equal(await previewCards.count(), 1);
  for (const multi of [false, true]) {
    capture = resetCapture();
    await page.goto(base + "new?source=live");
    await nameNewTask();
    if (multi) await mode.selectOption("multi");
    await page.getByRole("button",{name:"Create assessment"}).click();
    await page.getByRole("button",{name:/Find camera/}).click();
    if (multi) {
      await page.getByRole("checkbox",{name:"Use camera 2"}).check();
      assert.equal(await page.getByRole("button",{name:"Open camera previews"}).isDisabled(), true);
      await page.getByRole("checkbox",{name:"Use camera 4"}).check();
    } else {
      await page.getByRole("combobox",{name:"Camera", exact:true}).selectOption("4");
      assert.equal(await page.getByRole("checkbox", {name:/Use camera/}).count(), 0);
    }
    await page.getByRole("button",{name:/Open camera preview/}).click();
    await page.getByRole("button",{name:"Start recording"}).waitFor();
    assert.deepEqual(previewSources, multi ? [{deviceId:2,role:"front"},{deviceId:4,role:"side"}] : [{deviceId:4,role:"camera"}]);
    const checks = page.getByRole("region",{name:"Camera placement check"});
    await checks.first().waitFor();
    assert.equal(await checks.count(), multi ? 2 : 1);
    await checks.last().getByRole("button",{name:/visibility with AI/}).click();
    await checks.last().getByText("Grip contact area · unclear").waitFor();
    await checks.last().getByRole("button",{name:"Needs adjustment"}).click();
    await checks.last().getByRole("button",{name:"Needs adjustment",pressed:true}).waitFor();
    if (multi) assert.equal(await checks.first().getByRole("button",{name:"Needs adjustment",pressed:true}).count(),0);
  }
  assert.deepEqual(errors, []);
  console.log("Passed uploads, resume, existing footage and per-camera placement inspections and decisions in single/multi previews.");
} finally {
  await browser?.close();
  await vite.close();
}
