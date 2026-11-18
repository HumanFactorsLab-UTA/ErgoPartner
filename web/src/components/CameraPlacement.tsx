import { useEffect, useRef, useState } from "react";
import { workflow, type PlacementCamera, type PlacementChoice, type PlacementStatus } from "../lib/workflow";

const labels = {
  warming: "Checking the rehearsal…", clear: "Body regions in view",
  attention: "Check camera placement", unavailable: "Coverage unavailable",
};
const decisions: [PlacementChoice, string][] = [
  ["ready", "View is ready"], ["adjust", "Needs adjustment"], ["unclear", "Cannot tell"],
];

export function usePlacement(id: string, active: boolean) {
  const [status, setStatus] = useState<PlacementStatus | null>(null);
  const [error, setError] = useState("");
  const order = useRef(0);
  useEffect(() => {
    setStatus(null); setError("");
    if (!active) return;
    const controller = new AbortController();
    let timer = 0;
    async function poll() {
      const version = ++order.current;
      try {
        const result = await workflow.placement(id, controller.signal);
        if (!controller.signal.aborted && version === order.current) { setStatus(result); setError(""); }
      } catch (e) {
        if (!controller.signal.aborted && version === order.current) {
          setStatus(null); setError(e instanceof Error ? e.message : "Placement checks unavailable.");
        }
      } finally {
        if (!controller.signal.aborted) timer = window.setTimeout(poll, 1300);
      }
    }
    void poll();
    return () => { controller.abort(); clearTimeout(timer); ++order.current; };
  }, [active, id]);
  return { status, error, update: (value: PlacementStatus) => { ++order.current; setStatus(value); } };
}

export function CameraPlacement({ id, device, camera, update }: {
  id: string; device: string; camera: PlacementCamera; update: (status: PlacementStatus) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const visual = camera.visual;
  const checking = visual?.state === "checking";
  async function act(choice?: PlacementChoice) {
    setBusy(true); setError("");
    try {
      update(choice ? await workflow.decidePlacement(id, device, camera.id, choice)
                    : await workflow.inspectPlacement(id, device, camera.id));
    } catch (e) { setError(e instanceof Error ? e.message : "The preview changed. Try again."); }
    finally { setBusy(false); }
  }
  return <section className="camera-placement stack stack-2" data-state={camera.state} aria-label="Camera placement check">
    <strong role="status">{labels[camera.state]}</strong>
    {camera.state === "warming" && <p className="small muted">Rehearse the task while the camera samples the working area.</p>}
    {camera.issues.length > 0 && <ul className="small">{camera.issues.map(issue => <li key={issue}>{issue}</li>)}</ul>}
    {camera.regions.length > 0 && <details className="small"><summary>Coverage across {camera.samples} frames</summary>
      <ul>{camera.regions.map(item => <li key={item.name}>{item.name} · {item.status}</li>)}</ul></details>}
    <div><button type="button" className="btn" disabled={busy || checking || camera.samples < 3}
      onClick={() => void act()}>{checking ? "Inspecting sampled frames…" : visual ? "Recheck with AI" : "Check framing, grip and support visibility with AI"}</button></div>
    {visual?.detail && <p role="status" className="small muted">{visual.detail}</p>}
    {visual?.state === "ready" && <>
      <ul className="small">{visual.findings?.map(f => <li key={f.label}>
        <strong>{f.label} · {f.visibility.replaceAll("_", " ")}</strong><br />{f.cue} {f.suggestion}
        <small> · {f.frames.map(n => `Frame ${n}`).join(", ")}</small>
      </li>)}</ul>
      <details><summary className="small">Inspect the {visual.frames.length} sampled images</summary>
        {visual.frames.map(frame => <figure key={frame}><img loading="lazy" src={workflow.placementImage(id, device, visual.checkId, frame)}
          alt={`Live preview sample, camera ${device}, frame ${frame}`} /><figcaption className="small">Frame {frame}</figcaption></figure>)}
      </details>
    </>}
    {camera.state !== "warming" && <div role="group" aria-label="Your placement decision" className="source-options">
      {decisions.map(([value, label]) => <button type="button" key={value}
        className={camera.decision?.value === value ? "btn btn--primary" : "btn"}
        aria-pressed={camera.decision?.value === value} disabled={busy || checking}
        onClick={() => void act(value)}>{label}</button>)}
    </div>}
    {error && <p className="workflow-error small" role="alert">{error}</p>}
  </section>;
}
