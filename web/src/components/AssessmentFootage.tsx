import { useEffect, useState } from "react";
import { Field } from "./WorkflowParts";
import { CameraPlacement, usePlacement } from "./CameraPlacement";
import { duration } from "../lib/format";
import { multiRoles, roleLabels, workflow, type AssessmentDraft, type AssessmentVideo, type CameraRole, type CaptureStatus } from "../lib/workflow";

export function VideoPreview({ draftId, video }: { draftId: string; video: AssessmentVideo }) {
  const [state, setState] = useState({ ready: false, detail: "Preparing video preview…" });
  useEffect(() => {
    const controller = new AbortController(); let timer = 0;
    const poll = async () => {
      try { const next = await workflow.videoStatus(draftId, video.id, controller.signal); setState(next);
        if (!next.ready && next.state === 'building') timer = window.setTimeout(poll, 1500);
      } catch (err) { if (!controller.signal.aborted) setState({ ready: false, detail: err instanceof Error ? err.message : 'Preview unavailable.' }); }
    };
    void poll(); return () => { controller.abort(); clearTimeout(timer); };
  }, [draftId, video.id]);
  return state.ready ? <video controls preload="metadata" src={workflow.videoUrl(draftId, video.id)} aria-label={`Preview ${video.name}`} />
    : <div className="setup-video-placeholder" role="status">{state.detail}</div>;
}

export function CaptureControls({ draft, busy, setBusy, onDraft, onError, onCapture, saveSetup }: {
  draft: AssessmentDraft; busy: boolean; setBusy: (value: boolean) => void;
  onDraft: (draft: AssessmentDraft) => void; onError: (message: string | null) => void;
  onCapture: (value: CaptureStatus) => void; saveSetup: () => Promise<AssessmentDraft>;
}) {
  const [devices, setDevices] = useState<number[] | null>(null);
  const [selected, setSelected] = useState<{ deviceId: number; role: CameraRole }[]>([]);
  const [capture, setCapture] = useState<CaptureStatus | null>(null);
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    const poll = () => workflow.capture(controller.signal).then(next => { setCapture(next); onCapture(next); setTick(Date.now()); }).catch(() => {});
    void poll(); const timer = window.setInterval(poll, 1000);
    return () => { controller.abort(); clearInterval(timer); };
  }, [draft.id, onCapture]);
  const owned = capture?.assessmentId === draft.id;
  const active = owned && Boolean(capture?.preview_active);
  const recording = owned && Boolean(capture?.recording);
  const placement = usePlacement(draft.id, active && !recording);
  const savedPlacement = draft.takes.find(take => take.id === draft.selectedTake)?.manifest?.placement_review;
  const multi = draft.setup.cameraMode === 'multi';
  async function perform(action: () => Promise<void>) {
    setBusy(true); onError(null);
    try { await action(); const next = await workflow.capture(); setCapture(next); onCapture(next); }
    catch (err) { onError(err instanceof Error ? err.message : 'Camera action failed.'); }
    finally { setBusy(false); }
  }
  return <section className="stack stack-4">
    {capture?.assessmentId && !owned ? <p role="alert" className="workflow-error">Cameras are being used by another assessment. Return to its controls using the banner above.</p> : null}
    {!owned ? <>
      <p className="small muted">{multi ? 'Connect two or three cameras and assign each a distinct view.' : 'Connect your camera and check that the participant and task are visible.'}</p>
      <div><button className="btn" type="button" disabled={busy || Boolean(capture?.assessmentId)} onClick={() => void perform(async () => {
        const result = await workflow.devices(); setDevices(result.devices);
        setSelected(!multi && result.devices.length === 1 ? [{ deviceId: result.devices[0], role: 'camera' }] : []);
      })}>{busy ? 'Checking devices…' : multi ? 'Find cameras' : 'Find camera'}</button></div>
      {devices?.length === 0 && <p role="status" className="workflow-note">No cameras were found. Connect a camera, close other apps using it, and try again.</p>}
      {!multi && Boolean(devices?.length) && <Field label="Camera"><select value={selected[0]?.deviceId ?? ''} disabled={busy} onChange={event => setSelected([{deviceId:Number(event.target.value),role:'camera'}])}>
        <option value="" disabled>Choose a camera</option>
        {devices?.map(device => <option key={device} value={device}>Camera {device}</option>)}
      </select></Field>}
      {multi && devices?.map(device => {
        const chosen = selected.find(item => item.deviceId === device);
        return <div className="capture-device" key={device}>
          <input type="checkbox" aria-label={`Use camera ${device}`} checked={Boolean(chosen)} disabled={busy || (!chosen && selected.length >= 3)} onChange={event => {
            if (event.target.checked) { const role = multiRoles.find(r => !selected.some(item => item.role === r)); if (role) setSelected([...selected,{deviceId:device,role}]); }
            else setSelected(selected.filter(item => item.deviceId !== device));
          }} /><span>Camera {device}</span>
          <select aria-label={`Camera ${device} view`} disabled={!chosen || busy} value={chosen?.role ?? 'front'} onChange={event => {
            const role = event.target.value as CameraRole;
            if (chosen) setSelected(selected.map(item => item.deviceId === device ? {...item,role} : item.role === role ? {...item,role:chosen.role} : item));
          }}>
            {multiRoles.map(key => <option value={key} key={key}>{roleLabels[key]}</option>)}
          </select>
        </div>;
      })}
      {Boolean(devices?.length) && <div><button className="btn btn--primary" type="button" disabled={busy || selected.length < (multi ? 2 : 1)} onClick={() => void perform(async () => { await saveSetup(); await workflow.preview(draft.id,selected); })}>{multi ? 'Open camera previews' : 'Open camera preview'}</button></div>}
    </> : <>
      <div className="workflow-heading"><h2>{recording ? 'Recording the task' : multi ? 'Check camera views' : 'Check camera'}</h2>
        <span className="badge badge--neutral">{recording ? `${duration(capture?.elapsed_sec ?? 0)} recorded` : 'Preview only'}</span></div>
      <div className="setup-video-grid">{Object.entries(capture?.cameras ?? {}).map(([device,camera]) => <div className="setup-video stack stack-2" key={device}>
        <strong>{multi ? `${roleLabels[camera.label as CameraRole] ?? camera.label} · Camera ${device}` : 'Camera'}</strong>
        {camera.error ? <p role="alert" className="workflow-error">{camera.error}</p> : camera.running ? <img src={workflow.captureImage(draft.id,device,tick)} alt={`${camera.label} camera preview`} /> : <p>Waiting for camera…</p>}
        <span className="small muted">{camera.actual_fps.toFixed(1)} frames per second</span>
        {active && !recording && placement.status?.cameras[device] && <CameraPlacement
          key={placement.status.preview} id={draft.id} device={device} camera={placement.status.cameras[device]}
          update={placement.update} />}
      </div>)}</div>
      {active && !recording && placement.error && <p role="status" className="small muted">{placement.error}</p>}
      {multi && active && !recording && Boolean(placement.status?.noClearView?.length) &&
        <p className="workflow-note">No sampled view has consistent landmark coverage for {placement.status?.noClearView?.join(", ").toLowerCase()}.
          Adjust one camera to cover these regions, then rehearse again.</p>}
      <p className="small muted">{capture?.synchronized_frame_sets ?? 0} {multi ? 'synchronized frame sets' : 'frames'} · {capture?.dropped_frame_sets ?? 0} dropped. Analysis runs after you save and review a take.</p>
      <div className="source-options">
        {recording ? <button type="button" className="btn btn--accent" disabled={busy} onClick={() => void perform(async () => onDraft(await workflow.stopCapture(draft.id)))}>Stop and save take</button>
          : <button type="button" className="btn btn--primary" disabled={busy || !active || Object.values(capture?.cameras ?? {}).some(c => !c.running || c.error)} onClick={() => void perform(async () => { await workflow.startCapture(draft.id); })}>Start recording</button>}
        <button type="button" className="btn" disabled={busy || recording} onClick={() => void perform(async () => { await workflow.closeCapture(draft.id); })}>Close preview</button>
      </div>
    </>}
    {draft.takes.length > 0 && <Field label="Take to analyze" hint="Every take is retained separately. Only the selected take is analyzed.">
      <select value={draft.selectedTake ?? ''} disabled={busy || owned} onChange={event => void perform(async () => onDraft(await workflow.selectTake(draft.id,event.target.value)))}>
        {draft.takes.map((take,index) => <option key={take.id} value={take.id}>Take {index+1} · {duration(take.durationSec)} · {new Date(take.createdAt).toLocaleTimeString()}</option>)}
      </select>
    </Field>}
    {savedPlacement && Object.values(savedPlacement.cameras).some(c => c.samples > 0) &&
      <details className="small"><summary>Placement review for this take</summary>
        {Object.entries(savedPlacement.cameras).map(([device, camera]) =>
          <div key={device} className="stack stack-1">
            <strong>{multi ? `Camera ${device}` : "Camera"} · {camera.decision?.value === "ready" ? "Confirmed ready"
              : camera.decision?.value === "adjust" ? "Adjustment requested" : camera.decision?.value === "unclear" ? "Marked unclear" : "No reviewer decision"}</strong>
            {camera.issues.map(issue => <p key={issue}>{issue}</p>)}
          </div>)}
      </details>}
  </section>;
}
