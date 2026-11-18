import { cloneElement, isValidElement, useEffect, useId, useState, type ReactElement, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { api, useAsync } from "../lib/api";
import { reviewLink, studyDisplayName, workflow, type AssessmentDraft, type CaptureStatus } from "../lib/workflow";
import { ActionMenu, TrashDialog, type TrashSelection } from "./AssessmentActions";
import { ErrorState } from "./ui";
import "../styles/workflow.css";

export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  const id = useId();
  const control = isValidElement(children) ? cloneElement(children as ReactElement<{ 'aria-labelledby'?: string; 'aria-describedby'?: string }>, { 'aria-labelledby': id, 'aria-describedby': hint ? `${id}-hint` : undefined }) : children;
  return <label className="setup-field"><span id={id}>{label}</span>{control}{hint && <small id={`${id}-hint`}>{hint}</small>}</label>;
}

export function CaptureNotice() {
  const [capture, setCapture] = useState<CaptureStatus | null>(null);
  useEffect(() => {
    const controller = new AbortController();
    const poll = () => workflow.capture(controller.signal).then(setCapture).catch(() => {});
    void poll(); const timer = window.setInterval(poll, 2500);
    return () => { controller.abort(); clearInterval(timer); };
  }, []);
  if (!capture?.assessmentId) return null;
  return <div className="capture-notice" role="status">
    <span>{capture.recording ? "Recording is active" : "Camera preview is active"}</span>
    <Link to={`/assessments/${capture.assessmentId}`}>Return to camera controls</Link>
  </div>;
}

export const draftStatus = {draft: "Draft", interrupted: "Interrupted", failed: "Needs retry", processing: "Processing", ready: "Needs review"};
export function draftDestination(draft: AssessmentDraft) {
  return draft.status === "ready" && draft.analysis ? reviewLink(draft.analysis.sessionId, draft.setup.primaryMethod) : `/assessments/${draft.id}`;
}
export function draftTrash(draft: AssessmentDraft): TrashSelection {
  return {
    label: draft.setup.taskName || "Untitled assessment",
    target: draft.published ? {kind: "task", ...draft.published} : {kind: "draft", draftId: draft.id},
  };
}

interface RecentTask {
  id: string; label: string; participant: string; study: string; date: number; status: string;
  href: string; ready: boolean; active: boolean; removal: TrashSelection;
}
function timestamp(date?: string | null) {return date ? Date.parse(date) || 0 : 0;}

export function DraftList({ limit, onLoaded }: { limit?: number; onLoaded?: (drafts: AssessmentDraft[]) => void }) {
  const { data, error, reload, loading } = useAsync(async signal => {
    const [drafts,catalog] = await Promise.all([workflow.list(signal),api.participants("",signal)]);
    return {drafts: drafts.assessments, people: catalog.participants};
  }, []);
  const [removal,setRemoval] = useState<TrashSelection | null>(null);
  const [message,setMessage] = useState("");
  useEffect(() => { if (data) onLoaded?.(data.drafts); }, [data, onLoaded]);
  if (error) return <ErrorState message={error} onRetry={reload} />;
  const items = new Map<string, RecentTask>();
  for (const draft of data?.drafts ?? []) {
    const ready = draft.status === "ready" && Boolean(draft.analysis);
    const id = ready ? `session:${draft.analysis?.sessionId}` : draft.id;
    const removal = draftTrash(draft);
    items.set(id,{id,label:removal.label,participant:draft.setup.participantId,study:studyDisplayName(draft.setup.studyName, draft.setup.studyFolder ?? ""),
      date:timestamp(draft.updatedAt),status:draftStatus[draft.status],href:draftDestination(draft),ready,
      active:draft.status === "processing",removal});
  }
  for (const person of data?.people ?? []) {
    for (const task of person.trials) {
      if (task.status !== "analysed" || !task.sessionId) continue;
      const id = `session:${task.sessionId}`;
      const saved = items.get(id);
      items.set(id,{id,label:task.name,participant:person.participantId,study:studyDisplayName(person.studyName || saved?.study, person.study),
        date:Math.max(saved?.date ?? 0,timestamp(task.updatedAt ?? task.recordedAt)),
        status:task.reviewStatus === "reviewed" ? "Reviewed" : "Needs review",href:reviewLink(task.sessionId,task.primaryTool ?? "reba"),
        ready:true,active:false,removal:{label:task.name,target:{kind:"task",participantKey:person.key,taskFolder:task.folder}}});
    }
  }
  const tasks = Array.from(items.values()).sort((a,b) => b.date - a.date || a.label.localeCompare(b.label)).slice(0, limit);
  return <div className="draft-list">
    {message && <p role="status" className="small">{message} <Link to="/assessments?view=trash">Open Trash</Link></p>}
    {loading && !data && <p className="small muted" role="status">Loading recent assessments…</p>}
    {!loading && !tasks.length && <p className="small muted">Your saved tasks and completed assessments will appear here.</p>}
    {tasks.map(task => <div className="recent-task" key={task.id}>
      <Link className="draft-row" to={task.href}><div><strong>{task.label}</strong>
        <span className="small muted">{[task.participant && `Participant ${task.participant}`, task.study].filter(Boolean).join(" · ")}</span></div>
        <span className="badge badge--neutral">{task.status}</span><span className="small">{task.ready ? "Review" : "Continue"} →</span>
      </Link>
      <ActionMenu label={`More actions for ${task.label}`}><button className="btn btn--ghost" type="button" disabled={task.active}
        onClick={() => setRemoval(task.removal)}>Move to Trash</button></ActionMenu>
    </div>)}
    {removal && <TrashDialog selection={removal} onClose={() => setRemoval(null)} onMoved={() => {
      setMessage(`${removal.label} moved to Trash.`); setRemoval(null); reload();
    }} />}
  </div>;
}
