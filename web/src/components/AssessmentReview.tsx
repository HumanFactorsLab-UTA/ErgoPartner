import { useEffect, useState } from "react";
import { useBlocker } from "react-router-dom";
import { useAsync } from "../lib/api";
import { workflow, type AssessmentReview as Review } from "../lib/workflow";
import { Field } from "./WorkflowParts";
import { ReportButton } from "./ReportButton";
import { ErrorState } from "./ui";
import { SavedInterpretations } from "./SavedInterpretations";
import type { AssistantCitation } from "../lib/types";

export function AssessmentReview({ sessionId, onCitation }: { sessionId: string; onCitation: (citation: AssistantCitation) => void }) {
  const loaded = useAsync(signal => workflow.review(sessionId,signal),[sessionId]);
  const [value,setValue] = useState<Review | null>(null); const [dirty,setDirty] = useState(false);
  const [busy,setBusy] = useState(false); const [error,setError] = useState(''); const [saved,setSaved] = useState('');
  const [selectionPending,setSelectionPending] = useState(false);
  const blocker = useBlocker(dirty || busy);
  useEffect(() => { if (loaded.data) { setValue(loaded.data); setDirty(false); } },[loaded.data]);
  useEffect(() => { const handler = (event: BeforeUnloadEvent) => { if (dirty) { event.preventDefault(); event.returnValue = ''; } }; window.addEventListener('beforeunload',handler); return () => window.removeEventListener('beforeunload',handler); },[dirty]);
  function edit(key: 'reviewer' | 'findings' | 'actions', next: string) { if (value) { setValue({...value,[key]:next,reviewed:false}); setDirty(true); setSaved(''); } }
  async function save() {
    if (!value) return false; setBusy(true); setError('');
    try { const next = await workflow.saveReview(sessionId,value); setValue(next); setDirty(false); setSaved(next.reviewed ? 'Assessment marked reviewed.' : 'Review notes saved.'); return true; }
    catch (err) { setError(err instanceof Error ? err.message : 'Review could not be saved.'); return false; }
    finally { setBusy(false); }
  }
  if (loaded.error) return <ErrorState message={loaded.error} onRetry={loaded.reload} />;
  if (!value) return <p>Loading review notes…</p>;
  return <section className="review-report stack stack-5">
    <div className="workflow-heading"><div><p className="eyebrow">Ergonomist review</p><h2>Findings and recommended actions</h2></div><span className="badge badge--neutral">{value.reviewed && !dirty ? 'Reviewed' : 'Needs review'}</span></div>
    <SavedInterpretations sessionId={sessionId} onCitation={onCitation} onSelectionPending={setSelectionPending} />
    {value.stale && <p className="workflow-note">The analysis has changed since these notes were saved. Review the new results before confirming the assessment again.</p>}
    {blocker.state === 'blocked' && <div className="workflow-note" role="alert"><p>Save your review notes before leaving?</p><div className="source-options">
      <button className="btn btn--primary" disabled={busy} onClick={() => void save().then(ok => { if (ok) blocker.proceed?.(); })}>Save and leave</button>
      <button className="btn" onClick={() => blocker.reset?.()}>Stay here</button>
      <button className="btn btn--ghost" disabled={busy} onClick={() => { setDirty(false); blocker.proceed?.(); }}>Discard unsaved changes</button>
    </div></div>}
    <Field label="Reviewer"><input value={value.reviewer} maxLength={160} disabled={busy} onChange={event => edit('reviewer',event.target.value)} /></Field>
    <Field label="Findings" hint="Describe the postures, tasks, and evidence supporting your conclusions."><textarea value={value.findings} maxLength={12000} disabled={busy} onChange={event => edit('findings',event.target.value)} /></Field>
    <Field label="Recommended actions" hint="Include practical changes, follow-up assessment, and any unresolved questions."><textarea value={value.actions} maxLength={12000} disabled={busy} onChange={event => edit('actions',event.target.value)} /></Field>
    <label className="setup-check"><input type="checkbox" checked={value.reviewed} disabled={busy || !value.reviewer.trim() || !value.findings.trim()} onChange={event => { setValue({...value,reviewed:event.target.checked}); setDirty(true); setSaved(''); }} /><span>I have reviewed the evidence, inputs, and limitations and completed this assessment.</span></label>
    <div className="source-options"><button className="btn btn--primary" disabled={busy} onClick={() => void save()}>{busy ? 'Saving…' : 'Save review'}</button>{!dirty && <ReportButton sessionId={sessionId} disabled={selectionPending} />}</div>
    {dirty && <p className="small muted">Save your notes to include them in the PDF report.</p>}
    {saved && <p role="status">{saved}</p>}{error && <p className="workflow-error" role="alert">{error}</p>}
    {value.reviewedAt && <p className="small muted">Reviewed {new Date(value.reviewedAt).toLocaleString()}</p>}
  </section>;
}
