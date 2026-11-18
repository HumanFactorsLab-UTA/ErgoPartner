import { useEffect, useState } from "react";
import { Shell } from "../components/Shell";
import { StudiesFolderPicker } from "../components/StudiesFolderPicker";
import { Field } from "../components/WorkflowParts";
import { ErrorState } from "../components/ui";
import { useAsync } from "../lib/api";
import { Link } from "react-router-dom";
import { VisionModelSelect } from "../components/VisionModelSelect";
import { profileLabels, workflow, type Preferences } from "../lib/workflow";

export default function Settings() {
  const loaded = useAsync(signal => workflow.options(signal),[]);
  const [value,setValue] = useState<Preferences | null>(null);
  const [busy,setBusy] = useState(false); const [message,setMessage] = useState(''); const [error,setError] = useState('');
  useEffect(() => { if (loaded.data) setValue(loaded.data.preferences); },[loaded.data]);
  async function save() { if (!value) return; setBusy(true); setMessage(''); setError('');
    try { setValue(await workflow.preferences(value)); setMessage('Defaults saved for new assessments.'); }
    catch (err) { setError(err instanceof Error ? err.message : 'Settings could not be saved.'); } finally { setBusy(false); }
  }
  return <Shell crumbs={[{label:'Home',to:'/'},{label:'Settings'}]}><div className="assessment-setup stack stack-5">
    <header className="stack stack-2"><p className="eyebrow">Workspace preferences</p><h1>Settings</h1><p className="lede">Choose your study library and defaults for new assessments.</p></header>
    <StudiesFolderPicker disabled={busy} onChanged={loaded.reload} />
    {loaded.error && <ErrorState message={loaded.error} onRetry={loaded.reload} />}
    {value && <section className="card stack stack-5"><h2>New assessment defaults</h2><div className="setup-fields">
      <Field label="Open results in"><select value={value.primaryMethod} onChange={event => setValue({...value,primaryMethod:event.target.value as Preferences['primaryMethod']})}><option value="reba">REBA · Whole body</option><option value="rula">RULA · Upper limb</option></select></Field>
      <Field label="Analysis sampling"><select value={value.samplingFps} onChange={event => setValue({...value,samplingFps:Number(event.target.value)})}>{[5,10,15,30,60].map(fps => <option key={fps} value={fps}>{fps} fps</option>)}</select></Field>
      <Field label="Default assumption profile"><select value={value.assumptionProfile ?? ''} onChange={event => setValue({...value,assumptionProfile:(event.target.value || null) as Preferences['assumptionProfile']})}><option value="">No assumptions</option>{Object.entries(profileLabels).map(([key,label]) => <option key={key} value={key}>{label}</option>)}</select></Field>
      <Field label="Requested camera frame rate" hint="Actual capture rate depends on the camera and is shown during preview."><select value={value.cameraFps} onChange={event => setValue({...value,cameraFps:Number(event.target.value) as Preferences['cameraFps']})}>{[24,30,60].map(fps => <option key={fps} value={fps}>{fps} fps</option>)}</select></Field>
      <VisionModelSelect workstation disabled={busy} value={value.visionModel ?? null} onChange={visionModel => setValue({...value, visionModel})} />
    </div><div><button className="btn btn--primary" disabled={busy} onClick={() => void save()}>{busy ? 'Saving…' : 'Save defaults'}</button></div>
      {message && <p role="status" className="small">{message}</p>}{error && <p role="alert" className="workflow-error">{error}</p>}
    </section>}
    <details className="card"><summary>Research tools</summary>
      <p><Link to="/research">Research evaluation</Link></p>
    </details>
  </div></Shell>;
}
