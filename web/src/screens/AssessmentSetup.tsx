import { useCallback, useEffect, useState } from "react";
import { Link, useBlocker, useParams } from "react-router-dom";
import { Shell } from "../components/Shell";
import { Field } from "../components/WorkflowParts";
import { VisionModelSelect } from "../components/VisionModelSelect";
import { CaptureControls, VideoPreview } from "../components/AssessmentFootage";
import { ErrorState, Skeleton } from "../components/ui";
import { useAsync } from "../lib/api";
import { duration } from "../lib/format";
import { multiRoles, studyDisplayName, profileLabels, roleLabels, workflow, type AssessmentDraft, type AssessmentSetup as Setup, type CameraMode, type CameraRole, type CaptureStatus } from "../lib/workflow";

const STEPS = ['Assessment details','Footage','Task conditions','Analyze'];
const booleanFields = [ ['shockForce','Shock or rapid force build-up'], ['leftArmSupport','Left arm supported'], ['rightArmSupport','Right arm supported'], ['leftWristEndRange','Left wrist twist at end range'], ['rightWristEndRange','Right wrist twist at end range'] ] as const;

export default function AssessmentSetup() {
  const { assessmentId = '' } = useParams();
  const loaded = useAsync(signal => workflow.draft(assessmentId,signal),[assessmentId]);
  const options = useAsync(signal => workflow.options(signal),[]);
  const [draft,setDraft] = useState<AssessmentDraft | null>(null);
  const [dirty,setDirty] = useState(false); const [busy,setBusy] = useState(false);
  const [error,setError] = useState<string | null>(null); const [notice,setNotice] = useState('');
  const [capture,setCapture] = useState<CaptureStatus | null>(null);
  const [chooseSingle, setChooseSingle] = useState(false);
  const blocker = useBlocker(dirty || busy);
  useEffect(() => { if (loaded.data) { setDraft(loaded.data); setDirty(false); } },[loaded.data]);
  useEffect(() => {
    const handler = (event: BeforeUnloadEvent) => { if (dirty || busy || capture?.assessmentId === assessmentId) { event.preventDefault(); event.returnValue = ''; } };
    window.addEventListener('beforeunload',handler); return () => window.removeEventListener('beforeunload',handler);
  },[dirty,busy,capture,assessmentId]);
  useEffect(() => {
    if (draft?.status !== 'processing') return;
    const controller = new AbortController();
    const timer = window.setInterval(() => { workflow.draft(assessmentId,controller.signal).then(setDraft).catch(err => { if (!controller.signal.aborted) setError(err.message); }); },1500);
    return () => { controller.abort(); clearInterval(timer); };
  },[draft?.status,assessmentId]);
  const onCapture = useCallback((next: CaptureStatus) => setCapture(next),[]);
  const accept = useCallback((next: AssessmentDraft) => { setDraft(next); setDirty(false); setNotice('Saved'); },[]);
  const cameraActive = capture?.assessmentId === assessmentId;
  const locked = Boolean(draft?.analysis || draft?.published);
  async function save(step?: number) {
    if (!draft) throw new Error('Assessment has not loaded.');
    const next = await workflow.save(draft,step ?? draft.step); accept(next); return next;
  }
  async function perform(action: () => Promise<void>) {
    setBusy(true); setError(null); setNotice('');
    try { await action(); } catch (err) { setError(err instanceof Error ? err.message : 'The assessment could not be updated.'); }
    finally { setBusy(false); }
  }
  function change<K extends keyof Setup>(key: K, value: Setup[K]) {
    setDraft(current => current ? {...current,setup:{...current.setup,[key]:value,...(key === 'conditionsReviewed' || key === 'footageReviewed' || key === 'synchronizationConfirmed' ? {} : {conditionsReviewed:false})}} : current);
    setDirty(true); setNotice('Unsaved changes');
  }
  async function go(step: number) {
    if (chooseSingle) { setError('Choose the video for this assessment or switch back to Multiple cameras.'); return; }
    if (cameraActive) { setError('Close the camera previews before continuing. Your saved takes will remain available.'); return; }
    if (locked) { setDraft(current => current ? {...current,step} : current); return; }
    await perform(async () => { await save(step); });
  }
  if (loaded.error) return <Shell crumbs={[{label:'Assessments',to:'/assessments'},{label:'Setup'}]}><ErrorState message={loaded.error} onRetry={loaded.reload} /></Shell>;
  if (!draft) return <Shell><Skeleton height={400} /></Shell>;
  const setup = draft.setup; const step = draft.step;
  const multi = setup.cameraMode === 'multi';
  const sourceLimit = multi ? 3 : 1;
  const hasRequiredSources = multi ? draft.videos.length >= 2 && draft.videos.length <= 3 : draft.videos.length === 1;
  const missingDetails = !setup.studyName.trim() || !setup.taskName.trim() || !setup.participantId.trim();
  const missingFootage = !hasRequiredSources || !setup.footageReviewed || (multi && !setup.synchronizationConfirmed);
  const canAnalyze = !missingDetails && !missingFootage && setup.conditionsReviewed;
  const available = draft.availableVideos ?? draft.videos;
  function selectFootage(mode: CameraMode, videoIds?: string[]) {
    if (!draft) return;
    void perform(async () => {
      const current = dirty ? await save() : draft;
      accept(await workflow.selectFootage(current, mode, videoIds));
      setChooseSingle(false);
    });
  }
  const ready = draft.status === 'ready';
  const fixedParticipant = draft.grouped || draft.source === 'existing';
  const parentKey = draft.published?.participantKey ?? draft.existing?.participantKey ??
    (setup.studyFolder && setup.participantId ? `${setup.studyFolder}::${setup.participantId}` : null);
  const addTask = parentKey ? `/new?${new URLSearchParams({participant:parentKey,study:setup.studyFolder ?? parentKey.split("::")[0]})}` : null;
  return <Shell crumbs={[{label:'Home',to:'/'},{label:'Assessments',to:'/assessments'},{label:setup.taskName || 'New assessment'}]}>
    <div className="assessment-setup stack stack-5">
      <header className="workflow-heading"><div className="stack stack-2"><p className="eyebrow">{draft.source === 'live' ? 'Live recording' : 'Recorded video'}</p>
        <h1>{setup.taskName || 'Set up your assessment'}</h1><p className="small muted">{setup.participantId ? `Participant ${setup.participantId} · ` : ''}{studyDisplayName(setup.studyName,setup.studyFolder ?? '') || 'Choose a study and task'}</p></div>
        <span className="badge badge--neutral">{ready ? 'Ready to review' : locked ? draft.status : 'Draft'}</span></header>
      <nav className="setup-steps" aria-label="Assessment setup steps">{STEPS.map((label,index) => <button type="button" key={label} disabled={busy || cameraActive} aria-current={step === index+1 ? 'step' : undefined} onClick={() => void go(index+1)}><span>0{index+1}</span>{label}</button>)}</nav>
      {blocker.state === 'blocked' && <div className="workflow-note" role="alert"><strong>{busy ? 'An operation is still running.' : 'Save your changes before leaving?'}</strong><div className="source-options">
        <button className="btn btn--primary" disabled={busy} onClick={() => void perform(async () => { await save(); blocker.proceed?.(); })}>Save and leave</button>
        <button className="btn" onClick={() => blocker.reset?.()}>Stay here</button>
        <button className="btn btn--ghost" disabled={busy} onClick={() => { setDirty(false); blocker.proceed?.(); }}>Discard unsaved changes</button>
      </div></div>}
      {error && <p className="workflow-error" role="alert">{error}</p>}
      {locked && step !== 4 && <p className="workflow-note">This is the saved setup used for this analysis.</p>}
      {step === 1 && <section className="card stack stack-5">
        <h2>Assessment details</h2>
        {fixedParticipant && <p>{studyDisplayName(setup.studyName,setup.studyFolder ?? '')} · Participant {setup.participantId}</p>}
        <div className="setup-fields">
          {!fixedParticipant && <>
          <Field label="Study"><select disabled={busy || locked || cameraActive || draft.source === 'existing'} value={setup.studyFolder ?? ''} onChange={event => {
            const selected = options.data?.studies.find(s => s.folder === event.target.value); change('studyFolder',selected?.folder ?? null); change('studyName',selected?.name ?? '');
          }}><option value="">Create a new study</option>{options.data?.studies.map(s => <option key={s.folder} value={s.folder}>{studyDisplayName(s.name,s.folder)}</option>)}</select></Field>
          <Field label="Study name"><input value={setup.studyName} maxLength={160} disabled={busy || locked || cameraActive || Boolean(setup.studyFolder)} onChange={event => change('studyName',event.target.value)} placeholder="e.g. Packing workstation review" /></Field>
          <Field label="Participant ID" hint="Use an ID rather than the participant's name."><input value={setup.participantId} maxLength={64} disabled={busy || locked || cameraActive || draft.source === 'existing'} onChange={event => change('participantId',event.target.value)} placeholder="e.g. P001" /></Field></>}
          <Field label="Task name"><input value={setup.taskName} maxLength={160} disabled={busy || locked || cameraActive || draft.source === 'existing'} onChange={event => change('taskName',event.target.value)} placeholder="e.g. Lifting boxes to shelf" /></Field>
          <Field label="Participant height (cm)" hint="Optional. Used for estimated lifting geometry."><input type="number" min={50} max={260} step="0.1" disabled={busy || locked || cameraActive} value={setup.heightCm ?? ''} onChange={event => change('heightCm',event.target.value === '' ? null : Number(event.target.value))} /></Field>
        </div>
      </section>}
      {step === 2 && <section className="card stack stack-5"><h2>{draft.source === 'live' ? 'Record and review a take' : 'Select and preview footage'}</h2>
        <Field label="Camera setup"><select value={chooseSingle ? 'single' : setup.cameraMode} disabled={busy || locked || cameraActive} onChange={event => {
          const mode = event.target.value as CameraMode;
          if (mode === 'single' && draft.videos.length > 1) setChooseSingle(true);
          else if (mode === setup.cameraMode) setChooseSingle(false);
          else selectFootage(mode);
        }}><option value="single">Single camera</option><option value="multi">Multiple cameras</option></select></Field>
        {chooseSingle && <Field label="Choose the video to keep in this assessment">
          <select value="" disabled={busy} onChange={event => selectFootage('single', [event.target.value])}>
            <option value="" disabled>Choose a video</option>
            {draft.videos.map(video => <option key={video.id} value={video.id}>{video.name}</option>)}
          </select>
        </Field>}
        {!chooseSingle && <>
        {!locked && !cameraActive && (draft.source === 'existing' || available.some(v => !draft.videos.some(s => s.id === v.id))) && (multi ?
          <fieldset><legend>Recordings to analyze</legend>{available.map(video => {
            const ids = draft.videos.map(v => v.id);
            const selected = ids.includes(video.id);
            return <label className="setup-check" key={video.id}><input type="checkbox" checked={selected} disabled={busy || (!selected && ids.length >= 3)} onChange={() => selectFootage('multi', selected ? ids.filter(id => id !== video.id) : [...ids, video.id])} />{video.name}</label>;
          })}</fieldset> :
          <Field label="Recording to analyze"><select value={draft.videos[0]?.id ?? ''} disabled={busy} onChange={event => selectFootage('single', [event.target.value])}>
            <option disabled value="">Choose one video</option>
            {available.map(video => <option key={video.id} value={video.id}>{video.name}</option>)}
          </select></Field>
        )}
        {draft.source === 'live' && !locked && <CaptureControls key={setup.cameraMode} draft={draft} busy={busy} setBusy={setBusy} onDraft={accept} onError={setError} onCapture={onCapture} saveSetup={() => save()} />}
        {draft.source === 'video' && !locked && draft.videos.length < sourceLimit && <Field label={multi ? "Video files" : "Video"} hint={multi ? "Choose two or three synchronized videos of one take. Maximum 10 GB per file." : "MP4, MOV, AVI, MKV or WebM. Maximum 10 GB."}>
          <input type="file" accept=".mp4,.mov,.avi,.mkv,.webm" multiple={multi} disabled={busy} onChange={event => {
            const files = Array.from(event.target.files ?? []); event.target.value = '';
            if (files.length + draft.videos.length > sourceLimit) { setError(multi ? 'Choose at most three camera views in total.' : 'Choose one video.'); return; }
            void perform(async () => { await save(); for (const file of files) { setNotice(`Adding ${file.name}…`); accept(await workflow.upload(draft.id,file)); } });
          }} /></Field>}
        {busy && <p role="status" className="small muted">{notice || 'Working…'}</p>}
        <div className="setup-video-grid">{draft.videos.map(video => <article className="setup-video stack stack-3" key={video.id}>
          <strong>{video.name}</strong><VideoPreview draftId={draft.id} video={video} />
          <p className="small muted">{duration(video.durationSec)} · {video.width} × {video.height} · {video.fps.toFixed(1)} fps</p>
          {multi && <Field label={`View for ${video.name}`}><select disabled={busy || locked || cameraActive} value={video.role} onChange={event => {
            const role = event.target.value as CameraRole;
            setDraft(current => current ? {...current,videos:current.videos.map(v => v.id === video.id ? {...v,role} : v.role === role ? {...v,role:video.role} : v),setup:{...current.setup,footageReviewed:false,synchronizationConfirmed:false}} : current); setDirty(true);
          }}>{multiRoles.map(value => <option key={value} value={value}>{roleLabels[value]}</option>)}</select></Field>}
          {draft.source === 'video' && !locked && <button className="btn btn--ghost" type="button" disabled={busy} onClick={() => void perform(async () => { await save(); accept(await workflow.remove(draft.id,video.id)); })}>Remove video</button>}
        </article>)}</div>
        {draft.videos.length === 0 && <p className="small muted">{draft.source === 'live' ? 'Save a recording take to preview it here.' : 'Add footage to continue.'}</p>}
        {hasRequiredSources && <label className="setup-check"><input type="checkbox" disabled={busy || locked || cameraActive} checked={setup.footageReviewed} onChange={event => change('footageReviewed',event.target.checked)} /><span>I reviewed the footage and confirmed the participant and task.</span></label>}
        {multi && draft.videos.length === 1 && <p className="workflow-note">Select a second recording to analyze multiple cameras.</p>}
        {multi && draft.videos.length > 1 && <><p className="small muted">Views must show the same take with aligned start times. Matching file durations alone cannot establish synchronization. Align mismatched recordings before importing.</p>
          <label className="setup-check"><input type="checkbox" disabled={busy || locked || cameraActive} checked={setup.synchronizationConfirmed} onChange={event => change('synchronizationConfirmed',event.target.checked)} /><span>I confirmed that these views are synchronized and show the same participant and take.</span></label></>}
        </>}
      </section>}
      {step === 3 && <section className="card stack stack-5"><div className="stack stack-2"><h2>Confirm task conditions</h2><p className="small muted">Enter what you measured or observed. Blank fields remain unknown unless a selected assumption profile supplies them.</p></div>
        <div className="setup-fields">
          <Field label="Handled load (kg)" hint="Enter 0 only when you know there is no handled load."><input type="number" min={0} max={1000} step="0.1" disabled={busy || locked} value={setup.loadKg ?? ''} onChange={event => change('loadKg',event.target.value === '' ? null : Number(event.target.value))} /></Field>
          <Field label="Coupling / grip quality"><select disabled={busy || locked} value={setup.coupling ?? ''} onChange={event => change('coupling',(event.target.value || null) as Setup['coupling'])}><option value="">Unknown</option>{['good','fair','poor','unacceptable'].map(v => <option key={v} value={v}>{v[0].toUpperCase()+v.slice(1)}</option>)}</select></Field>
          <Field label="Load application"><select disabled={busy || locked} value={setup.rulaLoadType ?? ''} onChange={event => change('rulaLoadType',(event.target.value || null) as Setup['rulaLoadType'])}><option value="">Unknown / derive from activity</option><option value="intermittent">Intermittent</option><option value="static_or_repeated">Static or repeated</option></select></Field>
          <Field label="Assumptions for unanswered fields"><select disabled={busy || locked} value={setup.assumptionProfile ?? ''} onChange={event => change('assumptionProfile',(event.target.value || null) as Setup['assumptionProfile'])}><option value="">No assumptions · retain unknowns</option>{Object.entries(profileLabels).map(([key,label]) => <option key={key} value={key}>{label}</option>)}</select></Field>
        </div>
        <details className="workflow-disclosure"><summary>Support, wrist position and force</summary><div className="setup-fields">
          {booleanFields.map(([key,label]) => <Field key={key} label={label}><select disabled={busy || locked} value={setup[key] === null ? '' : String(setup[key])} onChange={event => change(key,event.target.value === '' ? null : event.target.value === 'true')}><option value="">Unknown</option><option value="true">Yes</option><option value="false">No</option></select></Field>)}
        </div></details>
        <details className="workflow-disclosure"><summary>Analysis settings</summary><div className="setup-fields">
          <Field label="Analysis sampling (frames per second)"><select disabled={busy || locked} value={setup.samplingFps} onChange={event => change('samplingFps',Number(event.target.value))}>{[5,10,15,30,60].map(rate => <option key={rate} value={rate}>{rate} fps</option>)}</select></Field>
          <Field label="Open results in"><select disabled={busy || locked} value={setup.primaryMethod} onChange={event => change('primaryMethod',event.target.value as Setup['primaryMethod'])}><option value="reba">REBA · Whole body</option><option value="rula">RULA · Upper limb</option></select></Field>
          <VisionModelSelect disabled={busy || locked} value={setup.visionModel ?? null} onChange={model => change('visionModel',model)} />
        </div></details>
        {setup.assumptionProfile && <details className="workflow-note"><summary>Review assumptions in {profileLabels[setup.assumptionProfile]}</summary><div className="profile-values">{Object.entries(options.data?.profiles[setup.assumptionProfile] ?? {}).map(([key,value]) => <span key={key}>{key.replaceAll('_',' ')} = <strong>{String(value)}</strong></span>)}</div><p className="small">Entered and measured values take precedence. Applied assumptions are recorded with the results.</p></details>}
        <label className="setup-check"><input type="checkbox" disabled={busy || locked} checked={setup.conditionsReviewed} onChange={event => change('conditionsReviewed',event.target.checked)} /><span>I reviewed the entered conditions and the assumptions that may fill unanswered fields.</span></label>
      </section>}
      {step === 4 && <section className="card stack stack-5"><h2>{ready ? 'Analysis is ready for review' : draft.status === 'processing' ? 'Analyzing the recording' : 'Review setup and analyze'}</h2>
        <div className="setup-fields"><div><p className="small muted">Assessment</p><strong>{studyDisplayName(setup.studyName,setup.studyFolder ?? '') || 'Study name needed'} · {setup.participantId || 'Participant needed'}</strong><p>{setup.taskName || 'Task name needed'}</p></div>
          <div><p className="small muted">Footage</p><strong>{draft.videos.length} camera {draft.videos.length === 1 ? 'view' : 'views'}</strong><p>{setup.samplingFps} fps analysis · {setup.primaryMethod.toUpperCase()} opens first</p></div></div>
        <p className="small muted">{setup.loadKg === null ? 'Load is unanswered' : `Entered load ${setup.loadKg} kg`} · {setup.assumptionProfile ? `${profileLabels[setup.assumptionProfile]} assumptions` : 'Unknown inputs remain unknown'}</p>
        {draft.status === 'processing' && <><progress className="workflow-progress" max={100} value={draft.job?.progressPercent ?? 0} aria-label="Analysis progress" /><p role="status">{draft.job?.current?.message || 'Preparing analysis…'} · {Math.round(draft.job?.progressPercent ?? 0)}%</p><p className="small muted">You can leave this page and return from Assessments.</p></>}
        {draft.status === 'failed' && <div role="alert" className="workflow-error">{draft.job?.outcomes.filter(o => !o.ok).map(o => <p key={o.label}>{o.error}</p>)}<p>Your setup and footage are saved. Resolve the reported issue, then retry.</p></div>}
        {draft.status === 'interrupted' && <p role="alert" className="workflow-error">The previous analysis did not publish a matching result. Your setup is saved and can be retried.</p>}
        {!locked && !canAnalyze && <div className="workflow-note">Complete these steps to start analysis.
          <ul>
            {missingDetails && <li>Complete the study, participant and task in Assessment details.</li>}
            {missingFootage && <li>{multi ? 'Select two or three recordings, review the footage and confirm synchronization in Footage.' : 'Select one recording and confirm your review in Footage.'}</li>}
            {!setup.conditionsReviewed && <li>Confirm the inputs in Task conditions.</li>}
          </ul>
        </div>}
        {ready && draft.analysis ? <><p>The results still need the ergonomist's review. Open the evidence, record findings, and mark the assessment reviewed when complete.</p><Link className="btn btn--primary btn--lg" to={`/s/${draft.analysis.sessionId}/${setup.primaryMethod}`}>Review analysis</Link></>
          : draft.status !== 'processing' && <button className="btn btn--primary btn--lg" type="button" disabled={busy || cameraActive || !canAnalyze} onClick={() => void perform(async () => { if (!locked) await save(4); accept(await workflow.analyze(draft.id)); })}>{busy ? 'Starting…' : draft.analysis ? 'Retry analysis' : 'Start analysis'}</button>}
      </section>}
      <footer className="workflow-footer"><div className="source-options">
        {step > 1 && <button className="btn btn--ghost" type="button" disabled={busy || cameraActive} onClick={() => void go(step-1)}>Back</button>}
        {!locked && <button className="btn" type="button" disabled={busy || cameraActive || chooseSingle} onClick={() => void perform(async () => { await save(); })}>{busy ? 'Saving…' : 'Save draft'}</button>}
        <span className="small muted" role="status">{notice || `Saved ${new Date(draft.updatedAt).toLocaleTimeString()}`}</span>
      </div><div className="source-options">
        {addTask && !dirty && !busy && !cameraActive && <Link className="btn btn--ghost" to={addTask}>Add another task for this participant</Link>}
        {step < 4 && <button className="btn btn--primary" type="button" disabled={busy || cameraActive} onClick={() => void go(step+1)}>Continue</button>}</div></footer>
    </div>
  </Shell>;
}
