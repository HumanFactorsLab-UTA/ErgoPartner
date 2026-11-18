import { useEffect, useRef, useState } from "react";
import { ApiError, useAsync, useDebounced } from "../lib/api";
import { clock, humanise } from "../lib/format";
import {
  checkLabels, checks, focusedReview, type Answers, type Check, type Count,
  type ReviewCase, type ReviewImage, type ReviewMoment,
} from "../lib/focusedReview";
import type { MethodKey } from "../lib/types";
import { ErrorState, Segmented } from "./ui";
import "../styles/focused-review.css";

type Mode = "assisted" | "blind";
type Action = "predict" | "reference" | "decision";
const percent = (value: number) => `${Math.round(value * 100)}%`;
const imageName = (image: ReviewImage) => {
  const detail = image.view === "crop" ? ` · ${humanise(image.region || "detail")}`
    : image.view === "overlay" ? " · Worker highlighted" : "";
  return `Frame ${image.frame}${detail}`;
};
const matchName = (metric: Count) =>
  metric.accuracy == null ? "—" : `${metric.correct} / ${metric.n} (${percent(metric.accuracy)})`;

export function TestResults({ sessionId }: { sessionId?: string }) {
  const {data, error, reload} = useAsync(signal => focusedReview.evaluation(sessionId, signal), [sessionId]);
  const [group, setGroup] = useState("all");
  if (error) return <ErrorState message={error} onRetry={reload} />;
  if (!data) return <p role="status" className="small muted">Loading test results…</p>;
  const groupOptions = [
    ...Object.entries(data.byVisibility).map(([key, value]) => ({id: `visibility-${key}`, label: `${humanise(key)} visibility`, ...value})),
    ...Object.entries(data.byCameraMode).map(([key, value]) => ({id: `cameras-${key}`, label: key === "multi" ? "Multiple cameras" : "Single camera", ...value})),
  ];
  const selected = groupOptions.find(item => item.id === group);
  const metrics = selected?.metrics ?? data.metrics;
  return <div className="focused-results">
    <p className="small"><strong>{data.pairedCases} compared</strong> · {data.labelledCases} independently labeled · {data.assistedCases} assisted
      {sessionId == null && data.sessionCount != null ? ` · ${data.sessionCount} assessment${data.sessionCount === 1 ? "" : "s"}` : ""}</p>
    {Boolean(data.skippedSessions) && <p className="small muted">
      Current review evidence was unavailable for {data.skippedSessions} saved assessment{data.skippedSessions === 1 ? "" : "s"}.
    </p>}
    {!data.pairedCases ? <p className="small muted">Save blind labels and run the AI check on the same moment to see agreement.</p> : <>
      <label className="small">Show results for{" "}
        <select value={group} onChange={event => setGroup(event.target.value)}>
          <option value="all">All compared moments</option>
          {groupOptions.map(item => <option key={item.id} value={item.id}>{item.label} ({item.caseCount})</option>)}
        </select>
      </label>
      <div className="focused-results__table"><table>
        <caption className="sr-only">Model performance against independent labels{sessionId ? " for this assessment" : " across assessments"}</caption>
        <thead><tr><th scope="col">Check</th><th scope="col">Exact agreement</th><th scope="col">Correctly called unclear</th><th scope="col">Overconfident calls</th><th scope="col">Missed visible cases</th></tr></thead>
        <tbody>{checks.map(key => <tr key={key}><th scope="row">{checkLabels[key]}</th>
          <td>{matchName(metrics[key])}</td><td>{metrics[key].correctAbstentions} / {metrics[key].unclearCases}</td><td>{metrics[key].falseDefinites}</td><td>{metrics[key].unnecessaryAbstentions}</td>
        </tr>)}</tbody>
      </table></div>
      <p className="small muted">Overconfident calls are definite answers on examples the reviewer marked unclear. Missed visible cases are AI abstentions where the reviewer could judge the footage.</p>
      <p className="micro muted">{data.model} · {data.promptVersion}</p>
    </>}
  </div>;
}

function CaseEditor({sessionId, moment, mode, cameraLabel, onSaved}: {
  sessionId: string; moment: ReviewMoment; mode: Mode; cameraLabel: string; onSaved: () => void;
}) {
  const initial = useAsync(signal => focusedReview.case(sessionId, moment, signal), [sessionId, moment.frame, moment.camera, moment.method, moment.person]);
  const [saved, setSaved] = useState<ReviewCase | null>(null);
  const review = saved ?? initial.data;
  const [edits, setEdits] = useState<Partial<Answers>>({});
  const [selected, setSelected] = useState<number | null>(null);
  const [loaded, setLoaded] = useState<Set<string>>(new Set());
  const [imageAttempt, setImageAttempt] = useState(0);
  const [imageFailed, setImageFailed] = useState(false);
  const [busy, setBusy] = useState<Action | null>(null);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const active = useRef<AbortController | null>(null);
  const zoom = useRef<HTMLDialogElement>(null);
  useEffect(() => () => active.current?.abort(), []);
  if (initial.error) return <ErrorState message={initial.error} onRetry={initial.reload} />;
  if (!review) return <p className="small muted" role="status">Loading this view…</p>;

  const {evidence, prediction, reference, decision, questions} = review;
  const blind = mode === "blind";
  const blocked = blind && !reference && !review.blindEligible;
  const canAnswer = blind ? !blocked && !reference : Boolean(prediction);
  const currentImage = evidence.find(item => item.imageIndex === selected)
    ?? evidence.find(item => item.frame === moment.frame && item.view === "raw")
    ?? evidence[0];
  const imagesReady = evidence.length > 0 && evidence.every(image => loaded.has(image.imageUrl));
  const predicted = Object.fromEntries(checks.map(key => [key, prediction?.answers[key].value]));
  const values = {...(blind ? reference?.answers : decision?.answers ?? predicted), ...edits};
  const complete = checks.every(key => Boolean(values[key]));
  const unchanged = Boolean(decision && checks.every(key => decision.answers[key] === values[key]));
  const optionLabel = (key: Check, value: string) =>
    questions.find(q => q.key === key)?.options.find(option => option.value === value)?.label ?? humanise(value);
  const edit = (key: Check, value: string) => {
    setNotice("");
    setError("");
    setEdits(prior => key === "visibility" && value === "unclear"
      ? {...prior, visibility: value, grip: "unclear", armSupport: "unclear"} : {...prior, [key]: value});
  };
  async function submit(action: Action) {
    if (active.current || !review) return;
    const controller = new AbortController();
    active.current = controller;
    setBusy(action);
    setNotice("");
    setError("");
    try {
      const result = await focusedReview.save(sessionId, action, review, action === "predict" ? undefined : values as Answers, controller.signal);
      if (controller.signal.aborted) return;
      setSaved(result);
      setEdits({});
      setNotice(action === "reference" ? "Independent labels saved." : action === "decision" ? "Review saved." : "");
      onSaved();
    } catch (failure) {
      if (controller.signal.aborted) return;
      setError(failure instanceof Error ? failure.message : "The review could not be saved.");
      if (failure instanceof ApiError && failure.status === 409) {
        setSaved(null); setEdits({}); setLoaded(new Set()); initial.reload();
      }
    } finally {
      if (!controller.signal.aborted) {setBusy(null); active.current = null;}
    }
  }
  return <div className="focused-editor">
    <div className="focused-evidence">
      {currentImage && <>
        <button type="button" className="focused-evidence__large" aria-label={`Enlarge ${imageName(currentImage)}`}
          onClick={() => zoom.current?.showModal()}>
          <img src={currentImage.imageUrl} alt={`${cameraLabel}, ${imageName(currentImage)}`}/>
        </button>
        <div className="focused-evidence__caption"><strong>{imageName(currentImage)}</strong>
          <span>{currentImage.frame === moment.frame ? "Target" : "Nearby moment"} · {cameraLabel} · {clock(currentImage.sec)}</span>
        </div>
        <dialog ref={zoom} className="focused-zoom" aria-label="Enlarged review evidence">
          <div className="focused-evidence__caption"><strong>{cameraLabel} · {imageName(currentImage)}</strong>
            <button type="button" className="btn btn--sm" onClick={() => zoom.current?.close()}>Close image</button>
          </div>
          <img src={currentImage.imageUrl} alt={`${cameraLabel}, ${imageName(currentImage)} at ${clock(currentImage.sec)}`}/>
        </dialog>
        <div className="focused-evidence__strip" role="group" aria-label="Sampled images">
          {evidence.map(image => <button type="button" key={image.imageIndex} aria-pressed={image.imageIndex === currentImage.imageIndex}
            aria-label={`View image ${image.imageIndex}, ${imageName(image)}`} onClick={() => setSelected(image.imageIndex)}>
            <img key={imageAttempt} src={image.imageUrl} alt="" onError={() => setImageFailed(true)}
              onLoad={() => setLoaded(prior => prior.has(image.imageUrl) ? prior : new Set(prior).add(image.imageUrl))}/>
            <span>{image.view === "crop" ? "Detail" : image.frame === moment.frame ? "Target" : "Frame"} {image.frame}</span>
          </button>)}
        </div>
      </>}
      {!evidence.length && <p role="status" className="small muted">No images are available for this view.</p>}
      {imageFailed && !imagesReady && <div role="alert" className="small">Some evidence images could not be loaded.{" "}
        <button type="button" className="btn btn--sm" onClick={() => {setImageFailed(false); setImageAttempt(v => v + 1);}}>Retry images</button>
      </div>}
    </div>
    <div className="focused-questions">
      {blocked ? <p role="status">The AI suggestions for this moment have already been revealed. Choose a new test moment for independent labeling.</p> : <>
        {questions.map(q => {
          const answer = prediction?.answers[q.key];
          const showAI = Boolean(answer && (!blind || reference));
          return <fieldset key={q.key} className="focused-question">
            <legend>{checkLabels[q.key]}</legend>
            <p className="small">{q.prompt}</p>
            {(canAnswer || (blind && reference)) && <div className="focused-options">
              {q.options.map(option => <label key={option.value} data-picked={values[q.key] === option.value}>
                <input type="radio" name={`focused-${mode}-${q.key}`} value={option.value}
                  checked={values[q.key] === option.value} onChange={() => edit(q.key, option.value)}
                  disabled={Boolean(busy) || !canAnswer || (q.key !== "visibility" && values.visibility === "unclear")}
                />{option.label}
              </label>)}
            </div>}
            {showAI && answer && <div className="focused-answer">
              <p className="small"><strong>AI · {optionLabel(q.key, answer.value)}.</strong> {answer.reason}</p>
              {answer.evidence.map(image =>
                <button key={image.imageIndex} type="button" className="cite" onClick={() => setSelected(image.imageIndex)}>
                  {imageName(image)}
                </button>)}
            </div>}
          </fieldset>;
        })}
        <div className="focused-actions">
          {blind && !reference && <button type="button" className="btn btn--primary" disabled={!complete || !imagesReady || Boolean(busy)}
            onClick={() => void submit("reference")}>{busy === "reference" ? "Saving…" : "Save independent labels"}</button>}
          {(!blind || reference) && !prediction && <button type="button" className="btn btn--primary" disabled={!imagesReady || Boolean(busy)}
            onClick={() => void submit("predict")}>{busy === "predict" ? "Reading sampled images…" : reference ? "Compare with AI" : "Get AI suggestions"}</button>}
          {!blind && prediction && <button type="button" className="btn btn--primary" disabled={!complete || !imagesReady || unchanged || Boolean(busy)}
            onClick={() => void submit("decision")}>{busy === "decision" ? "Saving…" : unchanged ? "Review saved" : decision ? "Save changes" : "Confirm answers"}</button>}
          {blind && reference && <span className="small muted">Independent labels locked</span>}
        </div>
        {!blind && !prediction && !reference && <p className="small muted">To include this moment in an accuracy test, save blind labels before requesting suggestions.</p>}
      </>}
      {notice && <p role="status" className="small">{notice}</p>}
      {error && <p role="alert" className="workflow-error">{error}</p>}
    </div>
  </div>;
}

export function FocusedReview({sessionId, frame, method, person, camera, cameras, onCamera, onFrame}: {
  sessionId: string; frame: number | null; method: MethodKey; person: string | null; camera: number;
  cameras: {index: number; label: string}[]; onCamera: (index: number) => void; onFrame: (frame: number) => void;
}) {
  const [mode, setMode] = useState<Mode>("assisted");
  const [results, setResults] = useState(false);
  const [revision, setRevision] = useState(0);
  const settledFrame = useDebounced(frame, 240);
  const samples = useAsync(signal => person && mode === "blind"
    ? focusedReview.samples(sessionId, person, signal) : Promise.resolve({frames: []}), [sessionId, person, mode]);
  return <section className="focused-review" id="focused-review" tabIndex={-1} aria-labelledby="focused-review-title">
    <div className="focused-review__head">
      <div><h2 id="focused-review-title">Focused visual review</h2>
        <p className="small muted">Check the worker’s grip, arm support, and visibility in the selected camera.</p></div>
      <Segmented label="Visual review mode" value={mode} onChange={setMode}
        options={[{value:"assisted", label:"AI-assisted review"}, {value:"blind",label:"Blind test"}]} />
    </div>
    <div className="focused-review__tools">
      <span className="small"><strong>{frame == null ? "Select a frame" : `Frame ${frame}`}</strong>{person ? ` · ${person}` : ""}</span>
      {(cameras.length > 1 || (cameras.length === 1 && cameras[0].index !== camera)) && <label className="small">Camera{" "}
        <select aria-label="Review camera" value={cameras.some(c => c.index === camera) ? camera : ""} onChange={event => onCamera(Number(event.target.value))}>
        {!cameras.some(c => c.index === camera) && <option value="" disabled>Select a source</option>}
        {cameras.map(item => <option key={item.index} value={item.index}>{item.label}</option>)}
      </select></label>}
      {mode === "blind" && Boolean(samples.data?.frames.length) && <label className="small">Test moment{" "}
        <select value={samples.data!.frames.some(item => item.frame === frame) ? String(frame) : ""} onChange={event => onFrame(Number(event.target.value))}>
          <option value="" disabled>Choose a sampled moment</option>
          {samples.data!.frames.map(item => <option key={item.frame} value={item.frame}>{clock(item.sec)} · Frame {item.frame}</option>)}
        </select>
      </label>}
    </div>
    {mode === "blind" && <p className="small muted">Label the images yourself, then compare your saved answers with the model’s reading.</p>}
    {mode === "blind" && samples.error && <ErrorState message={samples.error} onRetry={samples.reload} />}
    {frame !== settledFrame ? <p role="status" className="small muted">Preparing the selected moment…</p>
      : frame != null && person && cameras.some(item => item.index === camera)
      ? <CaseEditor key={`${sessionId}:${frame}:${person}:${method}:${camera}:${mode}`} sessionId={sessionId}
          moment={{frame, person, method, camera}} mode={mode} cameraLabel={cameras.find(item => item.index === camera)!.label}
          onSaved={() => setRevision(value => value + 1)} />
      : <p className="small muted">Select a tracked worker and a moment with available source footage.</p>}
    <div className="focused-results__wrap">
      <button type="button" aria-expanded={results} className="btn btn--ghost btn--sm" onClick={() => setResults(value => !value)}>
        {results ? "Hide test results" : "View test results"}
      </button>
      {results && <TestResults key={revision} sessionId={sessionId} />}
    </div>
  </section>;
}
