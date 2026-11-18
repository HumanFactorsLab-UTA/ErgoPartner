import { useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { Shell } from "../components/Shell";
import { Field } from "../components/WorkflowParts";
import { StudiesFolderPicker } from "../components/StudiesFolderPicker";
import { ErrorState } from "../components/ui";
import { api, useAsync } from "../lib/api";
import {
  workflow,
  reviewLink,
  studyDisplayName,
  type AssessmentDraft,
  type CameraMode,
  type TaskGroup,
} from "../lib/workflow";

const NEW_STUDY = "__create_study__";

export default function NewAssessment() {
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const initialSource = params.get("source");
  const [source, setSource] = useState<AssessmentDraft["source"]>(
    initialSource === "existing" || initialSource === "live" ? initialSource : "video",
  );
  const [cameraMode, setCameraMode] = useState<CameraMode>("single");
  const [participant, setParticipant] = useState(params.get("participant") ?? "");
  const [trial, setTrial] = useState(params.get("trial") ?? "");
  const [studyChoice, setStudyChoice] = useState<string | null>(params.get("study"));
  const [studyName, setStudyName] = useState("");
  const [participantId, setParticipantId] = useState("");
  const [taskName, setTaskName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const catalog = useAsync((signal) => api.participants("", signal), []);
  const options = useAsync((signal) => workflow.options(signal), []);
  const people = catalog.data?.participants ?? [];
  const studies = options.data?.studies ?? [];
  const person = people.find((p) => p.key === participant);
  const selected = person?.trials.find((t) => t.folder === trial);
  const study =
    studyChoice ??
    person?.study ??
    (studies.length === 1 ? studies[0].folder : studies.length ? "" : NEW_STUDY);
  const peopleInStudy = people.filter((p) => p.study === study);
  const existingPerson = peopleInStudy.find((p) => p.key === participant);
  const existing = source === "existing";
  const catalogReady = Boolean(catalog.data && options.data);
  const missingParent = !existing && Boolean(participant) && catalogReady && !existingPerson;
  const studyAvailable = study === NEW_STUDY || studies.some((s) => s.folder === study);
  const detailsReady =
    studyAvailable &&
    !missingParent &&
    (study !== NEW_STUDY || Boolean(studyName.trim())) &&
    Boolean(taskName.trim());
  const canCreate =
    catalogReady &&
    (existing
      ? Boolean(selected && (cameraMode === "single" || (selected.cameras ?? 0) >= 2))
      : detailsReady);

  function resetFolder() {
    setParticipant("");
    setTrial("");
    setStudyChoice(null);
    catalog.reload();
    options.reload();
  }
  async function start(event: React.FormEvent) {
    event.preventDefault();
    if (busy || !canCreate) return;
    setBusy(true);
    setError(null);
    try {
      let group: TaskGroup | undefined;
      if (!existing) {
        group = { taskName: taskName.trim() };
        if (!existingPerson) {
          if (study === NEW_STUDY) group.studyName = studyName.trim();
          else group.studyFolder = study;
          if (participantId.trim()) group.participantId = participantId.trim();
        }
      }
      const draft = await workflow.create(
        source,
        existing ? participant : existingPerson?.key,
        existing ? trial : undefined,
        cameraMode,
        group,
      );
      navigate(`/assessments/${draft.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not create the assessment.");
      setBusy(false);
    }
  }
  return (
    <Shell
      crumbs={[
        { label: "Home", to: "/" },
        { label: "Assessments", to: "/assessments" },
        { label: "New assessment" },
      ]}
    >
      <div className="assessment-setup stack stack-5">
        <form
          id="create-assessment"
          className="stack stack-5"
          onSubmit={(event) => void start(event)}
        >
          <header className="stack stack-2">
            <h1>{params.has("participant") && !existing ? "Add a task" : "New assessment"}</h1>
            <p className="lede">Each task has its own recording, analysis and review.</p>
          </header>
          <div className="source-options" role="group" aria-label="Recording source">
            {(
              [
                ["video", "Select video files"],
                ["live", "Record live"],
                ["existing", "Existing study recording"],
              ] as const
            ).map(([value, label]) => (
              <button
                className={`btn ${source === value ? "btn--primary" : ""}`}
                type="button"
                key={value}
                aria-pressed={source === value}
                disabled={busy}
                onClick={() => {
                  setSource(value);
                  setError(null);
                }}
              >
                {label}
              </button>
            ))}
          </div>
          {catalog.error && <ErrorState message={catalog.error} onRetry={catalog.reload} />}
          {options.error && <ErrorState message={options.error} onRetry={options.reload} />}
          <section className="card stack stack-4">
            <h2>{existing ? "Choose a task recording" : "Study, participant and task"}</h2>
            {existing ? (
              <>
                <Field label="Participant and study">
                  <select
                    disabled={busy}
                    value={participant}
                    onChange={(event) => {
                      setParticipant(event.target.value);
                      setTrial("");
                      setStudyChoice(null);
                    }}
                  >
                    <option value="">Choose a participant</option>
                    {people
                      .filter((p) => p.trials.length)
                      .map((p) => (
                        <option key={p.key} value={p.key}>
                          {p.label} · {studyDisplayName(p.studyName ?? studies.find((s) => s.folder === p.study)?.name, p.study)}
                        </option>
                      ))}
                  </select>
                </Field>
                <Field label="Recording">
                  <select
                    value={trial}
                    disabled={busy || !person}
                    onChange={(event) => setTrial(event.target.value)}
                  >
                    <option value="">Choose a task recording</option>
                    {person?.trials.map((t) => (
                      <option key={t.folder} value={t.folder}>
                        {t.name} · {t.cameras ?? 0} {t.cameras === 1 ? "view" : "views"}
                      </option>
                    ))}
                  </select>
                </Field>
                {selected?.sessionId && (
                  <div className="workflow-note">
                    <p>This recording has a completed analysis.</p>
                    <Link to={reviewLink(selected.sessionId, selected.primaryTool ?? "reba")}>
                      Review existing analysis
                    </Link>
                  </div>
                )}
              </>
            ) : (
              <div className="setup-fields">
                <Field label="Study">
                  <select
                    required
                    disabled={busy || !catalogReady}
                    value={study}
                    onChange={(event) => {
                      setStudyChoice(event.target.value);
                      setParticipant("");
                      setParticipantId("");
                    }}
                  >
                    <option value="" disabled>
                      Choose a study
                    </option>
                    {studies.map((s) => (
                      <option key={s.folder} value={s.folder}>
                        {studyDisplayName(s.name, s.folder)}
                      </option>
                    ))}
                    <option value={NEW_STUDY}>Create a new study</option>
                  </select>
                </Field>
                {study === NEW_STUDY && (
                  <Field label="Study name">
                    <input
                      required
                      maxLength={160}
                      value={studyName}
                      disabled={busy}
                      onChange={(e) => setStudyName(e.target.value)}
                      placeholder="Packing workstation review"
                    />
                  </Field>
                )}
                <Field label="Participant">
                  <select
                    disabled={busy || !study || !catalogReady}
                    value={missingParent ? participant : (existingPerson?.key ?? "")}
                    onChange={(event) => {
                      setParticipant(event.target.value);
                      setParticipantId("");
                    }}
                  >
                    {missingParent && (
                      <option value={participant} disabled>
                        Previous participant unavailable
                      </option>
                    )}
                    <option value="">Create a new participant</option>
                    {peopleInStudy.map((p) => (
                      <option key={p.key} value={p.key}>
                        Participant {p.participantId}
                      </option>
                    ))}
                  </select>
                </Field>
                {!existingPerson && !missingParent && (
                  <Field
                    label="Participant ID (optional)"
                    hint="Leave blank to assign the next participant number."
                  >
                    <input
                      maxLength={64}
                      value={participantId}
                      disabled={busy}
                      onChange={(e) => setParticipantId(e.target.value)}
                      placeholder="Assign automatically"
                    />
                  </Field>
                )}
                <Field label="Task name">
                  <input
                    required
                    maxLength={160}
                    value={taskName}
                    disabled={busy}
                    onChange={(e) => setTaskName(e.target.value)}
                    placeholder="Lift boxes to shelf"
                  />
                </Field>
                {missingParent && (
                  <p className="workflow-note" role="status">
                    Choose an available participant in this study, or select Create a new
                    participant.
                  </p>
                )}
              </div>
            )}
            <Field
              label="Camera setup"
              hint={
                cameraMode === "multi"
                  ? "Use two or three synchronized views of this task."
                  : "Use one view of this task."
              }
            >
              <select
                disabled={busy}
                value={cameraMode}
                onChange={(event) => setCameraMode(event.target.value as CameraMode)}
              >
                <option value="single">Single camera</option>
                <option value="multi">Multiple cameras</option>
              </select>
            </Field>
            {existing && selected && cameraMode === "multi" && (selected.cameras ?? 0) < 2 && (
              <p className="workflow-note">
                This recording has one view. Choose Single camera to continue.
              </p>
            )}
          </section>
        </form>
        <details className="workflow-disclosure">
          <summary>Studies folder</summary>
          <StudiesFolderPicker disabled={busy} onChanged={resetFolder} />
        </details>
        {error && (
          <p role="alert" className="workflow-error">
            {error}
          </p>
        )}
        <div className="workflow-footer">
          <Link to="/assessments" className="btn btn--ghost">
            Back to assessments
          </Link>
          <button
            type="submit"
            form="create-assessment"
            className="btn btn--primary btn--lg"
            disabled={busy || !canCreate}
          >
            {busy ? "Preparing…" : "Create assessment"}
          </button>
        </div>
      </div>
    </Shell>
  );
}
