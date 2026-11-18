import { Link } from "react-router-dom";
import { type AnalysisScope, type ParticipantCard, type TrialCard } from "../lib/types";
import { reviewLink, type AssessmentDraft } from "../lib/workflow";
import { duration } from "../lib/format";
import { ActionMenu, type TrashSelection } from "./AssessmentActions";
import { draftDestination, draftStatus, draftTrash } from "./WorkflowParts";
import { BandBadge, ConfirmButton } from "./ui";
import { ReportButton } from "./ReportButton";

export interface TaskRow {
  key: string;
  participantKey?: string;
  trial?: TrialCard;
  draft?: AssessmentDraft;
}
export type RunAction = (
  scope: AnalysisScope,
  options: { participantKey?: string; taskFolder?: string; studyFolder?: string; rerun?: boolean },
) => void;
export function taskName(task: TaskRow) {
  return task.trial?.name || task.draft?.setup.taskName || "Untitled assessment";
}

export function taskState(task: TaskRow) {
  if (task.draft && task.draft.status !== "ready") return task.draft.status;
  if (task.trial?.status === "analysed")
    return task.trial.reviewStatus === "reviewed" ? "reviewed" : "ready";
  if (task.draft) return task.draft.status;
  return task.trial?.status === "incomplete" ? "failed" : "recorded";
}
const stateLabels = { ...draftStatus, reviewed: "Reviewed", recorded: "Awaiting setup" };

export function LibraryTask({
  task,
  jobActive,
  onRun,
  onTrash,
}: {
  task: TaskRow;
  jobActive: boolean;
  onRun: RunAction;
  onTrash: (item: TrashSelection) => void;
}) {
  const { trial, draft, participantKey } = task;
  const name = taskName(task);
  const state = taskState(task);
  const ready = state === "ready" || state === "reviewed";
  const session = trial?.sessionId || draft?.analysis?.sessionId;
  const setupLink = draft
    ? draftDestination(draft)
    : `/new?${new URLSearchParams({ source: "existing", participant: participantKey ?? "", trial: trial?.folder ?? "" })}`;
  const trash: TrashSelection | null =
    trial && participantKey
      ? { label: name, target: { kind: "task", participantKey, taskFolder: trial.folder } }
      : draft
        ? draftTrash(draft)
        : null;
  const cameras = trial?.cameras ?? draft?.videos.length;
  const info = [
    draft?.source === "existing" && !draft.published
      ? "Saved setup for an existing recording"
      : null,
    cameras ? `${cameras} ${cameras === 1 ? "view" : "views"}` : null,
    trial?.durationSec ? duration(trial.durationSec) : null,
  ]
    .filter(Boolean)
    .join(" · ");
  return (
    <article
      className="trial-row"
      data-status={ready ? "analysed" : state === "failed" ? "incomplete" : "not_analysed"}
    >
      <div className="trial-row__main">
        <span className="trial-row__state" aria-hidden="true" />
        <div className="stack stack-1">
          <strong>{name}</strong>
          <span className="micro muted">
            {[stateLabels[state], info].filter(Boolean).join(" · ")}
          </span>
        </div>
      </div>
      <div className="trial-row__scores">
        {ready && trial && (
          <>
            {(["reba", "rula"] as const).map((m) => (
              <span key={m}>
                <em>{m.toUpperCase()}</em>
                <BandBadge band={trial[m]?.band ?? null} score={trial[m]?.maximum} size="sm" />
              </span>
            ))}
          </>
        )}
      </div>
      <div className="trial-row__action">
        {ready && session ? (
          <Link
            className="btn"
            to={reviewLink(session, trial?.primaryTool ?? draft?.setup.primaryMethod)}
          >
            Review results
          </Link>
        ) : !draft && trial?.status === "incomplete" ? (
          <button
            className="btn"
            type="button"
            disabled={jobActive}
            onClick={() =>
              onRun("trial", { participantKey, taskFolder: trial.folder, rerun: true })
            }
          >
            Retry analysis
          </button>
        ) : (
          <Link className="btn" to={setupLink}>
            {state === "processing"
              ? "View progress"
              : draft
                ? "Continue setup"
                : "Set up assessment"}
          </Link>
        )}
        <ActionMenu label={`More actions for ${name}`}>
          {ready && session && <ReportButton sessionId={session} />}
          {ready && trial && (
            <ConfirmButton
              className="btn btn--ghost"
              disabled={jobActive}
              confirmLabel="Replace this result"
              onConfirm={() =>
                onRun("trial", { participantKey, taskFolder: trial.folder, rerun: true })
              }
            >
              Run analysis again
            </ConfirmButton>
          )}
          {trash && (
            <button
              className="btn btn--ghost"
              disabled={state === "processing"}
              type="button"
              onClick={() => onTrash(trash)}
            >
              Move to Trash
            </button>
          )}
        </ActionMenu>
      </div>
    </article>
  );
}

export function ParticipantTasks({
  person,
  tasks,
  allTasks,
  jobActive,
  onRun,
  onTrash,
}: {
  person: ParticipantCard;
  tasks: TaskRow[];
  allTasks: TaskRow[];
  jobActive: boolean;
  onRun: RunAction;
  onTrash: (item: TrashSelection) => void;
}) {
  const name = `Participant ${person.participantId}`;
  const pending = person.trials.filter((t) => t.status === "not_analysed").length;
  const processing = allTasks.some((t) => taskState(t) === "processing");
  return (
    <section className="participant-block" aria-label={name}>
      <header className="participant-block__head">
        <div className="stack stack-1">
          <h3>{name}</h3>
          <p className="micro muted">
            {allTasks.length} {allTasks.length === 1 ? "task" : "tasks"}
          </p>
        </div>
        <div className="row row-2">
          <Link
            className="btn"
            to={`/new?${new URLSearchParams({ participant: person.key, study: person.study })}`}
          >
            Add task
          </Link>
          <ActionMenu label={`More actions for ${name}`}>
            {pending > 0 && (
              <button
                className="btn btn--ghost"
                type="button"
                disabled={jobActive}
                onClick={() => onRun("participant", { participantKey: person.key })}
              >
                Analyze missing tasks
              </button>
            )}
            {person.trialCount > 0 && (
              <ConfirmButton
                className="btn btn--ghost"
                disabled={jobActive}
                confirmLabel="Replace participant results"
                onConfirm={() => onRun("participant", { participantKey: person.key, rerun: true })}
              >
                Re-run participant analysis
              </ConfirmButton>
            )}
            <button
              className="btn btn--ghost"
              disabled={processing}
              type="button"
              onClick={() =>
                onTrash({
                  label: name,
                  target: { kind: "participant", participantKey: person.key },
                  taskCount: allTasks.length,
                })
              }
            >
              Move participant to Trash
            </button>
          </ActionMenu>
        </div>
      </header>
      {tasks.map((task) => (
        <LibraryTask
          key={task.key}
          task={task}
          jobActive={jobActive}
          onRun={onRun}
          onTrash={onTrash}
        />
      ))}
      {!allTasks.length && (
        <p className="library__empty-participant small muted">
          Add the first task for this participant.
        </p>
      )}
    </section>
  );
}
