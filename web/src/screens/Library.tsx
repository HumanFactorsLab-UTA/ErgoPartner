import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { Shell } from "../components/Shell";
import { StudiesFolderPicker } from "../components/StudiesFolderPicker";
import { Field } from "../components/WorkflowParts";
import { ActionMenu, TrashDialog, type TrashSelection } from "../components/AssessmentActions";
import { LibraryTask, ParticipantTasks, taskName, taskState, type TaskRow } from "../components/LibraryTasks";
import { IconGauge, IconSearch } from "../components/Icons";
import { ConfirmButton, EmptyState, ErrorState, Skeleton } from "../components/ui";
import { ApiError, api, useAsync, useDebounced } from "../lib/api";
import { humanise, number } from "../lib/format";
import { studyDisplayName, workflow } from "../lib/workflow";
import type { AnalysisJob, AnalysisScope } from "../lib/types";
import "../styles/library.css";

function JobProgress({ job }: { job: AnalysisJob }) {
  const currentProgress =
    job.current && job.current.totalFrames > 0
      ? (job.current.progressFrame / job.current.totalFrames) * 100
      : 0;
  const overall =
    Number.isFinite(job.progressPercent) && job.progressPercent >= 0
      ? job.progressPercent
      : ((job.completed + currentProgress / 100) / Math.max(1, job.total)) * 100;
  const finished = job.done || ["completed", "completed_with_errors", "failed"].includes(job.status);

  return (
    <section className="analysis-progress" data-finished={finished}>
      <div className="analysis-progress__icon">
        <IconGauge size={20} />
        {!finished ? <span aria-hidden="true" /> : null}
      </div>
      <div className="analysis-progress__main">
        <div className="analysis-progress__head">
          <div>
            <p className="eyebrow">{finished ? "Analysis update" : "Analysis in progress"}</p>
            <h3>
              {finished
                ? job.failed
                  ? "Analysis finished with items to review"
                  : "Results are ready"
                : job.currentLabel ?? "Preparing the analysis queue"}
            </h3>
          </div>
          <strong className="num">{Math.round(Math.min(100, overall))}%</strong>
        </div>
        <div
          className="analysis-progress__track"
          role="progressbar"
          aria-valuenow={Math.round(Math.min(100, overall))}
          aria-valuemin={0}
          aria-valuemax={100}
        >
          <span style={{ width: `${Math.min(100, overall)}%` }} />
        </div>
        <div className="analysis-progress__meta">
          <span>
            {job.completed} of {job.total} trials complete
          </span>
          {job.current ? (
            <span>
              {humanise(job.current.stage)} · frame{" "}
              <span className="num">
                {job.current.progressFrame.toLocaleString()} /{" "}
                {job.current.totalFrames.toLocaleString()}
              </span>
              {job.current.etaSec > 0 ? ` · about ${number(job.current.etaSec / 60, 1)} min left` : ""}
            </span>
          ) : null}
          {job.failed ? <span>{job.failed} failed</span> : null}
        </div>
      </div>
    </section>
  );
}

export default function Library() {
  const [params] = useSearchParams();
  const showingTrash = params.get("view") === "trash";
  const [query,setQuery] = useState("");
  const [filter,setFilter] = useState("all");
  const [job,setJob] = useState<AnalysisJob | null>(null);
  const [jobError,setJobError] = useState<string | null>(null);
  const [starting,setStarting] = useState(false);
  const [removal,setRemoval] = useState<TrashSelection | null>(null);
  const [restoring,setRestoring] = useState<string | null>(null);
  const [message,setMessage] = useState("");
  const settled = useDebounced(query.trim().toLowerCase(),150);
  const {data,error,loading,reload} = useAsync(async signal => {
    const [people,drafts,options] = await Promise.all([api.participants("",signal),workflow.list(signal),workflow.options(signal)]);
    return {people:people.participants,drafts:drafts.assessments,studies:options.studies};
  },[],{keepPrevious:true});
  const trash = useAsync(async signal => showingTrash ? workflow.trash(signal) : {items:[]},[showingTrash]);
  useEffect(() => {
    const controller = new AbortController();
    api
      .activeAnalysisJobs(controller.signal)
      .then((payload) => {
        if (payload.jobs.length) setJob(payload.jobs[0]);
      })
      .catch(() => {
        /* No active job is a valid initial state. */
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!job || job.done || ["completed", "completed_with_errors", "failed"].includes(job.status)) {
      return;
    }
    const controller = new AbortController();
    let timer = 0;
    const poll = () => {
      api
        .analysisJob(job.jobId, controller.signal)
        .then((next) => {
          setJobError(null);
          setJob(next);
          if (next.done || ["completed", "completed_with_errors", "failed"].includes(next.status)) {
            reload();
          }
        })
        .catch((requestError: unknown) => {
          if (!controller.signal.aborted) {
            setJobError(
              requestError instanceof Error
                ? requestError.message
                : "Could not refresh analysis progress",
            );
            timer = window.setTimeout(poll, 1600);
          }
        });
    };
    timer = window.setTimeout(poll, 850);
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [job, reload]);

  const jobActive = Boolean(
    starting ||
      (job && !job.done && !["completed", "completed_with_errors", "failed"].includes(job.status)),
  );
  const run = useCallback(
    async (
      scope: AnalysisScope,
      options: {
        participantKey?: string;
        taskFolder?: string;
        studyFolder?: string;
        rerun?: boolean;
      },
    ) => {
      setStarting(true);
      setJobError(null);
      try {
        const created = await api.startAnalysis(scope, options);
        setJob(created);
        if (
          created.done ||
          ["completed", "completed_with_errors", "failed"].includes(created.status)
        ) {
          reload();
        }
      } catch (requestError) {
        if (requestError instanceof ApiError && requestError.status === 409) reload();
        setJobError(
          requestError instanceof Error ? requestError.message : "The analysis could not start",
        );
      } finally {
        setStarting(false);
      }
    },
    [reload],
  );

  const groups = useMemo(() => {
    if (!data) return {participants:[],loose:[] as TaskRow[],studies:[] as string[],total:0,missing:0};
    const grouped = data.people.map(person => ({
      person, tasks:person.trials.map(trial => ({key:`task:${trial.folder}`,participantKey:person.key,trial} as TaskRow)),
    }));
    const byKey = new Map(grouped.map(group => [group.person.key,group]));
    const loose: TaskRow[] = [];
    for (const draft of data.drafts) {
      const folder = draft.setup.studyFolder;
      const key = draft.published?.participantKey ?? draft.existing?.participantKey ?? (folder && draft.setup.participantId ? `${folder}::${draft.setup.participantId}` : undefined);
      const parent = key ? byKey.get(key) : undefined;
      const published = draft.published ? parent?.tasks.find(t => t.trial?.folder === draft.published?.taskFolder) : undefined;
      if (published) {
        if (!published.draft) published.draft = draft;
      } else {
        (parent?.tasks ?? loose).push({key:draft.id,participantKey:key,draft});
      }
    }
    const studies = [...new Set(grouped.map(group => group.person.study))];
    return {participants:grouped,loose,studies,total:data.people.reduce((n,p) => n+p.trialCount,0),
      missing:data.people.reduce((n,p) => n+p.trials.filter(t => t.status === "not_analysed").length,0)};
  },[data]);

  const studyLabel = (folder:string) => studyDisplayName(data?.studies.find(s => s.folder === folder)?.name,folder);
  const matchesStatus = (task:TaskRow) => {
    const state = taskState(task);
    return filter === "all" || state === filter || (filter === "unfinished" && state !== "ready" && state !== "reviewed");
  };
  const match = (value:string | undefined | null) => String(value ?? "").toLowerCase().includes(settled);
  const visible = groups.participants.map(group => {
    const {person} = group;
    const personMatches = [`Participant ${person.participantId}`,person.label,studyLabel(person.study),person.recordedAt,person.sex,person.dominantHand].some(match);
    const tasks = group.tasks.filter(task => matchesStatus(task) && (personMatches || match(taskName(task)) || match(task.trial?.description)));
    return {...group,visibleTasks:tasks,show:tasks.length > 0 || (!group.tasks.length && filter === "all" && personMatches)};
  }).filter(group => group.show);
  const loose = groups.loose.filter(t => matchesStatus(t) && [taskName(t),t.draft?.setup.participantId,t.draft?.setup.studyName].some(match));
  function resetFolder() {setQuery("");setFilter("all");setJob(null);setJobError(null);setMessage("");reload();trash.reload();}
  async function restore(id:string,label:string) {
    setRestoring(id);setJobError(null);setMessage("");
    try {await workflow.restore(id);setMessage(`${label} restored.`);reload();trash.reload();}
    catch (err) {setJobError(err instanceof Error ? err.message : "Could not restore this item.");}
    finally {setRestoring(null);}
  }
  return <Shell crumbs={[{label:"Home",to:"/"},{label:"Assessments",to:showingTrash ? "/assessments" : undefined},...(showingTrash ? [{label:"Trash"}] : [])]}>
    <div className="library">
      <header className="library__hero"><div><h1>{showingTrash ? "Trash" : "Assessments"}</h1>
        <p className="lede">{showingTrash ? "Restore participants and tasks to their original study." : "Find a participant, continue a task or review a completed assessment."}</p></div>
        <div className="row row-2">
          {showingTrash ? <Link className="btn" to="/assessments">Back to assessments</Link> : <>
            <Link className="btn btn--primary" to="/new">New assessment</Link>
            <ActionMenu label="More library actions">
              <Link className="btn btn--ghost" to="/assessments?view=trash">Open Trash</Link>
              {groups.missing > 0 && <button className="btn btn--ghost" type="button" disabled={jobActive} onClick={() => void run("all",{})}>Analyze all missing tasks</button>}
              {groups.total > 0 && <ConfirmButton className="btn btn--ghost" disabled={jobActive} confirmLabel={`Replace ${groups.total} results`} onConfirm={() => void run("all",{rerun:true})}>Re-run all recordings</ConfirmButton>}
            </ActionMenu></>}
        </div></header>
      {message && <p className="workflow-note small" role="status">{message} {!showingTrash && <Link to="/assessments?view=trash">Open Trash</Link>}</p>}
      {job && !showingTrash && <JobProgress job={job} />}
      {jobError && <div className="library__notice" role="alert"><span>{jobError}</span><button type="button" onClick={() => setJobError(null)}>Dismiss</button></div>}
      {showingTrash ? <>
        {trash.error ? <ErrorState message={trash.error} onRetry={trash.reload} /> : trash.loading ? <Skeleton height={80} />
          : !trash.data?.items.length ? <EmptyState title="Trash is empty" detail="Participants and tasks you remove from this library will appear here." /> :
          <div className="participant-block">{trash.data.items.map(item => <article key={item.id} className="library-trash-row">
            <div className="stack stack-1"><strong>{item.label}</strong><p className="small muted">{item.kind === "participant" ? `Participant · ${item.taskCount} ${item.taskCount === 1 ? "task" : "tasks"}` : item.kind === "draft" ? "Saved setup" : "Task"} · Moved {new Date(item.deletedAt).toLocaleDateString()}</p></div>
            <button className="btn" type="button" disabled={restoring !== null} aria-label={`Restore ${item.label}`} onClick={() => void restore(item.id,item.label)}>{restoring === item.id ? "Restoring…" : "Restore"}</button>
          </article>)}</div>}
      </> : <>
        <section className="library__controls" aria-label="Find assessments"><label className="field"><IconSearch size={15} />
          <input type="search" value={query} placeholder="Search studies, participants or tasks" aria-label="Search assessments" onChange={e => setQuery(e.target.value)} /></label>
          <Field label="Status"><select value={filter} onChange={e => setFilter(e.target.value)}><option value="all">All tasks</option><option value="unfinished">In progress or awaiting analysis</option><option value="ready">Needs review</option><option value="reviewed">Reviewed</option></select></Field>
        </section>
        {error ? <ErrorState message={error} onRetry={reload} /> : loading && !data ? <Skeleton height={140} /> : <>
          {groups.studies.map(study => {
            const people = visible.filter(p => p.person.study === study);
            if (!people.length) return null;
            return <section key={study} className="library__participants" aria-label={studyLabel(study)}>
              <div className="library__study-head"><h2>{studyLabel(study)}</h2><Link className="btn btn--ghost" to={`/new?${new URLSearchParams({study})}`}>Add participant</Link></div>
              {people.map(p => <ParticipantTasks key={p.person.key} person={p.person} tasks={p.visibleTasks} allTasks={p.tasks} jobActive={jobActive} onRun={run} onTrash={setRemoval} />)}
            </section>;
          })}
          {loose.length > 0 && <section className="stack stack-3"><h2>Other saved assessments</h2><div className="participant-block">
            {loose.map(t => <LibraryTask key={t.key} task={t} jobActive={jobActive} onRun={run} onTrash={setRemoval} />)}</div></section>}
          {!visible.length && !loose.length && <EmptyState title={settled || filter !== "all" ? "No matching assessments" : "No assessments in this folder"} detail={settled || filter !== "all" ? "Try another search or choose All tasks." : "Create an assessment to add a participant and their first task, or select a studies folder below."} />}
        </>}
        <details className="workflow-disclosure"><summary>Studies folder</summary><StudiesFolderPicker disabled={jobActive} onChanged={resetFolder} /></details>
      </>}
      {removal && <TrashDialog selection={removal} onClose={() => setRemoval(null)} onMoved={() => {
        setMessage(`${removal.label} moved to Trash.`);setRemoval(null);reload();
      }} />}
    </div>
  </Shell>;
}
