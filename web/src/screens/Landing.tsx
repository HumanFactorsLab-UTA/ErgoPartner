import { useMemo, type CSSProperties } from "react";
import { Link } from "react-router-dom";
import { Shell } from "../components/Shell";
import {
  IconAlert,
  IconArrowRight,
  IconLibrary,
  IconRecord,
  IconTimeline,
} from "../components/Icons";
import { BandBadge, ErrorState, Skeleton } from "../components/ui";
import { api, useAsync } from "../lib/api";
import { duration } from "../lib/format";
import type {
  BandStatus,
  MethodKey,
  ParticipantCard,
  TrialCard,
} from "../lib/types";
import "../styles/dashboard.css";

const STATUS_ORDER: BandStatus[] = ["critical", "serious", "warning", "good"];
const STATUS_RANK: Record<BandStatus, number> = {
  good: 1,
  warning: 2,
  serious: 3,
  critical: 4,
};

const STATUS_LABEL: Record<BandStatus, string> = {
  good: "Good",
  warning: "Warning",
  serious: "Serious",
  critical: "Critical",
};

interface ReviewItem {
  participant: ParticipantCard;
  trial: TrialCard;
  method: MethodKey | null;
  status: BandStatus | null;
  flagged: boolean;
  recordedAt: number;
}

function parsedDate(value: string | null): number {
  if (!value) return 0;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

/**
 * Pick the method with the stronger published action-band category, never by
 * comparing its numeric score. Equal categories remain a deliberate choice.
 */
function reviewMethod(trial: TrialCard): MethodKey | null {
  const reba = trial.reba?.band?.status;
  const rula = trial.rula?.band?.status;
  if (reba && !rula) return "reba";
  if (rula && !reba) return "rula";
  if (!reba || !rula || STATUS_RANK[reba] === STATUS_RANK[rula]) return null;
  return STATUS_RANK[reba] > STATUS_RANK[rula] ? "reba" : "rula";
}

function strongestStatus(trial: TrialCard): BandStatus | null {
  const statuses = [trial.reba?.band?.status, trial.rula?.band?.status].filter(
    (status): status is BandStatus => Boolean(status),
  );
  return statuses.reduce<BandStatus | null>(
    (strongest, status) =>
      strongest === null || STATUS_RANK[status] > STATUS_RANK[strongest]
        ? status
        : strongest,
    null,
  );
}

function DashboardSkeleton() {
  return (
    <div className="dashboard__loading" aria-label="Loading research overview">
      <div className="dashboard__metric-grid">
        {[0, 1, 2, 3].map((item) => (
          <Skeleton key={item} height={92} radius="var(--r-lg)" />
        ))}
      </div>
      <div className="dashboard__analysis-grid">
        <Skeleton height={278} radius="var(--r-lg)" />
        <Skeleton height={278} radius="var(--r-lg)" />
      </div>
      <Skeleton height={260} radius="var(--r-lg)" />
    </div>
  );
}

export default function Landing() {
  const { data, error, loading, reload } = useAsync(
    (signal) => api.participants("", signal),
    [],
  );
  const participants = data?.participants ?? [];

  const summary = useMemo(() => {
    const trials = participants.flatMap((participant) =>
      participant.trials.map((trial) => ({ participant, trial })),
    );
    const analysed = trials.filter(
      ({ trial }) => trial.status === "analysed" && Boolean(trial.sessionId),
    );
    const distribution: Record<BandStatus, number> = {
      good: 0,
      warning: 0,
      serious: 0,
      critical: 0,
    };
    let unclassified = 0;
    let rebaAvailable = 0;
    let rulaAvailable = 0;

    const reviewItems: ReviewItem[] = analysed.map(({ participant, trial }) => {
      const status = strongestStatus(trial);
      if (status) distribution[status] += 1;
      else unclassified += 1;
      if (trial.reba?.band) rebaAvailable += 1;
      if (trial.rula?.band) rulaAvailable += 1;
      return {
        participant,
        trial,
        method: reviewMethod(trial),
        status,
        flagged: status === "serious" || status === "critical",
        recordedAt: parsedDate(trial.recordedAt ?? participant.recordedAt),
      };
    });

    reviewItems.sort(
      (a, b) =>
        Number(b.flagged) - Number(a.flagged) ||
        (b.status ? STATUS_RANK[b.status] : 0) -
          (a.status ? STATUS_RANK[a.status] : 0) ||
        b.recordedAt - a.recordedAt,
    );

    return {
      recordings: trials.length,
      analysed: analysed.length,
      coverage: trials.length ? Math.round((analysed.length / trials.length) * 100) : 0,
      flagged: reviewItems.filter((item) => item.flagged).length,
      distribution,
      unclassified,
      rebaAvailable,
      rulaAvailable,
      reviewItems: reviewItems.slice(0, 6),
    };
  }, [participants]);

  const riskTotal = STATUS_ORDER.reduce(
    (total, status) => total + summary.distribution[status],
    0,
  );

  return (
    <Shell crumbs={[{ label: "Home", to: "/" }, { label: "Study overview" }]}>
      <div className="dashboard">
        <header className="dashboard__hero">
          <div className="stack stack-2">
            <p className="eyebrow">Research overview</p>
            <h1>Study overview</h1>
            <p className="lede">
              Triage recorded tasks, investigate exposure patterns, and trace every finding back
              to its source frame.
            </p>
          </div>
          <div className="dashboard__actions" aria-label="Workspace actions">
            <Link to="/library" className="btn btn--primary">
              <IconLibrary size={15} />
              Open assessments
            </Link>
            <Link to="/record" className="btn">
              <IconRecord size={15} />
              Record live
            </Link>
          </div>
        </header>

        {error ? (
          <ErrorState message={error} onRetry={reload} />
        ) : loading && !data ? (
          <DashboardSkeleton />
        ) : (
          <>
            <section className="dashboard__metric-grid" aria-label="Portfolio summary">
              <article className="dashboard__metric">
                <span className="dashboard__metric-label">Participants</span>
                <strong className="dashboard__metric-value num">{participants.length}</strong>
                <span className="dashboard__metric-note">in the current library</span>
              </article>
              <article className="dashboard__metric">
                <span className="dashboard__metric-label">Recordings</span>
                <strong className="dashboard__metric-value num">{summary.recordings}</strong>
                <span className="dashboard__metric-note">across all participants</span>
              </article>
              <article className="dashboard__metric">
                <span className="dashboard__metric-label">Analysis ready</span>
                <strong className="dashboard__metric-value num">{summary.analysed}</strong>
                <span className="dashboard__metric-note">{summary.coverage}% of recordings</span>
              </article>
              <article className="dashboard__metric dashboard__metric--priority">
                <span className="dashboard__metric-icon" aria-hidden="true">
                  <IconAlert size={15} />
                </span>
                <span className="dashboard__metric-label">Priority review</span>
                <strong className="dashboard__metric-value num">{summary.flagged}</strong>
                <span className="dashboard__metric-note">
                  serious or critical in either method
                </span>
              </article>
            </section>

            <section className="dashboard__analysis-grid" aria-label="Portfolio analysis">
              <article className="dashboard__panel dashboard__coverage">
                <div className="dashboard__panel-head">
                  <div>
                    <p className="dashboard__kicker">Analysis coverage</p>
                    <h2>Review readiness</h2>
                  </div>
                  <IconTimeline size={19} aria-hidden="true" />
                </div>

                <div className="dashboard__coverage-body">
                  <div
                    className="dashboard__ring"
                    role="img"
                    aria-label={`${summary.coverage}% of recordings have completed analysis`}
                  >
                    <svg viewBox="0 0 120 120" aria-hidden="true">
                      <circle className="dashboard__ring-track" cx="60" cy="60" r="49" />
                      <circle
                        className="dashboard__ring-value"
                        cx="60"
                        cy="60"
                        r="49"
                        pathLength="100"
                        strokeDasharray={`${summary.coverage} ${100 - summary.coverage}`}
                      />
                    </svg>
                    <span className="dashboard__ring-label">
                      <strong className="num">{summary.coverage}%</strong>
                      <span>analysed</span>
                    </span>
                  </div>

                  <dl className="dashboard__method-coverage">
                    <div>
                      <dt>REBA available</dt>
                      <dd className="num">{summary.rebaAvailable}</dd>
                    </div>
                    <div>
                      <dt>RULA available</dt>
                      <dd className="num">{summary.rulaAvailable}</dd>
                    </div>
                    <div>
                      <dt>Awaiting analysis</dt>
                      <dd className="num">
                        {Math.max(0, summary.recordings - summary.analysed)}
                      </dd>
                    </div>
                  </dl>
                </div>
              </article>

              <article className="dashboard__panel">
                <div className="dashboard__panel-head">
                  <div>
                    <p className="dashboard__kicker">Action-band distribution</p>
                    <h2>Portfolio risk signals</h2>
                  </div>
                  <span className="badge badge--neutral">{riskTotal} classified</span>
                </div>

                <div className="dashboard__risk" role="list">
                  {STATUS_ORDER.map((status) => {
                    const count = summary.distribution[status];
                    const width = riskTotal ? (count / riskTotal) * 100 : 0;
                    return (
                      <div className="dashboard__risk-row" role="listitem" key={status}>
                        <span className="dashboard__risk-label">
                          <span className="dashboard__risk-dot" data-status={status} />
                          {STATUS_LABEL[status]}
                        </span>
                        <span
                          className="dashboard__risk-track"
                          role="img"
                          aria-label={`${count} recordings in the ${STATUS_LABEL[status].toLowerCase()} category`}
                        >
                          <span
                            className="dashboard__risk-fill"
                            data-status={status}
                            style={{ "--dashboard-risk-width": `${width}%` } as CSSProperties}
                          />
                        </span>
                        <strong className="dashboard__risk-count num">{count}</strong>
                      </div>
                    );
                  })}
                </div>

                <p className="dashboard__method-note">
                  Each recording is placed in the strongest action-band category assigned by either
                  available method. Raw RULA and REBA scores are never compared.
                  {summary.unclassified
                    ? ` ${summary.unclassified} analysed recording${
                        summary.unclassified === 1 ? "" : "s"
                      } had no valid action band.`
                    : ""}
                </p>
              </article>
            </section>

            <section className="dashboard__panel dashboard__queue" aria-labelledby="review-queue">
              <div className="dashboard__panel-head dashboard__queue-head">
                <div>
                  <p className="dashboard__kicker">Triage</p>
                  <h2 id="review-queue">Review queue</h2>
                  <p className="small muted">
                    Serious and critical action bands first, followed by the newest analysed
                    recordings.
                  </p>
                </div>
                <Link to="/library" className="btn btn--ghost">
                  Browse all studies
                  <IconArrowRight size={14} />
                </Link>
              </div>

              {summary.reviewItems.length ? (
                <div className="dashboard__recordings">
                  {summary.reviewItems.map((item, index) => {
                    const sessionId = item.trial.sessionId as string;
                    const destination = item.method
                      ? `/s/${sessionId}/${item.method}`
                      : `/s/${sessionId}`;
                    const action = item.method
                      ? `Review ${item.method.toUpperCase()}`
                      : "Choose analysis";
                    return (
                      <Link
                        className="dashboard__recording"
                        data-priority={item.flagged}
                        data-status={item.status ?? "none"}
                        to={destination}
                        key={`${item.participant.key}-${item.trial.folder}`}
                        style={{ "--dashboard-row-order": index } as CSSProperties}
                        aria-label={`${action} for ${item.trial.name}, participant ${item.participant.participantId}`}
                      >
                        <span className="dashboard__recording-marker" aria-hidden="true" />
                        <span className="dashboard__recording-main">
                          <span className="dashboard__recording-context">
                            Participant {item.participant.participantId}
                            <span aria-hidden="true"> · </span>
                            {item.trial.recordedAt ?? item.participant.recordedAt ?? "date unknown"}
                          </span>
                          <strong>{item.trial.name}</strong>
                          <span className="dashboard__recording-detail">
                            {duration(item.trial.durationSec)}
                            <span aria-hidden="true"> · </span>
                            {item.trial.cameras ?? "?"} camera
                            {item.trial.cameras === 1 ? "" : "s"}
                          </span>
                        </span>

                        <span
                          className="dashboard__recording-bands"
                          aria-label="Available method action bands"
                        >
                          <span>
                            <span className="dashboard__method-label">REBA</span>
                            <BandBadge band={item.trial.reba?.band} size="sm" />
                          </span>
                          <span>
                            <span className="dashboard__method-label">RULA</span>
                            <BandBadge band={item.trial.rula?.band} size="sm" />
                          </span>
                        </span>

                        <span className="dashboard__recording-action">
                          {item.flagged ? (
                            <span className="dashboard__priority-label">Priority</span>
                          ) : null}
                          <span>
                            {action}
                            <IconArrowRight size={14} />
                          </span>
                        </span>
                      </Link>
                    );
                  })}
                </div>
              ) : (
                <div className="dashboard__empty">
                  <span className="dashboard__empty-icon" aria-hidden="true">
                    <IconLibrary size={20} />
                  </span>
                  <div>
                    <h3>No analysed recordings yet</h3>
                    <p className="small muted">
                      Completed analyses will appear here with their action-band status and a direct
                      route to review.
                    </p>
                  </div>
                  <Link to="/library" className="btn">
                    Browse studies
                  </Link>
                </div>
              )}
            </section>
          </>
        )}
      </div>
    </Shell>
  );
}
