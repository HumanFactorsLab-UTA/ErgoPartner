import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import {
  ContributionComparison,
  EvidenceSummary,
  PersistenceMatrix,
  RiskDistribution,
  ScoreProfile,
} from "../components/AnalysisVisuals";
import { Assistant } from "../components/Assistant";
import { AssessmentReview } from "../components/AssessmentReview";
import { FrameDataExplorer } from "../components/FrameDataExplorer";
import { FocusedReview } from "../components/FocusedReview";
import { FramePanel } from "../components/FramePanel";
import { IconArrowRight, IconInfo } from "../components/Icons";
import { PostureReplay } from "../components/PostureReplay";
import { Shell } from "../components/Shell";
import { TabPanel, WorkTabs, isTabKey, type TabKey } from "../components/WorkTabs";
import { Timeline } from "../components/Timeline";
import { VideoPanel } from "../components/VideoPanel";
import {
  BandBadge,
  ConfirmButton,
  Disclosure,
  ErrorState,
  Segmented,
  Skeleton,
  Stat,
} from "../components/ui";
import { api, useAsync, useDebounced } from "../lib/api";
import {
  clock,
  humanise,
  number,
  percent,
  statusInk,
  statusSoft,
  statusVar,
} from "../lib/format";
import type {
  AnalysisJob,
  AssistantCitation,
  BodyLoad,
  Episode,
  FrameDetail,
  MethodKey,
  SeriesPoint,
} from "../lib/types";
import "../styles/workspace.css";
import "../styles/assistant.css";

function SectionHeading({
  eyebrow,
  title,
  note,
}: {
  eyebrow: string;
  title: string;
  note: string;
}) {
  return (
    <div className="workflow-head">
      <div>
        <p className="workflow-head__eyebrow">{eyebrow}</p>
        <h2>{title}</h2>
      </div>
      <p className="workflow-head__note">{note}</p>
    </div>
  );
}

function ContributionBars({
  loads,
  selectedRegion,
  onPick,
}: {
  loads: BodyLoad[];
  selectedRegion: string;
  onPick: (load: BodyLoad) => void;
}) {
  const top = Math.max(...loads.map((load) => load.meanPoints), 0.0001);
  return (
    <div className="bars">
      {loads.map((load, index) => (
        <button
          key={load.region}
          type="button"
          className="bar"
          data-selected={selectedRegion === load.region}
          onClick={() => onPick(load)}
          disabled={load.peakFrame === null}
          title={`Worst at ${clock(load.peakSec)} · frame ${load.peakFrame ?? "Unavailable"}`}
        >
          <span className="bar__label">{load.label}</span>
          <span className="bar__track">
            <span
              className="bar__fill"
              style={{
                width: `${(load.meanPoints / top) * 100}%`,
                background: index === 0 ? "var(--series-2)" : "var(--series-1)",
                opacity: index === 0 ? 1 : Math.max(0.48, 1 - index * 0.08),
              }}
            />
          </span>
          <span className="bar__value">+{number(load.meanPoints, 1)}</span>
        </button>
      ))}
    </div>
  );
}

function EventRow({
  episode,
  selected,
  maximumDuration,
  onPick,
}: {
  episode: Episode;
  selected: boolean;
  maximumDuration: number;
  onPick: (frame: number) => void;
}) {
  return (
    <button
      type="button"
      className="event"
      data-selected={selected}
      onClick={() => onPick(episode.peakFrame)}
      aria-label={`Episode ${episode.index}, ${episode.timeRange}, peak score ${episode.peakScore}`}
    >
      <span className="event__rank">{episode.index}</span>
      <span className="event__main">
        <span className="event__time">
          <strong className="num">{episode.timeRange}</strong>
          <span className="muted num">{number(episode.durationSec, 1)}s</span>
        </span>
        <span className="event__duration" aria-hidden="true">
          <span style={{ width: `${(episode.durationSec / maximumDuration) * 100}%` }} />
        </span>
        <span className="event__drivers">
          {episode.drivers.slice(0, 3).map((driver) => (
            <span key={driver.region}>
              {driver.label} +{number(driver.pointsAdded, 1)}
            </span>
          ))}
          {!episode.drivers.length ? <span>no component evidence</span> : null}
        </span>
      </span>
      <span className="event__risk">
        <span
          className="event__score num"
          style={{ color: statusInk(episode.band?.status) }}
        >
          {episode.peakScore}
        </span>
        <BandBadge band={episode.band} size="sm" />
        <IconArrowRight size={14} />
      </span>
    </button>
  );
}

function SelectedMomentCard({
  detail,
  loading,
  typicalScore,
  episode,
  onInspect,
}: {
  detail: FrameDetail | null;
  loading: boolean;
  typicalScore: number | null;
  episode: Episode | null;
  onInspect: () => void;
}) {
  if (!detail) {
    return (
      <aside className="moment-card">
        <Skeleton height={16} width={120} />
        <Skeleton height={42} width={180} />
        <Skeleton height={120} radius="var(--r-md)" />
      </aside>
    );
  }
  const delta =
    detail.score !== null && typicalScore !== null ? detail.score - typicalScore : null;
  const drivers = detail.drivers.filter((driver) => driver.pointsAdded > 0).slice(0, 4);

  return (
    <aside className="moment-card" data-updating={loading}>
      <div className="moment-card__content" key={detail.frame}>
        <div className="moment-card__head">
          <div>
            <p className="insight-card__kicker">Selected moment</p>
            <h3 className="num">{detail.clock}</h3>
          </div>
          <BandBadge band={detail.band} />
        </div>

        <div className="moment-score">
          <div>
            <span>{detail.method} now</span>
            <strong
              className="num"
              style={{ color: detail.band ? statusInk(detail.band.status) : undefined }}
            >
              {detail.score ?? "Unavailable"}
            </strong>
          </div>
          <span className="moment-score__arrow" aria-hidden="true">
            →
          </span>
          <div>
            <span>Task typical</span>
            <strong className="num">{number(typicalScore, 1)}</strong>
          </div>
          {delta !== null ? (
            <span className="moment-score__delta num" data-positive={delta > 0}>
              {delta > 0 ? "+" : ""}
              {number(delta, 1)}
            </span>
          ) : null}
        </div>

        {episode ? (
          <p className="moment-card__episode">
            Inside episode {episode.index} · {episode.timeRange} · {number(episode.durationSec, 1)}s
          </p>
        ) : (
          <p className="moment-card__episode">Outside a sustained high-risk episode</p>
        )}

        <div className="moment-card__drivers">
          <p className="micro muted">Largest score drivers at this moment</p>
          {drivers.length ? (
            drivers.map((driver) => (
              <div key={driver.region}>
                <span>{driver.label}</span>
                <span className="moment-card__driverbar" aria-hidden="true">
                  <span
                    style={{
                      width: `${Math.max(8, (driver.score / Math.max(1, driver.maximum)) * 100)}%`,
                    }}
                  />
                </span>
                <strong className="num">+{number(driver.pointsAdded, 1)}</strong>
              </div>
            ))
          ) : (
            <span className="small muted">No contributing component evidence was reported.</span>
          )}
        </div>

        <div className="moment-card__quality">
          <span>
            Evidence <strong>{humanise(detail.quality.status)}</strong>
          </span>
          <span>
            Visibility <strong className="num">{number(detail.quality.meanVisibility, 2)}</strong>
          </span>
          {detail.quality.warnings.length ? (
            <span className="moment-card__warning">
              {detail.quality.warnings.length} warning
              {detail.quality.warnings.length === 1 ? "" : "s"}
            </span>
          ) : null}
        </div>

        <button type="button" className="btn btn--primary btn--block" onClick={onInspect}>
          Inspect posture evidence
          <IconArrowRight size={14} />
        </button>
      </div>
    </aside>
  );
}

export default function Workspace() {
  const { sessionId } = useParams();
  return <WorkspaceSession key={sessionId} />;
}

function WorkspaceSession() {
  const { sessionId = "", method = "reba" } = useParams<{
    sessionId: string;
    method: MethodKey;
  }>();
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const methodKey = (method === "rula" ? "rula" : "reba") as MethodKey;
  useEffect(() => { try { localStorage.setItem(`posture-method:${sessionId}`, methodKey); } catch { /* Storage may be disabled. */ } }, [sessionId, methodKey]);
  const requestedSecondParam = searchParams.get("t");
  const requestedSecond =
    requestedSecondParam === null ? null : Number(requestedSecondParam);

  const session = useAsync((signal) => api.session(sessionId, signal), [sessionId]);
  const [person, setPerson] = useState<string | null>(null);
  const [camera, setCamera] = useState(0);
  const pendingCitation = useRef<AssistantCitation | null>(null);
  const timeline = useAsync(
    (signal) => api.timeline(sessionId, methodKey, signal, person),
    [sessionId, methodKey, person],
  );

  const [frame, setFrame] = useState<number | null>(null);
  const [region, setRegion] = useState("");
  const [episodeLimit, setEpisodeLimit] = useState(5);
  const [evidenceRequest, setEvidenceRequest] = useState<{
    frame: number; camera?: number | null; overlay?: boolean; person?: string | null; method?: string | null;
  } | null>(null);
  const settledFrame = useDebounced(frame, 90);

  // The tab lives in the URL so a reload, a back button, or a link pasted into
  // a colleague's message all land on the same screen.
  const tabParam = searchParams.get("tab");
  const tab: TabKey = tabParam === "drivers" ? "investigate" : isTabKey(tabParam) ? tabParam : "overview";
  const [visited, setVisited] = useState<Set<TabKey>>(() => new Set<TabKey>([tab]));
  const [assistantOpen, setAssistantOpen] = useState(false);
  const visualReviewOpen = tab === "investigate" && searchParams.get("review") === "visual";

  function openVisualReview() {
    setAssistantOpen(false);
    const params = new URLSearchParams(searchParams);
    params.set("tab", "investigate");
    params.set("review", "visual");
    navigate(`/s/${sessionId}/${methodKey}?${params}`);
    if (visualReviewOpen) document.getElementById("focused-review")?.scrollIntoView({block:"start"});
  }
  useEffect(() => {
    if (!visualReviewOpen || !visited.has("investigate")) return;
    const task = requestAnimationFrame(() => {
      const section = document.getElementById("focused-review");
      section?.focus({preventScroll: true});
      section?.scrollIntoView({block:"start"});
    });
    return () => cancelAnimationFrame(task);
  }, [visualReviewOpen, visited]);

  useEffect(() => {
    setVisited((current) => (current.has(tab) ? current : new Set(current).add(tab)));
  }, [tab]);

  const setTab = useCallback(
    (next: TabKey, options: { replace?: boolean } = {}) => {
      const params = new URLSearchParams(searchParams);
      params.delete("review");
      if (next === "overview") params.delete("tab");
      else params.set("tab", next);
      const query = params.toString();
      navigate(`/s/${sessionId}/${methodKey}${query ? `?${query}` : ""}`, {
        replace: options.replace ?? false,
      });
      window.scrollTo({ top: 0, behavior: "smooth" });
    },
    [navigate, sessionId, methodKey, searchParams],
  );

  useEffect(() => {
    if (
      frame !== null ||
      !timeline.data?.series.points.length ||
      timeline.data.method.toLowerCase() !== methodKey
    ) {
      return;
    }
    if (requestedSecond !== null && Number.isFinite(requestedSecond)) {
      let nearest = timeline.data.series.points[0];
      for (const point of timeline.data.series.points) {
        if (Math.abs(point.t - requestedSecond) < Math.abs(nearest.t - requestedSecond)) {
          nearest = point;
        }
      }
      setFrame(nearest.f);
      return;
    }
    if (timeline.data.verdict.peakFrame !== null) {
      setFrame(timeline.data.verdict.peakFrame);
    }
  }, [timeline.data, frame, requestedSecond, methodKey]);

  const showCitation = useCallback((citation: AssistantCitation) => {
    if (citation.frame === null) return;
    setFrame(citation.frame);
    setRegion(citation.region ?? "");
    if (citation.camera != null) setCamera(citation.camera);
    setEvidenceRequest({
      frame: citation.frame, camera: citation.camera, method: citation.method, person: citation.person,
      overlay: citation.overlay ?? (citation.kind !== "observed" || Boolean(citation.view?.includes("overlay"))),
    });
  }, []);
  useEffect(() => {
    const citation = pendingCitation.current;
    if (citation && (!citation.method || citation.method === methodKey) && (!citation.person || citation.person === person)) {
      pendingCitation.current = null;
      showCitation(citation);
    } else {
      setFrame(null);
      setRegion("");
      setEvidenceRequest(null);
    }
    setEpisodeLimit(5);
  }, [methodKey, person, showCitation]);

  const detail = useAsync(
    (signal) =>
      settledFrame === null
        ? Promise.reject(new Error("No frame selected"))
        : api.frame(
            sessionId,
            settledFrame,
            methodKey,
            person ?? timeline.data?.person ?? null,
            signal,
          ),
    [sessionId, settledFrame, methodKey, person, timeline.data?.person],
    { keepPrevious: true },
  );

  const points = timeline.data?.series.points ?? [];
  const selectedIndex = useMemo(
    () => points.findIndex((point) => point.f === frame),
    [points, frame],
  );
  const selectedSec = useMemo(() => {
    if (!timeline.data || frame === null) return null;
    const point = timeline.data.series.points.find((candidate) => candidate.f === frame);
    return point ? point.t : (detail.data?.frame === frame ? detail.data.timestampSec : null);
  }, [timeline.data, frame, detail.data]);
  const selectedEpisode = useMemo(() => {
    if (!timeline.data || frame === null) return null;
    return (
      timeline.data.episodes.find(
        (episode) => frame >= episode.startFrame && frame <= episode.endFrame,
      ) ?? null
    );
  }, [timeline.data, frame]);
  const roles = useMemo(() => {
    const map: Record<number, string> = {};
    for (const camera of detail.data?.cameras ?? []) map[camera.cameraIndex] = camera.role;
    return map;
  }, [detail.data]);

  /**
   * The one way to say "show me this moment".
   *
   * Every route into the evidence -- a ranked episode, a body region's worst
   * frame, a row of the frame table, a citation in the assistant -- ends here,
   * and every one of them has to land on the tab that actually shows a frame.
   * Before the tabs existed this was a `scrollIntoView`; the failure mode now is
   * worse than an awkward scroll position, because the result would render on a
   * screen the reviewer is not looking at.
   */
  const focusFrame = useCallback(
    (next: number, options: { region?: string | null; compare?: boolean } = {}) => {
      if (options.region !== undefined) setRegion(options.region ?? "");
      setFrame(next);
      if (options.compare) setEvidenceRequest({frame: next});
      if (tab !== "investigate") setTab("investigate");
      else {
        window.setTimeout(() => {
          document.getElementById("frame")?.scrollIntoView({ behavior: "smooth", block: "start" });
        }, 40);
      }
    },
    [tab, setTab],
  );

  const pick = useCallback((next: number) => focusFrame(next, { compare: true }), [focusFrame]);
  const pickPoint = useCallback(
    (point: SeriesPoint) => {
      // Selecting a low-risk point is browsing; selecting a serious one is
      // usually the start of an inspection, so the comparison opens with it.
      setFrame(point.f);
      if (point.b === "serious" || point.b === "critical") setEvidenceRequest({frame: point.f});
    },
    [],
  );
  const inspectSelected = useCallback(() => {
    if (frame !== null) setEvidenceRequest({frame});
  }, [frame]);
  const inspectFrame = useCallback(
    (next: number) => focusFrame(next, { compare: true }),
    [focusFrame],
  );
  const openCitation = (citation: AssistantCitation) => {
    if (citation.frame === null) return;
    const targetMethod = citation.method ?? methodKey;
    if (targetMethod !== methodKey || (citation.person && citation.person !== person)) {
      pendingCitation.current = citation;
      if (citation.person) setPerson(citation.person);
    } else {
      showCitation(citation);
    }
    navigate(`/s/${sessionId}/${targetMethod}?tab=investigate`);
  };
  const pickLoad = useCallback(
    (load: BodyLoad) => {
      if (load.peakFrame === null) return;
      focusFrame(load.peakFrame, { region: load.region });
    },
    [focusFrame],
  );
  const pickTaskLoad = useCallback(
    (load: BodyLoad) => {
      if (load.peakFrame === null) return;
      focusFrame(load.peakFrame, { region: null });
    },
    [focusFrame],
  );
  const step = useCallback(
    (direction: -1 | 1) => {
      if (!points.length) return;
      const base = selectedIndex < 0 ? 0 : selectedIndex;
      const next = Math.min(points.length - 1, Math.max(0, base + direction));
      setFrame(points[next].f);
    },
    [points, selectedIndex],
  );
  const seekToSecond = useCallback(
    (seconds: number) => {
      if (!points.length) return;
      let best = points[0];
      for (const point of points) {
        if (Math.abs(point.t - seconds) < Math.abs(best.t - seconds)) best = point;
      }
      setFrame(best.f);
    },
    [points],
  );
  /**
   * Re-run this one recording.
   *
   * It lives beside the reproducibility record rather than in the toolbar
   * because that is where the question arises: you are reading what produced
   * this result and deciding whether it should be produced again. It is also
   * what makes an accepted condition from the assistant mean anything -- until
   * a run picks it up, "recorded for the next analysis run" has no next run.
   */
  const [rerun, setRerun] = useState<{ job: AnalysisJob | null; error: string | null }>({
    job: null,
    error: null,
  });

  const startRerun = useCallback(async () => {
    const participantKey = session.data?.participant.key;
    const taskFolder = session.data?.trial.folder;
    if (!participantKey || !taskFolder) return;
    setRerun({ job: null, error: null });
    try {
      const job = await api.startAnalysis("trial", {
        participantKey,
        taskFolder,
        // Re-run at the sampling this result was produced at, not the default.
        // Publishing to a different rate would leave the old result in place and
        // create a second one beside it, which is not what "re-run" means.
        samplingFps: session.data?.sampling ?? undefined,
        rerun: true,
      });
      setRerun({ job, error: null });
    } catch (error) {
      setRerun({
        job: null,
        error: error instanceof Error ? error.message : "The analysis could not start",
      });
    }
  }, [session.data]);

  // Poll while it runs, then reload the analysis in place so the workspace shows
  // the new result without the reviewer navigating anywhere.
  useEffect(() => {
    const job = rerun.job;
    if (!job || job.done) return;
    let live = true;
    const timer = window.setInterval(async () => {
      try {
        const next = await api.analysisJob(job.jobId);
        if (!live) return;
        setRerun((current) => ({ ...current, job: next }));
        if (next.done) {
          window.clearInterval(timer);
          timeline.reload();
          session.reload();
        }
      } catch {
        /* a dropped poll is not a failed run; the next tick retries */
      }
    }, 1500);
    return () => {
      live = false;
      window.clearInterval(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rerun.job?.jobId, rerun.job?.done]);

  const changeMethod = useCallback(
    (next: MethodKey) => {
      // Switching method keeps both the moment and the tab: comparing RULA
      // against REBA at the same instant is the point of having both.
      const params = new URLSearchParams();
      if (selectedSec !== null) params.set("t", selectedSec.toFixed(3));
      if (tab !== "overview") params.set("tab", tab);
      const query = params.toString();
      navigate(`/s/${sessionId}/${next}${query ? `?${query}` : ""}`);
    },
    [navigate, selectedSec, sessionId, tab],
  );

  const crumbs = [
    { label: "Home", to: "/" },
    { label: "Assessments", to: "/assessments" },
    {
      label: session.data ? `Participant ${session.data.participant.participantId}` : "Recording",
      to: `/s/${sessionId}`,
    },
    { label: `${methodKey.toUpperCase()} analysis` },
  ];

  if (timeline.error || session.error) {
    return (
      <Shell crumbs={crumbs}>
        <ErrorState
          message={timeline.error ?? session.error ?? "Unknown error"}
          onRetry={() => {
            timeline.reload();
            session.reload();
          }}
        />
        <p style={{ marginTop: "var(--s-4)" }}>
          <Link to="/library">Back to the library</Link>
        </p>
      </Shell>
    );
  }

  const data = timeline.data;
  const overview = session.data;
  const cameraCount = overview?.cameraCount ?? overview?.cameras.length ?? 1;
  const maximumEpisodeDuration = Math.max(
    ...(data?.episodes.map((episode) => episode.durationSec) ?? []),
    1,
  );

  return (
    <Shell
      crumbs={crumbs}
      session={overview}
      method={methodKey}
      wide
      rail={
        assistantOpen ? (
          <Assistant
            sessionId={sessionId}
            method={methodKey}
            selection={{method: methodKey, frame, camera, region: region || null, person: person ?? data?.person ?? null, timestampSec: selectedSec}}
            cameraLabel={index => cameraCount > 1 ? humanise(roles[index] ?? `Camera ${index}`) : "Camera"}
            onCitation={openCitation}
            onFocusedReview={openVisualReview}
            layout="rail"
            onClose={() => setAssistantOpen(false)}
          />
        ) : null
      }
      actions={
        <div className="row row-3">
          {overview && overview.people.length > 1 && <label className="small">Worker{" "}
            <select value={person ?? data?.person ?? overview.people[0]} onChange={event => setPerson(event.target.value)}>
              {overview.people.map(id => <option key={id} value={id}>{id}</option>)}
            </select>
          </label>}
          <Segmented
            label="Assessment method"
            value={methodKey}
            onChange={changeMethod}
            options={[
              { value: "reba", label: "REBA" },
              { value: "rula", label: "RULA" },
            ]}
          />
          <button
            type="button"
            className={`btn ${assistantOpen ? "btn--primary" : "btn--ghost"}`}
            onClick={() => setAssistantOpen((open) => !open)}
            aria-pressed={assistantOpen}
            title="Ask about this assessment"
          >
            <IconInfo size={14} />
            Assistant
          </button>
        </div>
      }
    >
      <WorkTabs
        value={tab}
        onChange={setTab}
        counts={{
          investigate: data?.episodes.length ? data.episodes.length : undefined,
        }}
      />

      <div className="ws">
        <TabPanel tab="overview" active={tab} visited={visited.has("overview")}>
        <section className="workflow-section" id="overview">
          <h2>Assessment summary</h2>

          {!data ? (
            <Skeleton height={260} radius="var(--r-xl)" />
          ) : (
            <div
              className="verdict"
              style={
                {
                  "--verdict-accent": statusVar(data.verdict.peakBand?.status),
                  "--verdict-soft": statusSoft(data.verdict.peakBand?.status),
                } as React.CSSProperties
              }
            >
              <div className="verdict__narrative">
                <div className="verdict__context">
                  <span>{overview?.trial.name ?? "Recording"}</span>
                  <span>Participant {overview?.participant.participantId ?? "Unavailable"}</span>
                  <span>{data.method}</span>
                </div>
                <BandBadge band={data.verdict.peakBand} />
                <h1 className="verdict__headline">{data.verdict.headline}</h1>
                <p className="verdict__detail">
                  {data.verdict.detail.charAt(0).toUpperCase() + data.verdict.detail.slice(1)}
                </p>
                <button
                  type="button"
                  className="btn btn--primary"
                  onClick={() => {
                    if (data.verdict.peakFrame !== null) inspectFrame(data.verdict.peakFrame);
                    else setTab("investigate");
                  }}
                >
                  {data.verdict.peakFrame === null ? "Open posture review" : `Review peak at ${data.verdict.peakClock}`}
                  <IconArrowRight size={14} />
                </button>
                <button type="button" className="btn btn--ghost" onClick={() => setTab("assurance")}>
                  Evidence and calculation details
                </button>
              </div>

              <div className="verdict__metrics">
                <Stat
                  label={`Worst ${data.method}`}
                  value={data.verdict.peakScore ?? "Unavailable"}
                  status={data.verdict.peakBand?.status}
                  hint={data.verdict.peakBand?.label}
                  emphasis
                />
                <Stat
                  label="Task typical"
                  value={number(data.verdict.typicalScore, 1)}
                  hint="duration weighted"
                  emphasis
                />
                <Stat
                  label={`Time at ${data.actionThreshold}+`}
                  value={data.verdict.scored ? percent(data.verdict.percentAtAction) : "Unavailable"}
                  hint={`${number(data.verdict.timeAtActionSec, 0)}s of ${number(
                    data.verdict.scoredDurationSec,
                    0,
                  )}s`}
                  emphasis
                />
                <Stat
                  label="Sustained episodes"
                  value={data.verdict.scored ? data.verdict.episodeCount : "Unavailable"}
                  hint={data.verdict.scored ? "at or above the action threshold" : "No scored footage"}
                  emphasis
                />
              </div>
            </div>
          )}

          {data && overview ? (
            <div className="insight-grid">
              <RiskDistribution
                shares={data.shares}
                percentAtAction={data.verdict.percentAtAction}
                timeAtActionSec={data.verdict.timeAtActionSec}
              />
              <ScoreProfile
                points={data.series.points}
                bands={data.bands}
                typicalScore={data.verdict.typicalScore}
                peakScore={data.verdict.peakScore}
                scaleMaximum={data.scaleMaximum}
                actionThreshold={data.actionThreshold}
                method={data.method}
              />
              <EvidenceSummary session={overview} />
            </div>
          ) : null}
        </section>
        </TabPanel>

        <TabPanel tab="investigate" active={tab} visited={visited.has("investigate")}>
        <section className="workflow-section" id="timeline">
          <SectionHeading
            eyebrow="Exposure patterns"
            title="When does risk concentrate?"
            note="Select a moment to inspect the footage and its score."
          />

          {!data ? (
            <Skeleton height={360} radius="var(--r-xl)" />
          ) : (
            <Timeline
              points={data.series.points}
              bands={data.bands}
              episodes={data.episodes}
              scaleMaximum={data.scaleMaximum}
              method={data.method}
              selectedFrame={frame}
              onSelect={pickPoint}
              typicalScore={data.verdict.typicalScore}
              actionThreshold={data.actionThreshold}
              height={300}
            />
          )}

          <div className="investigation-grid">
            <div className="episode-panel" id="events">
              <div className="episode-panel__head">
                <div>
                  <p className="insight-card__kicker">Ranked episodes</p>
                  <h3>Sustained high-risk exposure</h3>
                </div>
                <span className="badge badge--neutral">
                  {data ? `${data.episodes.length} found` : "loading"}
                </span>
              </div>

              {!data ? (
                <div className="stack stack-2">
                  <Skeleton height={72} radius="var(--r-md)" />
                  <Skeleton height={72} radius="var(--r-md)" />
                </div>
              ) : data.episodes.length === 0 ? (
                <p className="small muted">
                  No stretch stayed at {data.method} {data.actionThreshold} or above long enough
                  to count as an episode.
                </p>
              ) : (
                <>
                  <div className="events">
                    {data.episodes.slice(0, episodeLimit).map((episode) => (
                      <EventRow
                        key={episode.index}
                        episode={episode}
                        selected={
                          frame !== null &&
                          frame >= episode.startFrame &&
                          frame <= episode.endFrame
                        }
                        maximumDuration={maximumEpisodeDuration}
                        onPick={pick}
                      />
                    ))}
                  </div>
                  {data.episodes.length > 5 ? (
                    <button
                      type="button"
                      className="btn btn--ghost episode-panel__more"
                      onClick={() =>
                        setEpisodeLimit((current) => (current > 5 ? 5 : data.episodes.length))
                      }
                    >
                      {episodeLimit > 5
                        ? "Show priority episodes"
                        : `Show all ${data.episodes.length} episodes`}
                    </button>
                  ) : null}
                </>
              )}
            </div>

            <SelectedMomentCard
              detail={detail.data}
              loading={detail.loading}
              typicalScore={data?.verdict.typicalScore ?? null}
              episode={selectedEpisode}
              onInspect={inspectSelected}
            />
          </div>
        </section>

        <section className="workflow-section" id="frame">
          <SectionHeading
            eyebrow="Posture evidence"
            title="What changed at the selected moment?"
            note="Replay the scored landmarks and inspect the recorded moment."
          />

          <PostureReplay
            sessionId={sessionId}
            method={methodKey}
            cameraCount={cameraCount}
            cameraRoles={roles}
            camera={camera}
            onCameraChange={setCamera}
            person={person ?? data?.person ?? null}
            selectedFrame={frame}
            region={region}
            onSelect={setFrame}
            onInspect={inspectFrame}
          />

          <FramePanel
            sessionId={sessionId}
            detail={detail.data}
            loading={detail.loading}
            region={region}
            person={detail.data?.person ?? person ?? data?.person ?? null}
            typicalScore={data?.verdict.typicalScore}
            onRegionChange={setRegion}
            onPrevious={() => step(-1)}
            onNext={() => step(1)}
            canPrevious={selectedIndex > 0}
            canNext={selectedIndex >= 0 && selectedIndex < points.length - 1}
            multi={cameraCount > 1}
            openEvidence={evidenceRequest}
            onCameraChange={setCamera}
            onComparisonOpened={() => setEvidenceRequest(null)}
          />

          <Disclosure summary="Contributing postures" defaultOpen={tabParam === "drivers"}>
          {!data ? (
            <div className="cause-grid">
              <Skeleton height={420} radius="var(--r-xl)" />
              <Skeleton height={420} radius="var(--r-xl)" />
            </div>
          ) : (
            <div className="stack stack-4">
              <div className="cause-grid">
                <ContributionComparison
                  loads={data.posture}
                  selectedRegion={region}
                  onPick={pickLoad}
                />
                <PersistenceMatrix
                  loads={data.posture}
                  selectedRegion={region}
                  onPick={pickLoad}
                />
              </div>

              {data.taskFactors.length ? (
                <Disclosure
                  summary="Task conditions"
                  hint={`${data.taskFactors.length} non-postural contributors`}
                >
                  <p className="small muted disclosure-intro">
                    Load, coupling, and repetition can change the score without a body part
                    moving. They are kept separate from posture so the intervention target stays
                    clear.
                  </p>
                  <ContributionBars
                    loads={data.taskFactors}
                    selectedRegion={region}
                    onPick={pickTaskLoad}
                  />
                </Disclosure>
              ) : null}

              <Disclosure summary="Body-region research table" hint="scores, exposure, angle, and timing">
                <div className="dt__scroll">
                  <table className="dt">
                    <thead>
                      <tr>
                        <th>Body region</th>
                        <th className="num">Typical impact</th>
                        <th className="num">Peak impact</th>
                        <th className="num">Above neutral</th>
                        <th>Peak at</th>
                        <th>Measurement</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.posture.map((load) => (
                        <tr key={load.region} data-selected={region === load.region}>
                          <td>
                            <button type="button" onClick={() => pickLoad(load)}>
                              {load.label}
                            </button>
                          </td>
                          <td className="num">+{number(load.meanPoints, 2)}</td>
                          <td className="num">+{number(load.worstPoints, 2)}</td>
                          <td className="num">{percent(load.percentRaised)}</td>
                          <td className="num">{clock(load.peakSec)}</td>
                          <td className="muted">
                            {load.measurement && load.peakValueDeg !== null
                              ? `${humanise(load.measurement)} ${number(load.peakValueDeg, 0)}°`
                              : "Unavailable"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Disclosure>
            </div>
          )}
          </Disclosure>

          <div className="video-disclosure" id="video">
            <Disclosure
              summary={cameraCount > 1 ? "Review synchronized source video" : "Review source video"}
              hint={
                selectedSec !== null
                  ? `opens at ${clock(selectedSec)}`
                  : "source evidence"
              }
            >
              {overview && overview.cameras.length ? (
                <VideoPanel
                  sessionId={sessionId}
                  cameras={cameraCount > 1 ? overview.cameras : overview.cameras.slice(0, 1)}
                  camera={camera}
                  onCameraChange={setCamera}
                  roles={roles}
                  selectedSec={selectedSec}
                  onScrubbed={seekToSecond}
                />
              ) : (
                <p className="small muted">No source recording is available for this session.</p>
              )}
            </Disclosure>
          </div>
          {visualReviewOpen ? <FocusedReview
            sessionId={sessionId}
            method={methodKey}
            person={person ?? data?.person ?? null}
            frame={frame}
            camera={camera}
            cameras={(overview?.cameras ?? []).map(item => ({
              index:item.camera_index,
              label:cameraCount === 1 ? "Camera" : humanise(roles[item.camera_index] || `Camera ${item.camera_index + 1}`),
            }))}
            onCamera={setCamera}
            onFrame={setFrame}
          /> : <div className="focused-entry card">
            <div><h3>Focused visual review</h3><p className="small muted">Inspect grip, arm support, and unclear footage at this moment.</p></div>
            <button type="button" className="btn" onClick={openVisualReview}>Open visual review</button>
          </div>}
        </section>
        </TabPanel>

        <TabPanel tab="data" active={tab} visited={visited.has("data")}>
        <section className="workflow-section" id="data">
          <SectionHeading
            eyebrow="Frame calculations"
            title="How did every observation classify?"
            note="Inspect scores, periods above the action threshold, and gaps in the recording."
          />

          <FrameDataExplorer
            sessionId={sessionId}
            method={methodKey}
            person={data?.person ?? null}
            selectedFrame={frame}
            bands={data?.bands ?? []}
            onSelect={inspectFrame}
          />
        </section>
        </TabPanel>

        <TabPanel tab="assurance" active={tab} visited={visited.has("assurance")}>
        <div className="workflow-heading"><p className="small muted">Inspect the measurement quality, assumptions, and calculation evidence.</p><button className="btn" onClick={() => setTab("data")}>Open detailed frame calculations</button></div>
        <section className="workflow-section" id="report">
          <SectionHeading
            eyebrow="Assurance"
            title="Can this result support a decision?"
            note="Review the recorded inputs, measurement quality, and limitations."
          />

          <div className="assurance-grid">
            <article className="assurance-card">
              <p className="insight-card__kicker">Reproducibility</p>
              <h3>Analysis record</h3>
              {overview ? (
                <dl className="assurance-list">
                  <div>
                    <dt>Software</dt>
                    <dd>{overview.softwareVersion ?? "Unavailable"}</dd>
                  </div>
                  <div>
                    <dt>Calibration</dt>
                    <dd>{humanise(overview.calibration)}</dd>
                  </div>
                  <div>
                    <dt>Tracking</dt>
                    <dd>{humanise(overview.trackingMode)}</dd>
                  </div>
                  <div>
                    <dt>Runtime</dt>
                    <dd className="num">{number(overview.runtimeSec, 0)}s</dd>
                  </div>
                </dl>
              ) : (
                <Skeleton height={150} />
              )}

              <div className="rerun">
                {rerun.job && !rerun.job.done ? (
                  <p className="rerun__running">
                    <span className="rerun__spin" aria-hidden="true" />
                    Re-analyzing this recording.{" "}
                    {rerun.job.current
                      ? `${rerun.job.current.stage || "processing"} ${Math.round(
                          rerun.job.current.totalFrames
                            ? (rerun.job.current.progressFrame / rerun.job.current.totalFrames) *
                                100
                            : 0,
                        )}%`
                      : "starting"}
                    . The workspace refreshes when it finishes.
                  </p>
                ) : rerun.job?.done ? (
                  <p className="rerun__done">
                    {rerun.job.failed
                      ? `Re-analysis failed: ${rerun.job.outcomes[0]?.error ?? "unknown error"}`
                      : "Re-analysis complete. This page now shows the new result."}
                  </p>
                ) : (
                  <ConfirmButton
                    className="btn btn--ghost btn--sm"
                    confirmLabel="Replace this result"
                    title="Analyse this recording again, replacing the current result"
                    onConfirm={() => void startRerun()}
                  >
                    Re-run this analysis
                  </ConfirmButton>
                )}
                {rerun.error ? <p className="rerun__error">{rerun.error}</p> : null}
                <p className="micro muted">
                  Re-running replaces this result with a fresh one from the same recording. The
                  previous complete result is archived, and any condition you confirmed in the
                  assistant is applied to the new run.
                </p>
              </div>
            </article>

            <article className="assurance-card">
              <p className="insight-card__kicker">Task-specific assessment</p>
              <h3>NIOSH lifting equation</h3>
              {overview ? (
                <>
                  <div className="rnle-result">
                    <Stat
                      label="Lifting index"
                      value={number(overview.rnle.liftingIndex, 2)}
                      hint={humanise(overview.rnle.validity)}
                      emphasis
                    />
                    <Stat
                      label="Recommended weight"
                      value={
                        overview.rnle.recommendedWeightKg !== null
                          ? `${number(overview.rnle.recommendedWeightKg, 1)} kg`
                          : "Unavailable"
                      }
                    />
                  </div>
                  <p className="micro muted">
                    Applicable only to supported two-handed lifting tasks; it is not blended into
                    the posture score.
                  </p>
                </>
              ) : (
                <Skeleton height={150} />
              )}
            </article>
          </div>

          {overview?.warnings.length ? (
            <Disclosure
              summary="Method, model, and evidence limitations"
              hint={`${overview.warnings.length} to review`}
            >
              <ul className="limitation-list">
                {overview.warnings.map((warning, index) => (
                  <li key={index}>{warning}</li>
                ))}
              </ul>
            </Disclosure>
          ) : null}

          <p className="method-note">
            RULA and REBA are observational screening methods. Pose confidence and ergonomic
            validity are reported separately and are never combined into one number.
          </p>
        </section>
        </TabPanel>
      </div>

      <TabPanel tab="findings" active={tab} visited={visited.has("findings")}><AssessmentReview sessionId={sessionId} onCitation={openCitation} /></TabPanel>
      <div className="sr-only" aria-live="polite">
        {detail.data ? `Frame ${detail.data.frame} at ${detail.data.clock} selected` : ""}
      </div>
    </Shell>
  );
}
