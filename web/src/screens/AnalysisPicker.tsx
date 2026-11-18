import { Link, useNavigate, useParams } from "react-router-dom";
import { Shell } from "../components/Shell";
import { IconArrowRight, IconBody, IconGauge } from "../components/Icons";
import { ErrorState, Skeleton, Stat } from "../components/ui";
import { api, useAsync } from "../lib/api";
import { clock, duration, number } from "../lib/format";
import type { MethodKey, SessionOverview } from "../lib/types";

const DESCRIPTIONS: Record<MethodKey, { title: string; scope: string; detail: string }> = {
  reba: {
    title: "REBA",
    scope: "Whole body",
    detail:
      "Rapid Entire Body Assessment. Covers trunk, neck, legs, arms and wrists together with load, coupling and activity. Use it for lifting, carrying and any task where the legs and back are doing work.",
  },
  rula: {
    title: "RULA",
    scope: "Upper limb",
    detail:
      "Rapid Upper Limb Assessment. Focuses on the arms, wrists, neck and trunk. Use it for seated or bench work where the upper limb dominates the exposure.",
  },
};

function MethodCard({
  method,
  session,
  onPick,
}: {
  method: MethodKey;
  session: SessionOverview;
  onPick: (method: MethodKey) => void;
}) {
  const copy = DESCRIPTIONS[method];
  const headline = session.methods[method];
  const unavailable = headline?.maximum === null || headline?.maximum === undefined;

  return (
    <button
      type="button"
      className="card card--interactive"
      style={{ padding: "var(--s-6)" }}
      onClick={() => onPick(method)}
      disabled={unavailable}
      aria-label={`Open ${copy.title} analysis`}
    >
      <div className="stack stack-4">
        <div className="row row-3" style={{ justifyContent: "space-between" }}>
          <span
            style={{
              display: "grid",
              placeItems: "center",
              width: 40,
              height: 40,
              borderRadius: "var(--r-lg)",
              background: method === "reba" ? "var(--brand-soft)" : "var(--accent-soft)",
              color: method === "reba" ? "var(--brand-active)" : "var(--accent-hover)",
            }}
          >
            {method === "reba" ? <IconBody size={20} /> : <IconGauge size={20} />}
          </span>
          <span className="badge badge--neutral">{copy.scope}</span>
        </div>

        <div className="stack stack-2">
          <h2>{copy.title} analysis</h2>
          <p className="small muted">{copy.detail}</p>
        </div>

        {unavailable ? (
          <span className="badge badge--neutral">No valid score in this recording</span>
        ) : (
          <div
            className="row row-5 wrap"
            style={{
              gap: "var(--s-5)",
              paddingTop: "var(--s-3)",
              borderTop: "1px solid var(--line)",
            }}
          >
            <Stat
              label="Worst score"
              value={headline.maximum}
              status={headline.band?.status}
              hint={headline.band?.label}
            />
            <Stat label="90th percentile" value={number(headline.percentile90 ?? null, 0)} />
            <Stat
              label="Typical"
              value={number(headline.durationWeightedMean ?? null, 1)}
              hint="duration-weighted"
            />
            <Stat label="Worst at" value={clock(headline.peakSec ?? null)} hint={`frame ${headline.peakFrame ?? "—"}`} />
          </div>
        )}

        <span
          className="row row-2 small"
          style={{ color: method === "reba" ? "var(--brand-active)" : "var(--accent-hover)", fontWeight: 600 }}
        >
          Open {copy.title.toLowerCase()} timeline
          <IconArrowRight size={14} />
        </span>
      </div>
    </button>
  );
}

export default function AnalysisPicker() {
  const { sessionId = "" } = useParams();
  const navigate = useNavigate();
  const { data, error, loading, reload } = useAsync(
    (signal) => api.session(sessionId, signal),
    [sessionId],
  );

  const crumbs = [
    { label: "Home", to: "/" },
    { label: "Assessments", to: "/assessments" },
    { label: data ? `Participant ${data.participant.participantId}` : "Recording" },
  ];

  if (error) {
    return (
      <Shell crumbs={crumbs}>
        <ErrorState message={error} onRetry={reload} />
        <p style={{ marginTop: "var(--s-4)" }}>
          <Link to="/library">Back to the library</Link>
        </p>
      </Shell>
    );
  }

  if (loading || !data) {
    return (
      <Shell crumbs={crumbs}>
        <div className="stack stack-5">
          <Skeleton height={30} width={340} />
          <Skeleton height={16} width={520} />
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "var(--s-4)" }}>
            <Skeleton height={300} radius="var(--r-lg)" />
            <Skeleton height={300} radius="var(--r-lg)" />
          </div>
        </div>
      </Shell>
    );
  }

  return (
    <Shell crumbs={crumbs} session={data}>
      <div className="stack stack-6" style={{ maxWidth: 940 }}>
        <div className="stack stack-3">
          <p className="eyebrow">Recording loaded</p>
          <h1>{data.trial.name}</h1>
          <p className="lede">{data.trial.description}</p>
          <div
            className="row row-5 wrap"
            style={{ gap: "var(--s-5)", paddingTop: "var(--s-2)" }}
          >
            <Stat label="Participant" value={data.participant.participantId} hint={data.participant.recordedAt ?? undefined} />
            <Stat label="Duration" value={duration(data.durationSec)} />
            <Stat label="Cameras" value={data.cameraCount ?? "—"} />
            <Stat
              label="Pose observations"
              value={(data.poseFrames ?? 0).toLocaleString()}
              hint={`of ${(data.processedFrames ?? 0).toLocaleString()} frames`}
            />
            <Stat label="Sampling" value={data.samplingLabel} />
          </div>
        </div>

        <div className="stack stack-3">
          <h2>Choose an analysis</h2>
          <p className="small muted" style={{ maxWidth: "62ch" }}>
            Both methods were calculated on every frame of this recording. They cover different
            body regions and use different scales, so the workspace follows one at a time. You can
            switch at any point without reloading.
          </p>
          <div
            className="stagger"
            style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "var(--s-4)" }}
          >
            <MethodCard method="reba" session={data} onPick={(m) => navigate(`/s/${sessionId}/${m}`)} />
            <MethodCard method="rula" session={data} onPick={(m) => navigate(`/s/${sessionId}/${m}`)} />
          </div>
        </div>

        {data.rnle.liftingIndex !== null || data.rnle.validity ? (
          <div className="card" style={{ padding: "var(--s-4) var(--s-5)" }}>
            <div className="row row-4 wrap" style={{ justifyContent: "space-between" }}>
              <div className="stack stack-1">
                <span className="small" style={{ fontWeight: 600 }}>
                  NIOSH lifting equation
                </span>
                <span className="micro muted">
                  Task-level, calculated once for the whole recording rather than per frame.
                </span>
              </div>
              <div className="row row-5" style={{ gap: "var(--s-5)" }}>
                <Stat label="Lifting index" value={number(data.rnle.liftingIndex, 2)} />
                <Stat label="RWL" value={number(data.rnle.recommendedWeightKg, 1)} hint="kg" />
                <Stat label="Validity" value={<span className="small">{data.rnle.validity ?? "—"}</span>} />
              </div>
            </div>
          </div>
        ) : null}
      </div>
    </Shell>
  );
}
