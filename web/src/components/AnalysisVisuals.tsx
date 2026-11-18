import { useMemo, useState } from "react";
import { clock, duration, humanise, number, percent, statusInk, statusVar } from "../lib/format";
import type { Band, BandShare, BodyLoad, SeriesPoint, SessionOverview } from "../lib/types";

const SERIES = [
  "var(--series-1)",
  "var(--series-2)",
  "var(--series-3)",
  "var(--series-4)",
  "var(--series-5)",
  "var(--series-6)",
];

function quantile(points: SeriesPoint[], q: number) {
  if (!points.length) return null;
  const scores = points.map((point) => point.s).sort((a, b) => a - b);
  const index = Math.min(scores.length - 1, Math.max(0, Math.ceil(scores.length * q) - 1));
  return scores[index];
}

function bandFor(score: number | null, bands: Band[]) {
  if (score === null) return null;
  return bands.find((band) => score >= band.lower && score <= band.upper) ?? null;
}

export function RiskDistribution({
  shares,
  percentAtAction,
  timeAtActionSec,
}: {
  shares: BandShare[];
  percentAtAction: number;
  timeAtActionSec: number;
}) {
  const visible = shares.filter((share) => share.percent > 0);
  const [active, setActive] = useState(visible[visible.length - 1]?.label ?? "");
  const gradient = useMemo(() => {
    if (!visible.length) return "conic-gradient(var(--line) 0 100%)";
    let cursor = 0;
    const stops = visible.map((share) => {
      const from = cursor;
      cursor += share.percent;
      return `${statusVar(share.status)} ${from}% ${cursor}%`;
    });
    return `conic-gradient(${stops.join(", ")})`;
  }, [visible]);
  const focused = visible.find((share) => share.label === active) ?? visible[0];

  return (
    <article className="insight-card insight-card--exposure">
      <div className="insight-card__head">
        <div>
          <p className="insight-card__kicker">Exposure mix</p>
          <h3>Time by action level</h3>
        </div>
        <span className="insight-card__meta">duration weighted</span>
      </div>

      <div className="exposure-plot">
        <div
          className="risk-donut"
          style={{ "--risk-gradient": gradient } as React.CSSProperties}
          role="img"
          aria-label={`${percent(percentAtAction)} of scored time was at or above the action threshold`}
        >
          <div className="risk-donut__core">
            <strong className="num">{percent(percentAtAction)}</strong>
            <span>at action level</span>
          </div>
        </div>

        <div className="exposure-legend" aria-label="Risk band exposure">
          {visible.map((share) => (
            <button
              key={share.label}
              type="button"
              className="exposure-legend__row"
              data-active={share.label === focused?.label}
              onPointerEnter={() => setActive(share.label)}
              onFocus={() => setActive(share.label)}
              onClick={() => setActive(share.label)}
              aria-pressed={share.label === focused?.label}
            >
              <span className="sharekey__dot" style={{ background: statusVar(share.status) }} />
              <span>{share.label}</span>
              <strong className="num">{percent(share.percent)}</strong>
            </button>
          ))}
        </div>
      </div>

      {focused ? (
        <div className="insight-card__focus" aria-live="polite">
          <span
            className="insight-card__focusdot"
            style={{ background: statusVar(focused.status) }}
          />
          <span>
            <strong>{focused.label}</strong> for {duration(focused.durationSec)} · {focused.action}
          </span>
        </div>
      ) : (
        <div className="insight-card__focus">
          {duration(timeAtActionSec)} at the action threshold
        </div>
      )}
    </article>
  );
}

export function ScoreProfile({
  points,
  bands,
  typicalScore,
  peakScore,
  scaleMaximum,
  actionThreshold,
  method,
}: {
  points: SeriesPoint[];
  bands: Band[];
  typicalScore: number | null;
  peakScore: number | null;
  scaleMaximum: number;
  actionThreshold: number;
  method: string;
}) {
  const p90 = useMemo(() => quantile(points, 0.9), [points]);
  const rows = [
    { label: "Typical", value: typicalScore, color: "var(--series-1)", detail: "duration weighted" },
    { label: "90th pct.", value: p90, color: "var(--series-3)", detail: "sustained high" },
    { label: "Peak", value: peakScore, color: "var(--series-2)", detail: "single worst" },
  ];
  const trackMin = 0.5;
  const trackSpan = Math.max(1, scaleMaximum);
  const position = (value: number) =>
    Math.min(100, Math.max(0, ((value - trackMin) / trackSpan) * 100));

  return (
    <article className="insight-card">
      <div className="insight-card__head">
        <div>
          <p className="insight-card__kicker">Score profile</p>
          <h3>Typical to worst</h3>
        </div>
        <span className="insight-card__meta">{method} scale</span>
      </div>

      <div className="score-profile" role="img" aria-label={`${method} typical, 90th percentile, and peak scores`}>
        {rows.map((row) => {
          const rowBand = bandFor(row.value, bands);
          return (
            <div className="score-profile__row" key={row.label}>
              <span className="score-profile__label">
                <strong>{row.label}</strong>
                <small>{row.detail}</small>
              </span>
              <span className="score-profile__track">
                {bands.map((band) => {
                  const left = position(band.lower - 0.5);
                  const right = position(Math.min(scaleMaximum, band.upper) + 0.5);
                  return (
                    <span
                      key={band.label}
                      className="score-profile__band"
                      style={{
                        left: `${left}%`,
                        width: `${Math.max(0, right - left)}%`,
                        background: statusVar(band.status),
                      }}
                    />
                  );
                })}
                <span
                  className="score-profile__threshold"
                  style={{ left: `${position(actionThreshold)}%` }}
                  aria-hidden="true"
                />
                {row.value !== null ? (
                  <span
                    className="score-profile__marker"
                    style={{ left: `${position(row.value)}%`, background: row.color }}
                  />
                ) : null}
              </span>
              <span className="score-profile__value">
                <strong className="num" style={{ color: rowBand ? statusInk(rowBand.status) : undefined }}>
                  {number(row.value, 1)}
                </strong>
                <small>{rowBand?.label ?? "not scored"}</small>
              </span>
            </div>
          );
        })}
        <div className="score-profile__axis" aria-hidden="true">
          <span>1</span>
          <span>action {actionThreshold}+</span>
          <span>{scaleMaximum}</span>
        </div>
      </div>
    </article>
  );
}

export function EvidenceSummary({ session }: { session: SessionOverview }) {
  const items = [
    { value: duration(session.durationSec), label: "recorded" },
    { value: session.cameraCount ?? "Unavailable", label: session.cameraCount === 1 ? "camera view" : "camera views" },
    { value: (session.poseFrames ?? 0).toLocaleString(), label: "pose observations" },
    { value: session.samplingLabel, label: "sampling" },
  ];

  return (
    <article className="insight-card insight-card--evidence">
      <div className="insight-card__head">
        <div>
          <p className="insight-card__kicker">Evidence coverage</p>
          <h3>What supports the result</h3>
        </div>
        <span className="badge badge--brand">frame linked</span>
      </div>
      <div className="evidence-grid">
        {items.map((item) => (
          <div className="evidence-grid__item" key={item.label}>
            <strong className="num">{item.value}</strong>
            <span>{item.label}</span>
          </div>
        ))}
      </div>
      <div className="insight-card__focus">
        <span className="evidence-pulse" aria-hidden="true" />
        <span>
          {humanise(session.trackingMode)} tracking · {humanise(session.calibration)} calibration
        </span>
      </div>
    </article>
  );
}

export function ContributionComparison({
  loads,
  selectedRegion,
  onPick,
}: {
  loads: BodyLoad[];
  selectedRegion: string;
  onPick: (load: BodyLoad) => void;
}) {
  const maximum = Math.max(...loads.flatMap((load) => [load.meanPoints, load.worstPoints]), 0.001);

  return (
    <article className="cause-card">
      <div className="insight-card__head">
        <div>
          <p className="insight-card__kicker">Posture comparison</p>
          <h3>Typical impact vs peak impact</h3>
        </div>
        <span className="insight-card__meta">score reduction if neutral</span>
      </div>
      <div className="comparison-legend" aria-hidden="true">
        <span><i style={{ background: "var(--series-1)" }} />Typical</span>
        <span><i style={{ background: "var(--series-2)" }} />Peak</span>
      </div>
      <div className="comparison-bars">
        {loads.map((load) => (
          <button
            type="button"
            className="comparison-row"
            data-selected={selectedRegion === load.region}
            key={load.region}
            onClick={() => onPick(load)}
            disabled={load.peakFrame === null}
            title={`Inspect ${load.label} at ${clock(load.peakSec)}`}
          >
            <span className="comparison-row__label">{load.label}</span>
            <span className="comparison-row__plots">
              <span className="comparison-row__track">
                <span
                  className="comparison-row__fill comparison-row__fill--mean"
                  style={{ width: `${(load.meanPoints / maximum) * 100}%` }}
                />
              </span>
              <span className="comparison-row__track">
                <span
                  className="comparison-row__fill comparison-row__fill--peak"
                  style={{ width: `${(load.worstPoints / maximum) * 100}%` }}
                />
              </span>
            </span>
            <span className="comparison-row__values num">
              <span>+{number(load.meanPoints, 1)}</span>
              <strong>+{number(load.worstPoints, 1)}</strong>
            </span>
          </button>
        ))}
      </div>
      <p className="cause-card__foot">
        Select a region to open its peak frame and highlight it in the footage.
      </p>
    </article>
  );
}

export function PersistenceMatrix({
  loads,
  selectedRegion,
  onPick,
}: {
  loads: BodyLoad[];
  selectedRegion: string;
  onPick: (load: BodyLoad) => void;
}) {
  const [hovered, setHovered] = useState("");
  const active = hovered || selectedRegion;
  const width = 430;
  const height = 270;
  const pad = { left: 42, right: 18, top: 18, bottom: 42 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const peakMaximum = Math.max(...loads.map((load) => load.worstPoints), 1);
  const meanMaximum = Math.max(...loads.map((load) => load.meanPoints), 0.001);
  const x = (value: number) => pad.left + (Math.min(100, Math.max(0, value)) / 100) * plotW;
  const y = (value: number) => pad.top + plotH - (Math.min(peakMaximum, Math.max(0, value)) / peakMaximum) * plotH;

  return (
    <article className="cause-card">
      <div className="insight-card__head">
        <div>
          <p className="insight-card__kicker">Driver pattern</p>
          <h3>Persistence × peak impact</h3>
        </div>
        <span className="insight-card__meta">select a point to inspect</span>
      </div>
      <div className="matrix-layout">
        <svg
          className="matrix"
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label="Body regions plotted by percent of task raised above neutral and peak score impact"
        >
          <rect x={pad.left} y={pad.top} width={plotW} height={plotH} className="matrix__surface" />
          <rect
            x={pad.left + plotW / 2}
            y={pad.top}
            width={plotW / 2}
            height={plotH / 2}
            className="matrix__priority"
          />
          {[0, 25, 50, 75, 100].map((tick) => (
            <g key={`x-${tick}`}>
              <line x1={x(tick)} x2={x(tick)} y1={pad.top} y2={pad.top + plotH} className="matrix__grid" />
              <text x={x(tick)} y={height - 21} textAnchor="middle" className="matrix__tick">
                {tick}%
              </text>
            </g>
          ))}
          {[0, peakMaximum / 2, peakMaximum].map((tick) => (
            <g key={`y-${tick}`}>
              <line x1={pad.left} x2={pad.left + plotW} y1={y(tick)} y2={y(tick)} className="matrix__grid" />
              <text x={pad.left - 8} y={y(tick)} textAnchor="end" dominantBaseline="middle" className="matrix__tick">
                +{number(tick, 1)}
              </text>
            </g>
          ))}
          <text x={pad.left + plotW / 2} y={height - 3} textAnchor="middle" className="matrix__axis">
            Share of observations above neutral
          </text>
          <text
            x={12}
            y={pad.top + plotH / 2}
            textAnchor="middle"
            className="matrix__axis"
            transform={`rotate(-90 12 ${pad.top + plotH / 2})`}
          >
            Peak impact
          </text>
          <text x={pad.left + plotW - 8} y={pad.top + 15} textAnchor="end" className="matrix__quadrant">
            sustained priority
          </text>

          {loads.map((load, index) => {
            const radius = 10 + (load.meanPoints / meanMaximum) * 5;
            const isActive = active === load.region;
            return (
              <g
                className="matrix__point"
                data-active={isActive}
                key={load.region}
                role="button"
                tabIndex={0}
                aria-label={`${load.label}: raised ${percent(load.percentRaised)} of observations, peak impact plus ${number(load.worstPoints, 1)}`}
                onPointerEnter={() => setHovered(load.region)}
                onPointerLeave={() => setHovered("")}
                onFocus={() => setHovered(load.region)}
                onBlur={() => setHovered("")}
                onClick={() => onPick(load)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    onPick(load);
                  }
                }}
              >
                <circle
                  cx={x(load.percentRaised)}
                  cy={y(load.worstPoints)}
                  r={radius}
                  fill={SERIES[index % SERIES.length]}
                />
                <text
                  x={x(load.percentRaised)}
                  y={y(load.worstPoints)}
                  textAnchor="middle"
                  dominantBaseline="central"
                  className="matrix__pointnumber"
                >
                  {index + 1}
                </text>
              </g>
            );
          })}
        </svg>

        <div className="matrix-key">
          {loads.map((load, index) => (
            <button
              type="button"
              key={load.region}
              data-active={active === load.region}
              onPointerEnter={() => setHovered(load.region)}
              onPointerLeave={() => setHovered("")}
              onFocus={() => setHovered(load.region)}
              onBlur={() => setHovered("")}
              onClick={() => onPick(load)}
              disabled={load.peakFrame === null}
            >
              <span style={{ background: SERIES[index % SERIES.length] }}>{index + 1}</span>
              <span>{load.label}</span>
              <strong className="num">{percent(load.percentRaised)}</strong>
            </button>
          ))}
        </div>
      </div>
      <p className="cause-card__foot">
        Upper right combines repeated exposure with a strong peak contribution.
      </p>
    </article>
  );
}
