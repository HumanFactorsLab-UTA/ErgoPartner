import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { clock, number, statusVar } from "../lib/format";
import type { Band, Episode, SeriesPoint } from "../lib/types";

interface Props {
  points: SeriesPoint[];
  bands: Band[];
  episodes: Episode[];
  scaleMaximum: number;
  method: string;
  selectedFrame: number | null;
  onSelect: (point: SeriesPoint) => void;
  typicalScore?: number | null;
  actionThreshold?: number | null;
  height?: number;
}

interface Extent {
  from: number;
  to: number;
}

const PAD = { top: 16, right: 84, bottom: 28, left: 40 };

/**
 * Score against time, on the method's own published action bands.
 *
 * Hand-drawn SVG rather than a charting library, because the timeline is the
 * product's primary control surface, not a figure on a page: it has to hit-test
 * a thousand points against a moving pointer, brush a range, keep a selection
 * marker in sync with a video element, and stay keyboard-operable. Wiring that
 * through a generic chart API costs more than drawing the shape.
 */
export function Timeline({
  points,
  bands,
  episodes,
  scaleMaximum,
  method,
  selectedFrame,
  onSelect,
  typicalScore = null,
  actionThreshold = null,
  height = 260,
}: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(960);
  const [hover, setHover] = useState<SeriesPoint | null>(null);
  const [extent, setExtent] = useState<Extent | null>(null);
  const [brush, setBrush] = useState<{ from: number; to: number } | null>(null);

  useEffect(() => {
    const node = wrapRef.current;
    if (!node) return;
    const observer = new ResizeObserver((entries) => {
      const next = entries[0]?.contentRect.width;
      if (next && Math.abs(next - width) > 1) setWidth(next);
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, [width]);

  const full = useMemo<Extent>(() => {
    if (points.length === 0) return { from: 0, to: 1 };
    return { from: points[0].t, to: points[points.length - 1].t };
  }, [points]);

  const view = extent ?? full;
  const visible = useMemo(
    () => points.filter((p) => p.t >= view.from && p.t <= view.to),
    [points, view],
  );

  const plotW = Math.max(80, width - PAD.left - PAD.right);
  const plotH = Math.max(60, height - PAD.top - PAD.bottom);
  const span = Math.max(view.to - view.from, 1e-6);

  const x = useCallback((t: number) => PAD.left + ((t - view.from) / span) * plotW, [view.from, span, plotW]);
  const y = useCallback(
    (score: number) => PAD.top + plotH - ((score - 0.5) / (scaleMaximum + 0.5 - 0.5)) * plotH,
    [plotH, scaleMaximum],
  );

  /** Step path: the score holds until the next observation, which is what it did. */
  const path = useMemo(() => {
    if (visible.length === 0) return "";
    const parts: string[] = [`M ${x(visible[0].t).toFixed(2)} ${y(visible[0].s).toFixed(2)}`];
    for (let i = 1; i < visible.length; i += 1) {
      const px = x(visible[i].t).toFixed(2);
      parts.push(`L ${px} ${y(visible[i - 1].s).toFixed(2)}`);
      parts.push(`L ${px} ${y(visible[i].s).toFixed(2)}`);
    }
    return parts.join(" ");
  }, [visible, x, y]);

  const area = useMemo(() => {
    if (!path) return "";
    const first = x(visible[0].t).toFixed(2);
    const last = x(visible[visible.length - 1].t).toFixed(2);
    const base = (PAD.top + plotH).toFixed(2);
    return `${path} L ${last} ${base} L ${first} ${base} Z`;
  }, [path, visible, x, plotH]);

  const ticks = useMemo(() => {
    const target = Math.max(3, Math.min(9, Math.floor(plotW / 110)));
    const steps = [1, 2, 5, 10, 15, 20, 30, 60, 120, 300, 600];
    const step = steps.find((s) => span / s <= target) ?? 900;
    const out: number[] = [];
    for (let t = Math.ceil(view.from / step) * step; t <= view.to; t += step) out.push(t);
    return out;
  }, [span, plotW, view]);

  const nearest = useCallback(
    (clientX: number): SeriesPoint | null => {
      const node = wrapRef.current?.querySelector("svg");
      if (!node || visible.length === 0) return null;
      const box = node.getBoundingClientRect();
      const t = view.from + ((clientX - box.left - PAD.left) / plotW) * span;
      let best = visible[0];
      let bestGap = Math.abs(best.t - t);
      for (const point of visible) {
        const gap = Math.abs(point.t - t);
        if (gap < bestGap) {
          best = point;
          bestGap = gap;
        }
      }
      return best;
    },
    [visible, view.from, plotW, span],
  );

  const selected = useMemo(
    () => points.find((p) => p.f === selectedFrame) ?? null,
    [points, selectedFrame],
  );

  const navigatorPath = useMemo(() => {
    if (points.length === 0) return "";
    const navigatorHeight = 38;
    const navigatorTop = 4;
    const navigatorWidth = Math.max(80, width - PAD.left - PAD.right);
    const fullSpan = Math.max(full.to - full.from, 1e-6);
    const navigatorX = (time: number) =>
      PAD.left + ((time - full.from) / fullSpan) * navigatorWidth;
    const navigatorY = (score: number) =>
      navigatorTop +
      navigatorHeight -
      ((score - 0.5) / Math.max(1, scaleMaximum)) * navigatorHeight;
    const parts = [
      `M ${navigatorX(points[0].t).toFixed(2)} ${navigatorY(points[0].s).toFixed(2)}`,
    ];
    for (let index = 1; index < points.length; index += 1) {
      const pointX = navigatorX(points[index].t).toFixed(2);
      parts.push(`L ${pointX} ${navigatorY(points[index - 1].s).toFixed(2)}`);
      parts.push(`L ${pointX} ${navigatorY(points[index].s).toFixed(2)}`);
    }
    return parts.join(" ");
  }, [points, width, full, scaleMaximum]);

  const onMove = (event: React.PointerEvent) => {
    const point = nearest(event.clientX);
    setHover(point);
    if (brush) {
      const node = wrapRef.current?.querySelector("svg");
      if (!node) return;
      const box = node.getBoundingClientRect();
      const t = view.from + ((event.clientX - box.left - PAD.left) / plotW) * span;
      setBrush({ ...brush, to: Math.min(Math.max(t, view.from), view.to) });
    }
  };

  const onDown = (event: React.PointerEvent) => {
    if (event.button !== 0) return;
    const node = wrapRef.current?.querySelector("svg");
    if (!node) return;
    const box = node.getBoundingClientRect();
    const t = view.from + ((event.clientX - box.left - PAD.left) / plotW) * span;
    setBrush({ from: t, to: t });
    setHover(nearest(event.clientX));
    try {
      // Capture keeps a drag tracking outside the SVG. It throws when the
      // pointer id is not active -- a synthetic event, or a pointer released
      // between the check and the call -- and a failure here must not take the
      // click with it.
      (event.target as Element).setPointerCapture?.(event.pointerId);
    } catch {
      /* dragging still works without capture; it just stops at the edge */
    }
  };

  const onUp = (event: React.PointerEvent) => {
    if (brush && Math.abs(brush.to - brush.from) > span * 0.012) {
      setExtent({ from: Math.min(brush.from, brush.to), to: Math.max(brush.from, brush.to) });
    } else {
      // Resolve from this event rather than from hover state. A tap, or a click
      // fast enough that no pointermove lands between down and up, leaves hover
      // still null from the previous render -- and the selection silently does
      // nothing, which is the one interaction this whole view exists for.
      const point = nearest(event.clientX);
      if (point) onSelect(point);
    }
    setBrush(null);
  };

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (!selected) return;
    const index = points.findIndex((p) => p.f === selected.f);
    if (index < 0) return;
    let next = index;
    if (event.key === "ArrowRight") next = Math.min(points.length - 1, index + 1);
    else if (event.key === "ArrowLeft") next = Math.max(0, index - 1);
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = points.length - 1;
    else return;
    event.preventDefault();
    onSelect(points[next]);
  };

  const tip = hover ?? selected;

  return (
    <div className="tl" ref={wrapRef}>
      <div className="tl__toolbar">
        <div className="tl__instructions">
          <span className="tl__live" aria-hidden="true" />
          <span className="micro muted">
            Drag to focus a range · click to inspect · arrow keys step observations
          </span>
        </div>
        <div className="tl__tools">
          {selected ? (
            <span className="tl__selection">
              <strong className="num">{clock(selected.t)}</strong>
              <span className="num">
                {method} {selected.s}
              </span>
            </span>
          ) : null}
          {extent ? (
            <button type="button" className="btn btn--ghost" onClick={() => setExtent(null)}>
              Show full task
            </button>
          ) : null}
        </div>
      </div>

      <svg
        width={width}
        height={height}
        role="application"
        aria-label={`${method} score against time. Use arrow keys to step between observations.`}
        tabIndex={0}
        onKeyDown={onKeyDown}
        onPointerMove={onMove}
        onPointerLeave={() => setHover(null)}
        onPointerDown={onDown}
        onPointerUp={onUp}
        style={{ touchAction: "none", cursor: brush ? "ew-resize" : "crosshair" }}
      >
        <defs>
          <linearGradient id="tl-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--brand)" stopOpacity="0.16" />
            <stop offset="100%" stopColor="var(--brand)" stopOpacity="0.01" />
          </linearGradient>
        </defs>

        {/* Action bands, faint: they are the reference scale, not the finding. */}
        {bands.map((band) => {
          const top = y(Math.min(band.upper, scaleMaximum) + 0.5);
          const bottom = y(band.lower - 0.5);
          return (
            <g key={band.label}>
              <rect
                x={PAD.left}
                y={top}
                width={plotW}
                height={Math.max(0, bottom - top)}
                fill={statusVar(band.status)}
                opacity={0.07}
              />
              <text
                x={PAD.left + plotW + 8}
                y={(top + bottom) / 2}
                dominantBaseline="middle"
                fontSize="11"
                fill="var(--ink-muted)"
              >
                {band.label}
              </text>
            </g>
          );
        })}

        {/* Episodes: the blocks of time that matter, not just the instants. */}
        {episodes.map((episode) => {
          const left = x(Math.max(episode.startSec, view.from));
          const right = x(Math.min(episode.endSec, view.to));
          if (right < PAD.left || left > PAD.left + plotW) return null;
          return (
            <rect
              key={episode.index}
              x={left}
              y={PAD.top}
              width={Math.max(1.5, right - left)}
              height={plotH}
              fill={statusVar(episode.band?.status ?? "none")}
              opacity={0.14}
            />
          );
        })}

        {ticks.map((t) => (
          <g key={t}>
            <line
              x1={x(t)}
              x2={x(t)}
              y1={PAD.top}
              y2={PAD.top + plotH}
              stroke="var(--grid)"
              strokeWidth={1}
            />
            <text
              x={x(t)}
              y={PAD.top + plotH + 17}
              textAnchor="middle"
              fontSize="11"
              fill="var(--ink-muted)"
              className="num"
            >
              {clock(t)}
            </text>
          </g>
        ))}

        {Array.from({ length: scaleMaximum }, (_, i) => i + 1)
          .filter((s) => scaleMaximum <= 8 || s % 2 === 1)
          .map((score) => (
            <text
              key={score}
              x={PAD.left - 8}
              y={y(score)}
              textAnchor="end"
              dominantBaseline="middle"
              fontSize="11"
              fill="var(--ink-muted)"
              className="num"
            >
              {score}
            </text>
          ))}

        <path d={area} fill="url(#tl-fill)" />
        {typicalScore !== null ? (
          <g className="tl__reference">
            <line
              x1={PAD.left}
              x2={PAD.left + plotW}
              y1={y(typicalScore)}
              y2={y(typicalScore)}
              stroke="var(--series-3)"
              strokeWidth={1}
              strokeDasharray="5 4"
            />
            <text
              x={PAD.left + 6}
              y={y(typicalScore) - 6}
              fontSize="10"
              fill="var(--ink-secondary)"
            >
              typical {number(typicalScore, 1)}
            </text>
          </g>
        ) : null}
        {actionThreshold !== null ? (
          <g className="tl__reference">
            <line
              x1={PAD.left}
              x2={PAD.left + plotW}
              y1={y(actionThreshold)}
              y2={y(actionThreshold)}
              stroke="var(--serious)"
              strokeWidth={1.25}
              strokeDasharray="2 4"
            />
            <text
              x={PAD.left + plotW - 6}
              y={y(actionThreshold) - 6}
              textAnchor="end"
              fontSize="10"
              fill="var(--serious-ink)"
            >
              action threshold {actionThreshold}
            </text>
          </g>
        ) : null}
        <path
          className="tl__area-line"
          d={path}
          fill="none"
          stroke="var(--brand)"
          strokeWidth={1.75}
          strokeLinejoin="round"
        />

        {brush ? (
          <rect
            x={Math.min(x(brush.from), x(brush.to))}
            y={PAD.top}
            width={Math.abs(x(brush.to) - x(brush.from))}
            height={plotH}
            fill="var(--brand)"
            opacity={0.14}
            stroke="var(--brand)"
            strokeWidth={1}
          />
        ) : null}

        {selected ? (
          <g className="tl__selected">
            <line
              x1={x(selected.t)}
              x2={x(selected.t)}
              y1={PAD.top}
              y2={PAD.top + plotH}
              stroke="var(--accent)"
              strokeWidth={2}
            />
            <circle
              cx={x(selected.t)}
              cy={y(selected.s)}
              r={6}
              fill="var(--accent)"
              stroke="var(--surface-raised)"
              strokeWidth={2}
            />
          </g>
        ) : null}

        {hover && hover.f !== selected?.f ? (
          <g className="tl__hover" pointerEvents="none">
            <line
              x1={x(hover.t)}
              x2={x(hover.t)}
              y1={PAD.top}
              y2={PAD.top + plotH}
              stroke="var(--ink-muted)"
              strokeWidth={1}
              strokeDasharray="3 3"
            />
            <circle
              cx={x(hover.t)}
              cy={y(hover.s)}
              r={4.5}
              fill={statusVar(hover.b)}
              stroke="var(--surface-raised)"
              strokeWidth={2}
            />
          </g>
        ) : null}
      </svg>

      {tip ? (
        <div
          className="tl__tip"
          style={{
            left: Math.min(Math.max(x(tip.t), PAD.left + 4), PAD.left + plotW - 4),
            top: Math.max(PAD.top + 50, y(tip.s) + 48),
          }}
          role="status"
        >
          <strong className="num">{clock(tip.t)}</strong>
          <span className="num">
            {method} {tip.s}
          </span>
          <span className="micro muted num">frame {tip.f}</span>
        </div>
      ) : null}

      <div className="tl__navigator" aria-hidden="true">
        <span className="tl__navigator-label">Full recording</span>
        <svg width={width} height={48} viewBox={`0 0 ${width} 48`}>
          <path
            d={navigatorPath}
            fill="none"
            stroke="var(--brand)"
            strokeWidth={1.25}
            opacity={0.55}
          />
          {extent ? (
            <rect
              x={
                PAD.left +
                ((extent.from - full.from) / Math.max(full.to - full.from, 1e-6)) * plotW
              }
              y={2}
              width={Math.max(
                2,
                ((extent.to - extent.from) / Math.max(full.to - full.from, 1e-6)) * plotW,
              )}
              height={42}
              rx={4}
              fill="var(--brand)"
              opacity={0.12}
              stroke="var(--brand)"
              strokeWidth={1}
            />
          ) : (
            <rect
              x={PAD.left}
              y={2}
              width={plotW}
              height={42}
              rx={4}
              fill="var(--brand)"
              opacity={0.05}
            />
          )}
        </svg>
      </div>
    </div>
  );
}
