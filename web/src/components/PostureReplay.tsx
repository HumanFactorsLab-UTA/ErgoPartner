import { useEffect, useMemo, useState } from "react";
import { api, useAsync, useDebounced } from "../lib/api";
import { clock, humanise, statusInk, statusVar } from "../lib/format";
import type { MethodKey, ReplayFrame, ReplayPoint } from "../lib/types";
import { IconArrowLeft, IconArrowRight, IconPause, IconPlay, IconTarget } from "./Icons";
import { BandBadge, Skeleton } from "./ui";

const REGION_POINTS: Record<string, Set<number>> = {
  neck: new Set([0, 7, 8, 11, 12, 100, 101]),
  trunk: new Set([11, 12, 23, 24, 100, 102]),
  left_shoulder: new Set([11, 13]),
  right_shoulder: new Set([12, 14]),
  left_elbow: new Set([11, 13, 15]),
  right_elbow: new Set([12, 14, 16]),
  left_wrist: new Set([15, 17, 19, 21]),
  right_wrist: new Set([16, 18, 20, 22]),
  legs: new Set([23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 102]),
};

function Pose({
  frame,
  connections,
  threshold,
  region,
  imageWidth,
  imageHeight,
  ghost = false,
}: {
  frame: ReplayFrame;
  connections: [number, number][];
  threshold: number;
  region: string;
  imageWidth: number;
  imageHeight: number;
  ghost?: boolean;
}) {
  const points = new Map(frame.points.map((point) => [point.id, point]));
  const selected = REGION_POINTS[region] ?? new Set<number>();
  const visible = (point: ReplayPoint | undefined) =>
    point && Number.isFinite(point.x) && Number.isFinite(point.y) && point.visibility >= threshold;

  return (
    <g className={ghost ? "replay-pose replay-pose--ghost" : "replay-pose"}>
      {connections.map(([from, to]) => {
        const first = points.get(from);
        const second = points.get(to);
        if (!visible(first) || !visible(second)) return null;
        const highlighted = selected.has(from) && selected.has(to);
        return (
          <line
            key={`${from}-${to}`}
            x1={(first as ReplayPoint).x * imageWidth}
            y1={(first as ReplayPoint).y * imageHeight}
            x2={(second as ReplayPoint).x * imageWidth}
            y2={(second as ReplayPoint).y * imageHeight}
            data-highlighted={highlighted}
          />
        );
      })}
      {!ghost
        ? frame.points.map((point) => {
            if (!visible(point)) return null;
            const highlighted = selected.has(point.id);
            return (
              <circle
                key={point.id}
                cx={point.x * imageWidth}
                cy={point.y * imageHeight}
                r={(highlighted ? 6.5 : 3.8) * (imageWidth / 1000)}
                opacity={Math.max(0.42, Math.min(1, point.visibility))}
                data-highlighted={highlighted}
              />
            );
          })
        : null}
    </g>
  );
}

export function PostureReplay({
  sessionId,
  method,
  cameraCount,
  cameraRoles,
  camera,
  onCameraChange,
  person,
  selectedFrame,
  region,
  onSelect,
  onInspect,
}: {
  sessionId: string;
  method: MethodKey;
  cameraCount: number;
  cameraRoles: Record<number, string>;
  camera: number;
  onCameraChange: (camera: number) => void;
  person: string | null;
  selectedFrame: number | null;
  region: string;
  onSelect: (frame: number) => void;
  onInspect: (frame: number) => void;
}) {
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const anchorFrame = useDebounced(selectedFrame, 120);
  const replay = useAsync(
    (signal) => api.replay(sessionId, method, camera, person, anchorFrame, signal),
    [sessionId, method, camera, person, anchorFrame],
    { keepPrevious: true },
  );
  const frames = replay.data?.frames ?? [];

  useEffect(() => {
    setPlaying(false);
    setIndex(0);
  }, [sessionId, method, camera]);

  useEffect(() => {
    if (playing || selectedFrame === null || !frames.length) return;
    let nearest = 0;
    for (let position = 1; position < frames.length; position += 1) {
      if (
        Math.abs(frames[position].f - selectedFrame) <
        Math.abs(frames[nearest].f - selectedFrame)
      ) {
        nearest = position;
      }
    }
    setIndex(nearest);
  }, [frames, playing, selectedFrame]);

  useEffect(() => {
    if (!playing || frames.length < 2) return;
    const current = frames[index];
    const nextIndex = index + 1 < frames.length ? index + 1 : 0;
    const next = frames[nextIndex];
    const recordedDelay =
      nextIndex === 0 ? 100 : Math.max(25, Math.min(500, (next.t - current.t) * 1000));
    const timer = window.setTimeout(() => setIndex(nextIndex), recordedDelay / speed);
    return () => window.clearTimeout(timer);
  }, [frames, index, playing, speed]);

  const current = frames[Math.min(index, Math.max(0, frames.length - 1))] ?? null;
  const ghost = frames[Math.max(0, index - 3)] ?? null;
  const cameras = useMemo(
    () => Array.from({ length: Math.max(1, cameraCount) }, (_, position) => position),
    [cameraCount],
  );

  const commit = (position: number) => {
    if (!frames.length) return;
    const bounded = Math.max(0, Math.min(frames.length - 1, position));
    setIndex(bounded);
    setPlaying(false);
    onSelect(frames[bounded].f);
  };

  return (
    <article className="replay-card" data-updating={replay.loading}>
      <header className="replay-card__head">
        <div>
          <p className="insight-card__kicker">Landmark reconstruction</p>
          <h3>Stick-figure posture replay</h3>
          <p className="micro muted">
            The same stored landmarks used for scoring, without the camera background.
          </p>
        </div>
        {cameraCount > 1 && <div className="replay-card__cameras" role="group" aria-label="Replay camera">
          {cameras.map((item) => (
            <button
              type="button"
              key={item}
              aria-pressed={camera === item}
              onClick={() => onCameraChange(item)}
            >
              {humanise(cameraRoles[item] ?? `camera ${item + 1}`)}
            </button>
          ))}
        </div>}
      </header>

      {!replay.data && replay.loading ? (
        <Skeleton height={390} radius="var(--r-lg)" />
      ) : replay.error ? (
        <div className="replay-card__empty">
          <IconTarget size={22} />
          <div>
            <strong>Landmark replay is unavailable</strong>
            <p className="micro muted">{replay.error}</p>
          </div>
        </div>
      ) : !current || !replay.data ? (
        <div className="replay-card__empty">
          <IconTarget size={22} />
          <div>
            <strong>No stored landmarks for this camera</strong>
            <p className="micro muted">A new analysis run will store the replay trace.</p>
          </div>
        </div>
      ) : (
        <>
          <div
            className="replay-stage"
            style={
              {
                "--replay-band": statusVar(current.band?.status),
              } as React.CSSProperties
            }
          >
            <svg
              viewBox={`0 0 ${replay.data.imageWidth} ${replay.data.imageHeight}`}
              role="img"
              aria-label={`Stick figure at ${clock(current.t)}, frame ${current.f}`}
              preserveAspectRatio="xMidYMid meet"
            >
              <defs>
                <radialGradient id="replay-ground" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="var(--brand)" stopOpacity="0.12" />
                  <stop offset="100%" stopColor="var(--brand)" stopOpacity="0" />
                </radialGradient>
              </defs>
              <ellipse
                cx={replay.data.imageWidth / 2}
                cy={replay.data.imageHeight * 0.937}
                rx={replay.data.imageWidth * 0.26}
                ry={replay.data.imageHeight * 0.039}
                fill="url(#replay-ground)"
              />
              {ghost && ghost.f !== current.f ? (
                <Pose
                  frame={ghost}
                  connections={replay.data.connections}
                  threshold={replay.data.visibilityThreshold}
                  region={region}
                  imageWidth={replay.data.imageWidth}
                  imageHeight={replay.data.imageHeight}
                  ghost
                />
              ) : null}
              <Pose
                frame={current}
                connections={replay.data.connections}
                threshold={replay.data.visibilityThreshold}
                region={region}
                imageWidth={replay.data.imageWidth}
                imageHeight={replay.data.imageHeight}
              />
            </svg>

            <div className="replay-stage__readout">
              <div>
                <span className="micro">Replay moment</span>
                <strong className="num">{clock(current.t)}</strong>
                <span className="micro num">frame {current.f}</span>
              </div>
              <div>
                <span className="micro">{method.toUpperCase()}</span>
                <strong
                  className="num"
                  style={{ color: current.band ? statusInk(current.band.status) : undefined }}
                >
                  {current.score ?? "—"}
                </strong>
                <BandBadge band={current.band} size="sm" />
              </div>
            </div>
          </div>

          <div className="replay-controls">
            <div className="replay-controls__transport">
              <button
                type="button"
                className="iconbtn"
                onClick={() => commit(index - 1)}
                disabled={index <= 0}
                aria-label="Previous replay observation"
              >
                <IconArrowLeft size={15} />
              </button>
              <button
                type="button"
                className="replay-controls__play"
                onClick={() => {
                  if (playing) onSelect(current.f);
                  setPlaying((value) => !value);
                }}
                aria-label={playing ? "Pause replay" : "Play replay"}
              >
                {playing ? <IconPause size={15} /> : <IconPlay size={15} />}
                {playing ? "Pause" : "Play"}
              </button>
              <button
                type="button"
                className="iconbtn"
                onClick={() => commit(index + 1)}
                disabled={index >= frames.length - 1}
                aria-label="Next replay observation"
              >
                <IconArrowRight size={15} />
              </button>
            </div>

            <label className="replay-controls__scrub">
              <span className="sr-only">Replay position</span>
              <input
                type="range"
                min={0}
                max={Math.max(0, frames.length - 1)}
                value={Math.min(index, Math.max(0, frames.length - 1))}
                onChange={(event) => commit(Number(event.target.value))}
              />
            </label>

            <label className="replay-controls__speed">
              <span>Speed</span>
              <select value={speed} onChange={(event) => setSpeed(Number(event.target.value))}>
                <option value={0.5}>0.5×</option>
                <option value={1}>1×</option>
                <option value={2}>2×</option>
              </select>
            </label>

            <button
              type="button"
              className="btn btn--ghost"
              onClick={() => onInspect(current.f)}
            >
              Open {cameraCount > 1 ? `${cameraCount}-camera` : "camera"} evidence
              <IconArrowRight size={14} />
            </button>
          </div>

          <footer className="replay-card__foot">
            <span>
              {replay.data.sampledCount.toLocaleString()} replay poses from{" "}
              {replay.data.sourceCount.toLocaleString()} observations
            </span>
            {replay.data.anchorFrame !== null ? (
              <span>
                {replay.data.anchorIncluded
                  ? "Exact selected-frame pose preserved"
                  : "Selected frame has no replayable pose; nearest observation shown"}
              </span>
            ) : null}
            <span>Recorded cadence; long evidence gaps are compressed for review</span>
            {region ? <span>Highlighting {humanise(region)}</span> : null}
          </footer>
        </>
      )}
    </article>
  );
}
