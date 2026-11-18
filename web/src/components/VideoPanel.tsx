import { useEffect, useRef, useState } from "react";
import { api, useVideoReady } from "../lib/api";
import { clock, humanise } from "../lib/format";
import { IconPause, IconPlay, IconTarget } from "./Icons";
import { Segmented } from "./ui";

interface Props {
  sessionId: string;
  cameras: { camera_index: number; name: string }[];
  roles: Record<number, string>;
  camera: number;
  onCameraChange: (camera: number) => void;
  /** Seconds into the recording that the timeline selection points at. */
  selectedSec: number | null;
  /** Called while the video plays so the timeline can follow it. */
  onScrubbed?: (seconds: number) => void;
}

/**
 * Video playback tied to the timeline selection.
 *
 * The link is deliberately one-way by default: choosing a moment on the timeline
 * seeks the video, but playing the video does not drag the timeline unless the
 * reviewer asks it to. Two-way binding turns every play into a stream of
 * selection changes, which refetches frame analysis continuously and makes the
 * page feel like it is fighting the user. "Follow playback" opts into it.
 */
export function VideoPanel({ sessionId, cameras, roles, camera, onCameraChange, selectedSec, onScrubbed }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  const [position, setPosition] = useState(0);
  const [length, setLength] = useState(0);
  const [follow, setFollow] = useState(false);
  const [failed, setFailed] = useState(false);
  const lastSeek = useRef<number | null>(null);
  const playback = useVideoReady(sessionId, camera);

  // Seek when the timeline selection moves, not when the video reports its own
  // progress; comparing against the last commanded seek stops the two from
  // chasing each other.
  useEffect(() => {
    const node = videoRef.current;
    if (!node || selectedSec === null || !Number.isFinite(selectedSec)) return;
    if (lastSeek.current !== null && Math.abs(lastSeek.current - selectedSec) < 0.02) return;
    lastSeek.current = selectedSec;
    try {
      node.currentTime = selectedSec;
      setPosition(selectedSec);
    } catch {
      /* the browser refuses a seek until metadata has loaded; loadedmetadata retries */
    }
  }, [selectedSec, camera]);

  useEffect(() => {
    setFailed(false);
  }, [camera]);

  const toggle = () => {
    const node = videoRef.current;
    if (!node) return;
    if (node.paused) {
      void node.play().then(() => setPlaying(true)).catch(() => setPlaying(false));
    } else {
      node.pause();
      setPlaying(false);
    }
  };

  const scrub = (seconds: number) => {
    const node = videoRef.current;
    if (!node) return;
    node.currentTime = seconds;
    lastSeek.current = seconds;
    setPosition(seconds);
    onScrubbed?.(seconds);
  };

  return (
    <div className="video">
      <div className="video__stage">
        {!playback.ready || failed ? (
          <div
            style={{
              aspectRatio: "16 / 9",
              display: "grid",
              placeItems: "center",
              alignContent: "center",
              gap: "var(--s-3)",
              color: "#c3c2b7",
              fontSize: "var(--t-small)",
              padding: "var(--s-5)",
              textAlign: "center",
            }}
          >
            {playback.state === "building" || playback.state === "checking" ? (
              <>
                <span
                  className="skeleton"
                  style={{ width: 180, height: 6, borderRadius: 999, opacity: 0.5 }}
                />
                <span>Preparing this recording for playback…</span>
                <span style={{ fontSize: "var(--t-micro)", opacity: 0.7, maxWidth: "44ch" }}>
                  The capture is stored in a format browsers cannot decode, so a playable copy is
                  being made once. Frame analysis above already works.
                </span>
              </>
            ) : (
              <span style={{ maxWidth: "46ch" }}>
                {failed
                  ? "The browser could not decode this recording."
                  : playback.detail || "Playback is unavailable for this recording."}
              </span>
            )}
          </div>
        ) : (
          <video
            ref={videoRef}
            src={api.videoUrl(sessionId, camera)}
            preload="metadata"
            playsInline
            onError={() => setFailed(true)}
            onLoadedMetadata={(event) => {
              const node = event.currentTarget;
              setLength(node.duration || 0);
              if (selectedSec !== null && Number.isFinite(selectedSec)) {
                node.currentTime = selectedSec;
                lastSeek.current = selectedSec;
              }
            }}
            onTimeUpdate={(event) => {
              const at = event.currentTarget.currentTime;
              setPosition(at);
              if (follow && playing) onScrubbed?.(at);
            }}
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
          />
        )}
        <div className="video__bar">
          <button
            type="button"
            className="iconbtn"
            onClick={toggle}
            aria-label={playing ? "Pause" : "Play"}
            disabled={failed || !playback.ready}
          >
            {playing ? <IconPause size={15} /> : <IconPlay size={15} />}
          </button>
          <span className="micro num muted" style={{ minWidth: 42 }}>
            {clock(position)}
          </span>
          <input
            className="scrub"
            type="range"
            min={0}
            max={Math.max(length, 0.1)}
            step={0.05}
            value={Math.min(position, length || 0)}
            onChange={(event) => scrub(Number(event.target.value))}
            aria-label="Scrub the recording"
            disabled={failed || !playback.ready || !length}
          />
          <span className="micro num muted" style={{ minWidth: 42 }}>
            {clock(length)}
          </span>
          <button
            type="button"
            className="iconbtn"
            onClick={() => selectedSec !== null && scrub(selectedSec)}
            aria-label="Jump to the selected frame"
            title="Jump to the selected frame"
            disabled={failed || !playback.ready || selectedSec === null}
          >
            <IconTarget size={15} />
          </button>
        </div>
      </div>

      <div className="stack stack-4">
        {cameras.length > 1 && <div className="stack stack-2">
          <span className="micro muted" style={{ textTransform: "uppercase", letterSpacing: "0.05em" }}>
            Camera
          </span>
          <div className="stack stack-1">
            {cameras.map((item) => (
              <button
                key={item.camera_index}
                type="button"
                className="navitem"
                aria-current={item.camera_index === camera ? "page" : undefined}
                onClick={() => onCameraChange(item.camera_index)}
              >
                {humanise(roles[item.camera_index] ?? `camera ${item.camera_index + 1}`)}
              </button>
            ))}
          </div>
        </div>}

        <div className="stack stack-2">
          <span className="micro muted" style={{ textTransform: "uppercase", letterSpacing: "0.05em" }}>
            Timeline link
          </span>
          <Segmented
            label="Timeline link"
            value={follow ? "follow" : "manual"}
            onChange={(value) => setFollow(value === "follow")}
            options={[
              { value: "manual", label: "Seek only" },
              { value: "follow", label: "Follow playback" },
            ]}
          />
          <p className="micro muted">
            {follow
              ? "Playing the video moves the timeline selection and reloads the frame analysis."
              : "Choosing a moment on the timeline seeks this player. Playback leaves the selection alone."}
          </p>
        </div>
      </div>
    </div>
  );
}
