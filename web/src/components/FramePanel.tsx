import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import { humanise, number, statusInk, statusSoft, statusVar } from "../lib/format";
import { BandBadge, Skeleton, Stat, Tabs } from "./ui";
import { IconArrowLeft, IconArrowRight, IconFrames, IconTarget } from "./Icons";
import type { Driver, FrameDetail } from "../lib/types";

/**
 * One camera view. The image is swapped by changing its src, and the previous
 * frame is held at reduced opacity while the next decodes, so stepping through
 * frames reads as an update rather than three boxes blinking to empty.
 */
function CameraView({
  sessionId,
  frame,
  cameraIndex,
  role,
  region,
  sourceFrame,
  person,
  onOpen,
}: {
  sessionId: string;
  frame: number;
  cameraIndex: number;
  role: string;
  region: string;
  sourceFrame: number | null;
  person: string | null;
  onOpen: () => void;
}) {
  const src = api.frameImage(sessionId, frame, cameraIndex, region, 720, true, person);
  const cameraLabel = role.toLowerCase() === "camera" ? "Camera" : `${humanise(role)} camera`;
  const [loaded, setLoaded] = useState(false);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    setLoaded(false);
    setFailed(false);
  }, [src]);

  return (
    <figure className="frameview" data-loading={!loaded && !failed} style={{ margin: 0 }}>
      {failed ? (
        <div
          style={{
            aspectRatio: "16 / 9",
            display: "grid",
            placeItems: "center",
            color: "var(--ink-muted)",
            fontSize: "var(--t-micro)",
            padding: "var(--s-3)",
            textAlign: "center",
          }}
        >
          This camera&rsquo;s source frame is unavailable
        </div>
      ) : (
        <button
          type="button"
          className="frameview__imagebtn"
          onClick={onOpen}
          aria-label={`Enlarge ${cameraLabel} at frame ${frame}`}
        >
          <img
            src={src}
            alt={`${cameraLabel} at frame ${frame}`}
            data-loading={!loaded}
            /* Eager, not lazy: these three views are the point of this section, and
               selecting a moment on the timeline has to update them at once. Lazy
               loading deferred them until the section scrolled into view, so a
               click would appear to do nothing until the user went looking. */
            loading="eager"
            fetchPriority="high"
            decoding="async"
            onLoad={() => setLoaded(true)}
            onError={() => setFailed(true)}
          />
          <span className="frameview__zoomhint">Click to zoom</span>
        </button>
      )}
      <figcaption className="frameview__cap">
        <span style={{ textTransform: "capitalize" }}>{humanise(role)}</span>
        <span className="row row-2">
          {region ? <span className="frameview__region">{humanise(region)}</span> : null}
          <span className="num">frame {sourceFrame ?? frame}</span>
        </span>
      </figcaption>
    </figure>
  );
}

function FrameLightbox({
  sessionId,
  frame,
  cameras,
  region,
  person,
  initialCamera,
  initialOverlay,
  onCameraChange,
  onClose,
}: {
  sessionId: string;
  frame: number;
  cameras: FrameDetail["cameras"];
  region: string;
  person: string | null;
  initialCamera: number | null;
  initialOverlay: boolean;
  onCameraChange: (camera: number) => void;
  onClose: () => void;
}) {
  const [focused, setFocused] = useState<number | null>(initialCamera);
  const [zoom, setZoom] = useState(1);
  const [overlay, setOverlay] = useState(initialOverlay);
  const focusCamera = (camera: number) => {
    setFocused(camera);
    onCameraChange(camera);
    setZoom(1);
  };
  const dialogRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const previousOverflow = document.body.style.overflow;
    const previousFocus =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;
    document.body.style.overflow = "hidden";
    const key = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
        return;
      }
      if (event.key !== "Tab" || !dialogRef.current) return;
      const focusable = Array.from(
        dialogRef.current.querySelectorAll<HTMLElement>(
          'button:not(:disabled), [href], input:not(:disabled), [tabindex]:not([tabindex="-1"])',
        ),
      );
      if (!focusable.length) {
        event.preventDefault();
        dialogRef.current.focus();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && (document.activeElement === first || document.activeElement === dialogRef.current)) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", key);
    dialogRef.current?.focus();
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", key);
      previousFocus?.focus();
    };
  }, [onClose]);

  const visible =
    focused === null
      ? cameras
      : cameras.filter((camera) => camera.cameraIndex === focused);

  return (
    <div className="frame-lightbox" role="dialog" aria-modal="true" aria-label="Camera evidence viewer">
      <button
        type="button"
        className="frame-lightbox__backdrop"
        onClick={onClose}
        aria-label="Close camera viewer"
      />
      <section className="frame-lightbox__dialog" ref={dialogRef} tabIndex={-1}>
        <header className="frame-lightbox__head">
          <div>
            <p className="eyebrow">{cameras.length > 1 ? "Synchronized evidence" : "Camera evidence"}</p>
            <h2 className="num">Frame {frame}</h2>
          </div>
          <div className="frame-lightbox__tools">
            {cameras.length > 1 && <button
              type="button"
              className="btn"
              aria-pressed={focused === null}
              onClick={() => {
                setFocused(null);
                setZoom(1);
              }}
            >
              <IconFrames size={15} />
              Compare all {cameras.length}
            </button>}
            <button
              type="button"
              className="btn"
              aria-pressed={overlay}
              onClick={() => setOverlay((value) => !value)}
            >
              Pose overlay {overlay ? "on" : "off"}
            </button>
            {focused !== null ? (
              <div className="frame-lightbox__zoom" role="group" aria-label="Image zoom">
                <button
                  type="button"
                  onClick={() => setZoom((value) => Math.max(1, value - 0.25))}
                  disabled={zoom <= 1}
                  aria-label="Zoom out"
                >
                  −
                </button>
                <span className="num">{Math.round(zoom * 100)}%</span>
                <button
                  type="button"
                  onClick={() => setZoom((value) => Math.min(3, value + 0.25))}
                  disabled={zoom >= 3}
                  aria-label="Zoom in"
                >
                  +
                </button>
              </div>
            ) : null}
            <button type="button" className="frame-lightbox__close" onClick={onClose}>
              Close
            </button>
          </div>
        </header>

        {cameras.length > 1 && <div className="frame-lightbox__camera-tabs" role="group" aria-label="Camera view">
          {cameras.map((camera) => (
            <button
              type="button"
              key={camera.cameraIndex}
              aria-pressed={focused === camera.cameraIndex}
              onClick={() => {
                focusCamera(camera.cameraIndex);
              }}
            >
              {humanise(camera.role)}
            </button>
          ))}
        </div>}

        <div className="frame-lightbox__stage" data-grid={focused === null}>
          {visible.map((camera) => (
            <figure
              key={camera.cameraIndex}
              onClick={() => {
                if (focused === null) focusCamera(camera.cameraIndex);
              }}
            >
              <div
                className="frame-lightbox__scroll"
                data-zoomed={focused !== null && zoom > 1}
              >
                <img
                  src={api.frameImage(
                    sessionId,
                    frame,
                    camera.cameraIndex,
                    region,
                    1920,
                    overlay,
                    person,
                  )}
                  alt={`${humanise(camera.role)} evidence at frame ${frame}`}
                  style={
                    focused !== null && zoom > 1
                      ? {
                          width: `${zoom * 100}%`,
                          height: "auto",
                          maxWidth: "none",
                          maxHeight: "none",
                        }
                      : undefined
                  }
                />
              </div>
              <figcaption>
                <strong>{humanise(camera.role)}</strong>
                <span className="num">source frame {camera.sourceFrame ?? frame}</span>
                {focused === null ? <span>Click to inspect</span> : null}
              </figcaption>
            </figure>
          ))}
        </div>
      </section>
    </div>
  );
}

function DriverRow({ driver }: { driver: Driver }) {
  const { label, score, maximum, pointsAdded: points, measurement, valueDeg } = driver;
  return (
    <div className="driver-row">
      <div className="stack stack-1 driver-row__label">
        <span className="small" style={{ fontWeight: 580 }}>{label}</span>
        {measurement && valueDeg !== null ? (
          <span className="micro muted">
            {humanise(measurement)} {number(valueDeg, 0)}&deg;
          </span>
        ) : null}
      </div>
      <span className="driver-row__track" aria-hidden="true">
        <span
          className="driver-row__fill"
          style={{ width: `${Math.max(4, (score / Math.max(1, maximum)) * 100)}%` }}
        />
      </span>
      <div className="row row-3 driver-row__values">
        <span className="micro muted num">
          {score} of {maximum}
        </span>
        <span
          className="badge"
          style={{
            background: points > 0 ? "var(--accent-soft)" : "var(--surface-sunken)",
            color: points > 0 ? "var(--accent-hover)" : "var(--ink-muted)",
            borderColor: points > 0 ? "var(--accent-border)" : "var(--line)",
          }}
        >
          {points > 0 ? `+${points}` : "no change"}
        </span>
      </div>
    </div>
  );
}

export function FramePanel({
  sessionId,
  detail,
  loading,
  region,
  person,
  typicalScore,
  onRegionChange,
  onPrevious,
  onNext,
  canPrevious = true,
  canNext = true,
  multi,
  openEvidence = null,
  onCameraChange,
  onComparisonOpened,
}: {
  sessionId: string;
  detail: FrameDetail | null;
  loading: boolean;
  region: string;
  person: string | null;
  typicalScore?: number | null;
  onRegionChange: (region: string) => void;
  onPrevious?: () => void;
  onNext?: () => void;
  canPrevious?: boolean;
  canNext?: boolean;
  multi: boolean;
  openEvidence?: {frame: number; camera?: number | null; overlay?: boolean; person?: string | null; method?: string | null} | null;
  onCameraChange: (camera: number) => void;
  onComparisonOpened?: () => void;
}) {
  const [tab, setTab] = useState<"drivers" | "angles" | "sides">("drivers");
  const [lightboxCamera, setLightboxCamera] = useState<number | "all" | null>(null);
  const [initialOverlay, setInitialOverlay] = useState(true);
  const closeEvidence = useCallback(() => setLightboxCamera(null), []);

  useEffect(() => {
    if (!detail || !openEvidence || openEvidence.frame !== detail.frame ||
        (openEvidence.person && openEvidence.person !== detail.person) ||
        (openEvidence.method && openEvidence.method !== detail.method.toLowerCase())) return;
    setLightboxCamera(openEvidence.camera ?? (multi ? "all" : 0));
    setInitialOverlay(openEvidence.overlay ?? true);
    onComparisonOpened?.();
  }, [detail, openEvidence, multi, onComparisonOpened]);

  if (!detail) {
    return (
      <div className="stack stack-3">
        <Skeleton height={20} width={260} />
        <div className="frames">
          <Skeleton height={150} radius="var(--r-md)" />
          {multi && <Skeleton height={150} radius="var(--r-md)" />}
        </div>
      </div>
    );
  }

  const frameCameras = detail.cameras.length
    ? detail.cameras
    : [{ cameraIndex: 0, role: "camera 1", sourceFrame: detail.frame, timestampSec: null, syncDeltaSec: null }];
  const cameras = multi ? frameCameras : frameCameras.slice(0, 1).map(camera => ({ ...camera, role: "Camera" }));

  const postural = detail.drivers.filter((d) => d.kind === "posture");
  const task = detail.drivers.filter((d) => d.kind === "task");
  const cameraAngles = detail.anglesByCamera ?? {};
  const angleCameras = cameras.filter((camera) => String(camera.cameraIndex) in cameraAngles);
  const compareViews = multi && angleCameras.length > 1;
  const angleRows = Array.from(new Set([
    ...Object.keys(detail.angles),
    ...(compareViews ? Object.values(cameraAngles).flatMap(Object.keys) : []),
  ])).sort();
  const scoreDelta =
    detail.score !== null && typicalScore !== null && typicalScore !== undefined
      ? detail.score - typicalScore
      : null;
  const sideComponents = Array.from(
    new Set([
      ...Object.keys(detail.sides.left.components),
      ...Object.keys(detail.sides.right.components),
    ]),
  ).sort();
  const sideComponentMaximum = Math.max(
    ...sideComponents.flatMap((component) => [
      detail.sides.left.components[component] ?? 0,
      detail.sides.right.components[component] ?? 0,
    ]),
    1,
  );

  return (
    <div className="frame-inspector" data-updating={loading}>
      <div className="frame-inspector__toolbar">
        <div className="frame-inspector__stepper" aria-label="Step through scored observations">
          <button
            type="button"
            className="iconbtn"
            onClick={onPrevious}
            disabled={!onPrevious || !canPrevious}
            aria-label="Previous scored observation"
          >
            <IconArrowLeft size={15} />
          </button>
          <div className="frame-inspector__time">
            <span className="micro muted">Selected moment</span>
            <strong className="num">{detail.clock}</strong>
            <span className="micro muted num">frame {detail.frame}</span>
          </div>
          <button
            type="button"
            className="iconbtn"
            onClick={onNext}
            disabled={!onNext || !canNext}
            aria-label="Next scored observation"
          >
            <IconArrowRight size={15} />
          </button>
        </div>

        <div className="selected-score">
          <Stat
            label={`${detail.method} now`}
            value={detail.score ?? "—"}
            status={detail.band?.status}
            emphasis
          />
          <span className="selected-score__divider" />
          <Stat label="Task typical" value={number(typicalScore, 1)} />
          {scoreDelta !== null ? (
            <span className="selected-score__delta num" data-positive={scoreDelta > 0}>
              {scoreDelta > 0 ? "+" : ""}
              {number(scoreDelta, 1)}
            </span>
          ) : null}
        </div>

        <div className="frame-inspector__badges">
          <BandBadge band={detail.band} />
          {detail.selectedSide ? (
            <span className="badge badge--neutral">{humanise(detail.selectedSide)} side reported</span>
          ) : null}
          {detail.quality.status ? (
            <span className="badge badge--neutral" title={detail.quality.warnings.join(" · ")}>
              Evidence {humanise(detail.quality.status)}
            </span>
          ) : null}
          <button
            type="button"
            className="btn btn--ghost"
            onClick={() => { setInitialOverlay(true); setLightboxCamera(multi ? "all" : 0); }}
          >
            <IconFrames size={14} />
            {multi ? 'Open comparison' : 'Open camera evidence'}
          </button>
        </div>
      </div>

      {detail.band ? (
        <p
          className="small"
          style={{
            padding: "var(--s-3) var(--s-4)",
            borderRadius: "var(--r-md)",
            background: statusSoft(detail.band.status),
            color: statusInk(detail.band.status),
            borderLeft: `3px solid ${statusVar(detail.band.status)}`,
          }}
        >
          <strong>{detail.band.label}.</strong> {detail.band.action}.
          {detail.selectionReason ? ` ${detail.selectionReason}` : ""}
        </p>
      ) : null}

      <div className="region-control">
        <span className="region-control__label">
          <IconTarget size={14} />
          Camera overlay
        </span>
        <div className="region-control__options" role="group" aria-label="Highlight a body region">
          <button
            type="button"
            aria-pressed={!region}
            onClick={() => onRegionChange("")}
          >
            Full body
          </button>
          {postural.map((driver) => (
            <button
              type="button"
              key={driver.region}
              aria-pressed={region === driver.region}
              onClick={() => onRegionChange(driver.region)}
            >
              {driver.label}
            </button>
          ))}
        </div>
      </div>

      <div className="frames">
        {cameras.map((camera) => (
          <CameraView
            key={camera.cameraIndex}
            sessionId={sessionId}
            frame={detail.frame}
            cameraIndex={camera.cameraIndex}
            role={camera.role}
            region={region}
            sourceFrame={camera.sourceFrame}
            person={person}
            onOpen={() => { setInitialOverlay(true); setLightboxCamera(camera.cameraIndex); onCameraChange(camera.cameraIndex); }}
          />
        ))}
      </div>

      <div className="evidence-line">
        <span>
          Visibility <strong className="num">{number(detail.quality.meanVisibility, 2)}</strong>
        </span>
        {(multi || !/^(front|side|iso|oblique)(?:[_ -]view)?$/i.test(detail.quality.cameraView ?? "")) && <span>
          View suitability <strong>{humanise(detail.quality.cameraView)}</strong>
        </span>}
        <span>
          Wrist source <strong>{humanise(detail.quality.wristSource)}</strong>
        </span>
        {detail.quality.warnings.length ? (
          <details className="evidence-line__warning">
            <summary>{detail.quality.warnings.length} evidence warning{detail.quality.warnings.length === 1 ? "" : "s"}</summary>
            <ul>{detail.quality.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul>
          </details>
        ) : null}
      </div>

      <div className="frame-detail">
        <Tabs
          value={tab}
          onChange={setTab}
          tabs={[
            { value: "drivers", label: "What drove this score" },
            { value: "angles", label: `Joint angles (${angleRows.length})` },
            { value: "sides", label: "Left / right" },
          ]}
        />

        {tab === "drivers" ? (
          <div className="stack stack-3 tab-panel" key="drivers">
            <p className="micro muted">
              Points each part adds: set that component back to its neutral value, run{" "}
              {detail.method}&rsquo;s own tables again, and see how far the score falls.
            </p>
            <div>
              {postural.map((driver) => (
                <DriverRow key={driver.region} driver={driver} />
              ))}
            </div>
            {task.length ? (
              <div>
                <p className="micro muted" style={{ marginBottom: "var(--s-1)" }}>
                  Task conditions
                </p>
                {task.map((driver) => (
                  <DriverRow key={driver.region} driver={driver} />
                ))}
              </div>
            ) : null}
          </div>
        ) : null}

        {tab === "angles" ? (
          <div className="dt__scroll tab-panel" key="angles">
            <table className="dt">
              {compareViews && <caption>Per-camera readings before view selection and condition smoothing</caption>}
              <thead>
                <tr>
                  <th>Measurement</th>
                  <th className="num">Value</th>
                  {multi && <th>Selected view</th>}
                  {compareViews && angleCameras.map((camera) => (
                    <th key={camera.cameraIndex} className="num">{humanise(camera.role)}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {angleRows.map((name) => (
                  <tr key={name}>
                    <td>{humanise(name)}</td>
                    <td className="num">{detail.angles[name] == null ? "—" : `${number(detail.angles[name], 1)}°`}</td>
                    {multi && <td className="muted">{humanise(detail.angleSources[name] ?? null)}</td>}
                    {compareViews && angleCameras.map((camera) => {
                      const reading = cameraAngles[String(camera.cameraIndex)][name];
                      return <td key={camera.cameraIndex} className="num">{reading == null ? "—" : `${number(reading, 1)}°`}</td>;
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}

        {tab === "sides" ? (
          <div className="side-compare tab-panel" key="sides">
            {(["left", "right"] as const).map((side) => {
              const sideDetail = detail.sides[side];
              return (
                <article
                  className="side-card"
                  data-reported={detail.selectedSide === side}
                  key={side}
                >
                  <div className="side-card__head">
                    <div>
                      <span className="micro muted">Calculated side</span>
                      <h3 style={{ textTransform: "capitalize" }}>{side}</h3>
                    </div>
                    <div className="side-card__score">
                      <strong className="num">{sideDetail.score ?? "—"}</strong>
                      {detail.selectedSide === side ? (
                        <span className="badge badge--accent">reported</span>
                      ) : null}
                    </div>
                  </div>
                  <p className="micro muted side-card__validity">
                    {humanise(sideDetail.validity)} validity
                  </p>
                  <div className="side-card__components">
                    {sideComponents.map((component) => {
                      const value = sideDetail.components[component];
                      return (
                        <div className="side-component" key={component}>
                          <span>{humanise(component)}</span>
                          <span className="side-component__track" aria-hidden="true">
                            <span
                              style={{ width: `${((value ?? 0) / sideComponentMaximum) * 100}%` }}
                            />
                          </span>
                          <strong className="num">{value ?? "—"}</strong>
                        </div>
                      );
                    })}
                  </div>
                  <div className="side-card__foot">
                    {sideDetail.missing.length ? (
                      <span>Missing: {sideDetail.missing.map(humanise).join(", ")}</span>
                    ) : (
                      <span>No required inputs missing</span>
                    )}
                    {sideDetail.invalid.length ? (
                      <span>Invalid: {sideDetail.invalid.map(humanise).join(", ")}</span>
                    ) : null}
                  </div>
                </article>
              );
            })}
          </div>
        ) : null}
      </div>
      {lightboxCamera !== null ? (
        <FrameLightbox
          key={`${detail.frame}-${lightboxCamera}-${initialOverlay}`}
          initialOverlay={initialOverlay}
          onCameraChange={onCameraChange}
          sessionId={sessionId}
          frame={detail.frame}
          cameras={cameras}
          region={region}
          person={person}
          initialCamera={lightboxCamera === "all" ? null : lightboxCamera}
          onClose={closeEvidence}
        />
      ) : null}
    </div>
  );
}
