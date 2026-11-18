import { useEffect, useMemo, useState } from "react";
import { api, useAsync } from "../lib/api";
import { clock, humanise, number, statusInk, statusVar } from "../lib/format";
import type { MethodKey, ObservationRow, RiskPeriod } from "../lib/types";
import { IconArrowLeft, IconArrowRight, IconReport, IconTarget } from "./Icons";
import { BandBadge, Disclosure, Skeleton } from "./ui";

function CalculationDetails({ row }: { row: ObservationRow }) {
  const components = Object.entries(row.components);
  const angles = Object.entries(row.angles)
    .filter(([, value]) => typeof value === "number" && Number.isFinite(value))
    .slice(0, 12);
  return (
    <div className="observation-calculations">
      <div>
        <p className="micro muted">Method calculation</p>
        {components.length ? (
          <dl>
            {components.map(([key, value]) => (
              <div key={key}>
                <dt>{humanise(key)}</dt>
                <dd className="num">{number(value, 0)}</dd>
              </div>
            ))}
          </dl>
        ) : (
          <p className="micro muted">No intermediate score was available.</p>
        )}
      </div>
      <div>
        <p className="micro muted">Measured posture</p>
        {angles.length ? (
          <dl>
            {angles.map(([key, value]) => (
              <div key={key}>
                <dt>{humanise(key)}</dt>
                <dd className="num">{number(value, 1)}°</dd>
              </div>
            ))}
          </dl>
        ) : (
          <p className="micro muted">No angle trace was available.</p>
        )}
      </div>
    </div>
  );
}

function ObservationTable({
  rows,
  selectedFrame,
  onSelect,
}: {
  rows: ObservationRow[];
  selectedFrame: number | null;
  onSelect: (frame: number) => void;
}) {
  const [expanded, setExpanded] = useState<number | null>(null);
  return (
    <div className="frame-data-table">
      <div className="frame-data-table__head" role="row">
        <span>Frame / time</span>
        <span>Score</span>
        <span>Classification</span>
        <span>Reported side</span>
        <span>Evidence</span>
        <span aria-hidden="true" />
      </div>
      {rows.map((row) => (
        <div
          className="frame-data-row"
          data-selected={selectedFrame === row.frame}
          data-expanded={expanded === row.frame}
          key={row.frame}
        >
          <button
            type="button"
            className="frame-data-row__select"
            onClick={() => onSelect(row.frame)}
            aria-label={`Inspect frame ${row.frame} at ${row.clock}`}
          >
            <span>
              <strong className="num">#{row.frame}</strong>
              <small className="num">{row.clock}</small>
            </span>
            <strong
              className="frame-data-row__score num"
              style={{ color: row.band ? statusInk(row.band.status) : undefined }}
            >
              {row.score ?? "—"}
            </strong>
            <span>
              <BandBadge band={row.band} size="sm" />
            </span>
            <span className="small">{humanise(row.selectedSide)}</span>
            <span className="frame-data-row__quality">
              <span>{humanise(row.validity)}</span>
              <small className="num">
                {row.meanVisibility === null ? "—" : number(row.meanVisibility, 2)}
              </small>
            </span>
          </button>
          <button
            type="button"
            className="frame-data-row__expand"
            aria-expanded={expanded === row.frame}
            onClick={() => setExpanded((value) => (value === row.frame ? null : row.frame))}
          >
            {expanded === row.frame ? "Hide" : "Calculation"}
          </button>
          {expanded === row.frame ? <CalculationDetails row={row} /> : null}
        </div>
      ))}
    </div>
  );
}

function PeriodSummary({
  periods,
  durationSec,
  selectedFrame,
  onSelect,
}: {
  periods: RiskPeriod[];
  durationSec: number;
  selectedFrame: number | null;
  onSelect: (frame: number) => void;
}) {
  const total = Math.max(durationSec, ...periods.map((period) => period.endSec), 1);
  const priority = periods
    .filter((period) => period.band?.status === "serious" || period.band?.status === "critical")
    .sort((a, b) => b.durationSec - a.durationSec)
    .slice(0, 5);
  return (
    <div className="period-summary">
      <div className="period-rail" aria-label="Official classifications over the recording">
        {periods.map((period) => (
          <button
            type="button"
            key={`${period.index}-${period.startFrame}`}
            className="period-rail__segment"
            data-selected={
              selectedFrame !== null &&
              selectedFrame >= period.startFrame &&
              selectedFrame <= period.endFrame
            }
            style={{
              left: `${(period.startSec / total) * 100}%`,
              width: `${Math.max(0.18, (period.durationSec / total) * 100)}%`,
              background: statusVar(period.band?.status),
            }}
            aria-label={`${period.band?.label ?? "Unscored"} from ${clock(
              period.startSec,
            )} to ${clock(period.endSec)}; peak ${period.peakScore ?? "unscored"}`}
            tabIndex={-1}
            title={`${period.band?.label ?? "Unscored"} · ${clock(period.startSec)}–${clock(
              period.endSec,
            )} · peak ${period.peakScore ?? "—"}`}
            onClick={() => onSelect(period.peakFrame)}
          />
        ))}
      </div>
      <div className="period-axis">
        <span className="num">0:00</span>
        <span className="num">{clock(total / 2)}</span>
        <span className="num">{clock(total)}</span>
      </div>
      <div className="period-priority">
        <div>
          <p className="micro muted">Longest priority periods</p>
          <strong>
            {priority.length
              ? `${priority.length} shown`
              : "No serious or immediate-action periods"}
          </strong>
        </div>
        {priority.map((period) => (
          <button type="button" key={period.index} onClick={() => onSelect(period.peakFrame)}>
            <span
              className="period-priority__swatch"
              style={{ background: statusVar(period.band?.status) }}
            />
            <span>
              <strong>{period.band?.label ?? "Unscored"}</strong>
              <small className="num">
                {clock(period.startSec)}–{clock(period.endSec)}
              </small>
            </span>
            <span className="num">{number(period.durationSec, 1)}s</span>
            <IconArrowRight size={13} />
          </button>
        ))}
      </div>
    </div>
  );
}

export function FrameDataExplorer({
  sessionId,
  method,
  person,
  selectedFrame,
  bands,
  onSelect,
}: {
  sessionId: string;
  method: MethodKey;
  person: string | null;
  selectedFrame: number | null;
  bands: { label: string; status: string }[];
  onSelect: (frame: number) => void;
}) {
  const [band, setBand] = useState("all");
  const [page, setPage] = useState(1);
  useEffect(() => {
    setBand("all");
    setPage(1);
  }, [sessionId, method, person]);
  const observations = useAsync(
    (signal) => api.observations(sessionId, method, page, band, person, signal),
    [sessionId, method, page, band, person],
  );
  const periods = useAsync(
    (signal) => api.riskPeriods(sessionId, method, person, signal),
    [sessionId, method, person],
  );
  const options = useMemo(
    () => [
      { label: "All observations", value: "all" },
      ...bands.map((item) => ({
        label: item.label,
        value: item.label,
      })),
      { label: "Unscored / evidence gap", value: "unscored" },
    ],
    [bands],
  );

  return (
    <div className="frame-data">
      <article className="frame-data__periods">
        <header className="frame-data__head">
          <div>
            <p className="insight-card__kicker">Official action bands</p>
            <h3>Risk and action periods across the task</h3>
            <p className="micro muted">
              Every contiguous classification period is shown, including unscored evidence gaps.
            </p>
          </div>
          <span className="badge badge--neutral">
            {periods.error
              ? "unavailable"
              : periods.data
                ? `${periods.data.periods.length} periods`
                : "loading"}
          </span>
        </header>
        {periods.error ? (
          <div className="frame-data__empty" role="alert">
            <IconTarget size={18} />
            <span>{periods.error}</span>
            <button type="button" className="btn btn--ghost" onClick={periods.reload}>
              Try again
            </button>
          </div>
        ) : !periods.data ? (
          <Skeleton height={180} radius="var(--r-lg)" />
        ) : (
          <PeriodSummary
            periods={periods.data.periods}
            durationSec={periods.data.durationSec}
            selectedFrame={selectedFrame}
            onSelect={onSelect}
          />
        )}
      </article>

      <Disclosure
        summary="Full per-frame calculations"
        hint={
          observations.data
            ? `${observations.data.total.toLocaleString()} observations · research CSV available`
            : "scores, classifications, intermediates, and angles"
        }
        defaultOpen
      >
        <div className="frame-data__toolbar">
          <div className="frame-data__filters" role="group" aria-label="Filter frame calculations">
            {options.map((option) => (
              <button
                type="button"
                key={option.value}
                aria-pressed={band === option.value}
                onClick={() => {
                  setBand(option.value);
                  setPage(1);
                }}
              >
                {option.label}
              </button>
            ))}
          </div>
          <a
            className="btn"
            href={api.frameDataExport(sessionId, method, band, person)}
            download
          >
            <IconReport size={14} />
            Download calculations CSV
          </a>
        </div>

        {observations.error ? (
          <div className="frame-data__empty">
            <IconTarget size={18} />
            <span>{observations.error}</span>
          </div>
        ) : !observations.data ? (
          <div className="stack stack-2">
            <Skeleton height={42} />
            <Skeleton height={58} />
            <Skeleton height={58} />
          </div>
        ) : observations.data.rows.length === 0 ? (
          <div className="frame-data__empty">
            <IconTarget size={18} />
            <span>No observations match this classification.</span>
          </div>
        ) : (
          <>
            <ObservationTable
              rows={observations.data.rows}
              selectedFrame={selectedFrame}
              onSelect={onSelect}
            />
            <footer className="frame-data__pagination">
              <span className="micro muted">
                Page <strong className="num">{observations.data.page}</strong> of{" "}
                <strong className="num">{observations.data.pages}</strong>
              </span>
              <div className="row row-2">
                <button
                  type="button"
                  className="btn btn--ghost"
                  disabled={page <= 1 || observations.loading}
                  onClick={() => setPage((value) => Math.max(1, value - 1))}
                >
                  <IconArrowLeft size={14} />
                  Previous
                </button>
                <button
                  type="button"
                  className="btn btn--ghost"
                  disabled={page >= observations.data.pages || observations.loading}
                  onClick={() => setPage((value) => value + 1)}
                >
                  Next
                  <IconArrowRight size={14} />
                </button>
              </div>
            </footer>
          </>
        )}
      </Disclosure>
    </div>
  );
}
