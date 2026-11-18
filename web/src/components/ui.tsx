import { useEffect, useId, useState, type ReactNode } from "react";
import { IconAlert, IconChevron, IconInfo } from "./Icons";
import { statusInk, statusSoft, statusVar } from "../lib/format";
import type { Band, BandStatus } from "../lib/types";

/**
 * A risk band, always as colour *and* the method's own word for the level.
 *
 * The status hues are deliberately sub-3:1 on this light surface for warning and
 * serious, so a band that travelled as colour alone would be unreadable for some
 * users and meaningless for the rest. The label is not decoration.
 */
export function BandBadge({
  band,
  score,
  size = "md",
}: {
  band: Band | null | undefined;
  score?: number | null;
  size?: "sm" | "md";
}) {
  if (!band) {
    return <span className="badge badge--neutral">No score</span>;
  }
  return (
    <span
      className="badge"
      style={{
        background: statusSoft(band.status),
        color: statusInk(band.status),
        borderColor: statusVar(band.status),
        fontSize: size === "sm" ? "var(--t-micro)" : "var(--t-small)",
        padding: size === "sm" ? "2px 8px" : "3px 10px",
      }}
      title={band.action}
    >
      <span className="badge__dot" style={{ background: statusVar(band.status) }} />
      {score !== undefined && score !== null ? `${score} · ` : ""}
      {band.label}
    </span>
  );
}

export function Skeleton({
  height = 16,
  width = "100%",
  radius,
}: {
  height?: number | string;
  width?: number | string;
  radius?: string;
}) {
  return (
    <div
      className="skeleton"
      style={{ height, width, borderRadius: radius }}
      aria-hidden="true"
    />
  );
}

export function EmptyState({
  title,
  detail,
  action,
}: {
  title: string;
  detail?: string;
  action?: ReactNode;
}) {
  return (
    <div className="state">
      <IconInfo size={22} className="state__icon" />
      <h3>{title}</h3>
      {detail ? <p className="small muted" style={{ maxWidth: "48ch" }}>{detail}</p> : null}
      {action}
    </div>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="state state--error" role="alert">
      <IconAlert size={22} />
      <h3>This did not load</h3>
      <p className="small" style={{ maxWidth: "52ch" }}>
        {message}
      </p>
      {onRetry ? (
        <button className="btn" onClick={onRetry} type="button">
          Try again
        </button>
      ) : null}
    </div>
  );
}

/**
 * Progressive disclosure. Closed by default, because the point of these panels
 * is that a reviewer meets the answer first and the working only when they ask.
 */
export function Disclosure({
  summary,
  hint,
  children,
  defaultOpen = false,
}: {
  summary: string;
  hint?: string;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const [rendered, setRendered] = useState(defaultOpen);
  const [expanded, setExpanded] = useState(defaultOpen);
  const id = useId();

  useEffect(() => {
    let firstFrame = 0;
    let secondFrame = 0;
    let timer = 0;
    if (open) {
      setRendered(true);
      firstFrame = window.requestAnimationFrame(() => {
        secondFrame = window.requestAnimationFrame(() => setExpanded(true));
      });
    } else {
      setExpanded(false);
      if (rendered) {
        timer = window.setTimeout(() => setRendered(false), 240);
      }
    }
    return () => {
      if (firstFrame) window.cancelAnimationFrame(firstFrame);
      if (secondFrame) window.cancelAnimationFrame(secondFrame);
      if (timer) window.clearTimeout(timer);
    };
  }, [open, rendered]);

  return (
    <section className="disclosure" data-open={open}>
      <button
        type="button"
        className="disclosure__summary"
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen((value) => !value)}
      >
        <IconChevron size={14} className="disclosure__chev" />
        {summary}
        {hint ? <span className="disclosure__hint">{hint}</span> : null}
      </button>
      {rendered ? (
        <div
          className="disclosure__region"
          data-open={expanded}
          id={id}
          aria-hidden={!open}
        >
          <div className="disclosure__clip">
            <div className="disclosure__body">{children}</div>
          </div>
        </div>
      ) : null}
    </section>
  );
}

export function Segmented<T extends string>({
  options,
  value,
  onChange,
  label,
}: {
  options: { value: T; label: string }[];
  value: T;
  onChange: (value: T) => void;
  label: string;
}) {
  return (
    <div className="segmented" role="group" aria-label={label}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          aria-pressed={option.value === value}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

export function Tabs<T extends string>({
  tabs,
  value,
  onChange,
}: {
  tabs: { value: T; label: string }[];
  value: T;
  onChange: (value: T) => void;
}) {
  return (
    <div className="tabs" role="tablist">
      {tabs.map((tab) => (
        <button
          key={tab.value}
          role="tab"
          type="button"
          aria-selected={tab.value === value}
          onClick={() => onChange(tab.value)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}

/** A labelled figure. Small by design — an oversized KPI card says nothing extra. */
export function Stat({
  label,
  value,
  hint,
  status,
  emphasis = false,
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  status?: BandStatus | "none";
  emphasis?: boolean;
}) {
  return (
    <div className="stack stack-1" style={{ minWidth: 0 }}>
      <span className="micro muted" style={{ letterSpacing: "0.04em" }}>
        {label}
      </span>
      <span
        key={typeof value === "string" || typeof value === "number" ? value : undefined}
        className="stat__value num"
        style={{
          fontSize: emphasis ? "1.75rem" : "1.25rem",
          fontWeight: 640,
          lineHeight: 1.1,
          letterSpacing: "-0.02em",
          color: status ? statusInk(status) : "var(--ink)",
        }}
      >
        {value}
      </span>
      {hint ? <span className="micro muted">{hint}</span> : null}
    </div>
  );
}

/**
 * A button that asks once before doing something it cannot undo.
 *
 * Re-running an analysis overwrites a completed result. The processor archives
 * the previous one, so nothing is destroyed, but the thing on screen is replaced
 * and a long batch cannot be called back once it starts -- so the click that
 * starts it should not be the same shape as the click that opens a chart.
 *
 * The confirmation is inline rather than a dialog. A modal for "are you sure"
 * trains people to dismiss modals; a button that changes into its own
 * consequence does not, and it keeps the count of what is about to happen next
 * to the thing being clicked.
 */
export function ConfirmButton({
  children,
  confirmLabel,
  onConfirm,
  className = "btn",
  disabled = false,
  icon,
  title,
}: {
  children: ReactNode;
  confirmLabel: ReactNode;
  onConfirm: () => void;
  className?: string;
  disabled?: boolean;
  icon?: ReactNode;
  title?: string;
}) {
  const [armed, setArmed] = useState(false);

  // An armed button left alone is a trap for the next click, so it disarms
  // itself. Long enough to read the consequence, short enough that it is never
  // still waiting when attention comes back to the page.
  useEffect(() => {
    if (!armed) return;
    const timer = window.setTimeout(() => setArmed(false), 6000);
    return () => window.clearTimeout(timer);
  }, [armed]);

  if (!armed) {
    return (
      <button
        type="button"
        className={className}
        disabled={disabled}
        title={title}
        onClick={() => setArmed(true)}
      >
        {icon}
        {children}
      </button>
    );
  }

  return (
    <span className="confirm">
      <button
        type="button"
        className="btn btn--danger"
        disabled={disabled}
        onClick={() => {
          setArmed(false);
          onConfirm();
        }}
        autoFocus
      >
        {confirmLabel}
      </button>
      <button type="button" className="btn btn--ghost btn--sm" onClick={() => setArmed(false)}>
        Cancel
      </button>
    </span>
  );
}
