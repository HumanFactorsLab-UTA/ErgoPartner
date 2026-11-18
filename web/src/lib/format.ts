import type { BandStatus } from "./types";

/** m:ss — how a reviewer scrubs a video, not decimal seconds. */
export function clock(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds) || seconds < 0) {
    return "Unavailable";
  }
  const total = Math.round(seconds);
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

/** A duration in words, for cards where "2 min 26 s" beats "146.03". */
export function duration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "Unavailable";
  const total = Math.round(seconds);
  const minutes = Math.floor(total / 60);
  const rest = total % 60;
  if (minutes === 0) return `${rest}s`;
  return rest === 0 ? `${minutes}m` : `${minutes}m ${rest}s`;
}

export function number(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "Unavailable";
  return value.toFixed(digits);
}

export function percent(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "Unavailable";
  return `${value.toFixed(digits)}%`;
}

export function humanise(value: string | null | undefined): string {
  if (!value) return "Unavailable";
  const text = value.replace(/_/g, " ").trim();
  return text.charAt(0).toUpperCase() + text.slice(1);
}

export function statusVar(status: BandStatus | "none" | null | undefined): string {
  switch (status) {
    case "good":
      return "var(--good)";
    case "warning":
      return "var(--warning)";
    case "serious":
      return "var(--serious)";
    case "critical":
      return "var(--critical)";
    default:
      return "var(--ink-muted)";
  }
}

export function statusSoft(status: BandStatus | "none" | null | undefined): string {
  switch (status) {
    case "good":
      return "var(--good-soft)";
    case "warning":
      return "var(--warning-soft)";
    case "serious":
      return "var(--serious-soft)";
    case "critical":
      return "var(--critical-soft)";
    default:
      return "var(--surface-sunken)";
  }
}

export function statusInk(status: BandStatus | "none" | null | undefined): string {
  switch (status) {
    case "good":
      return "var(--good-ink)";
    case "warning":
      return "var(--warning-ink)";
    case "serious":
      return "var(--serious-ink)";
    case "critical":
      return "var(--critical-ink)";
    default:
      return "var(--ink-secondary)";
  }
}
