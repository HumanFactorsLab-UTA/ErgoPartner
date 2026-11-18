import type { ReactNode } from "react";
import { useEffect } from "react";

export type TabKey = "overview" | "investigate" | "data" | "assurance" | "findings";

const WORK_TABS = [
  { key: "overview", label: "Summary" },
  { key: "investigate", label: "Posture review" },
  { key: "findings", label: "Findings & report" },
] as const;

export function isTabKey(value: string | null): value is TabKey {
  return value === "assurance" || value === "data" || WORK_TABS.some(tab => tab.key === value);
}

export function WorkTabs({
  value,
  onChange,
  counts,
}: {
  value: TabKey;
  onChange: (next: TabKey) => void;
  counts?: Partial<Record<TabKey, ReactNode>>;
}) {
  // Number keys jump straight to a tab. This is a research tool used for hours
  // at a time; reaching for the mouse to compare the timeline against the
  // driver ranking gets old on the tenth trial.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      const target = event.target as HTMLElement | null;
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
      if (target?.isContentEditable) return;
      const index = Number(event.key) - 1;
      if (Number.isInteger(index) && index >= 0 && index < WORK_TABS.length) {
        event.preventDefault();
        onChange(WORK_TABS[index].key);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onChange]);

  return (
    <div className="worktabs" role="tablist" aria-label="Analysis stages">
      {WORK_TABS.map((tab, index) => (
        <button
          key={tab.key}
          id={`tab-${tab.key}`}
          aria-controls={`panel-${tab.key}`}
          role="tab"
          type="button"
          className="worktab"
          aria-selected={tab.key === value}
          onClick={() => onChange(tab.key)}
          title={`${tab.label} (${index + 1})`}
        >
          <span className="worktab__label">
            {tab.label}
            {counts?.[tab.key] ? <span className="worktab__count">{counts[tab.key]}</span> : null}
          </span>
        </button>
      ))}
    </div>
  );
}

/**
 * One tab's contents.
 *
 * Mounted on first visit and then kept mounted but hidden. Two reasons: the
 * replay canvas and the frame table measure their own width, which is zero
 * while hidden, so they must first render visible; and the frame table's paging
 * and the replay's playhead should survive a trip to another tab and back.
 */
export function TabPanel({
  tab,
  active,
  visited,
  children,
}: {
  tab: TabKey;
  active: TabKey;
  visited: boolean;
  children: ReactNode;
}) {
  if (!visited) return null;
  const shown = tab === active;
  const auxiliary = tab === "assurance" || tab === "data";
  return (
    <section
      className="tabpanel"
      id={`panel-${tab}`}
      role={auxiliary ? "region" : "tabpanel"}
      aria-labelledby={auxiliary ? undefined : `tab-${tab}`}
      aria-label={auxiliary ? (tab === "data" ? "Detailed frame calculations" : "Evidence and quality") : undefined}
      hidden={!shown}
    >
      {children}
    </section>
  );
}
