import { useState } from "react";
import { get, useAsync } from "../lib/api";
import { Field } from "./WorkflowParts";

interface ModelCatalog {
  models: { tag: string; size: number | null; capabilities: string[] }[];
  configured: string;
  inherited: string;
  selection: string | null;
  detail: string;
}

export function VisionModelSelect({
  value,
  onChange,
  disabled = false,
  workstation = false,
  sessionId,
  editingSessionDefault = false,
}: {
  value: string | null;
  onChange: (model: string | null) => void;
  disabled?: boolean;
  workstation?: boolean;
  sessionId?: string;
  editingSessionDefault?: boolean;
}) {
  const [revision, setRevision] = useState(0);
  const catalog = useAsync(
    (signal) => {
      const params = new URLSearchParams();
      if (sessionId) params.set("sessionId", sessionId);
      if (revision) params.set("refresh", "true");
      return get<ModelCatalog>(`/api/assistant/models?${params}`, signal);
    },
    [sessionId, revision],
  );
  const options = catalog.data?.models ?? [];
  const fallback = workstation
    ? catalog.data?.configured
    : ((editingSessionDefault ? null : catalog.data?.selection) ?? catalog.data?.inherited);
  const defaultLabel = workstation
    ? "Configured default"
    : sessionId
      ? "Task default"
      : "Workstation default";
  const selected = options.find((item) => item.tag === (value ?? fallback));
  const detail =
    catalog.error ||
    catalog.data?.detail ||
    (selected && !selected.capabilities.includes("tools")
      ? "This model supports visual review. Assessment chat requires a model with tool support."
      : catalog.data && !selected
        ? "Choose an installed vision model to run a visual review."
        : "");
  return (
    <div className="stack stack-2">
      <Field label="Local vision model">
        <select
          disabled={disabled || catalog.loading}
          value={value ?? ""}
          onChange={(event) => onChange(event.target.value || null)}
        >
          <option value="">
            {defaultLabel}
            {fallback ? ` (${fallback})` : ""}
          </option>
          {value && !options.some((model) => model.tag === value) && (
            <option value={value}>
              {value}
              {catalog.loading ? "" : " (unavailable)"}
            </option>
          )}
          {options.map((model) => (
            <option key={model.tag} value={model.tag}>
              {model.tag}
              {model.size ? ` (${(model.size / 1e9).toFixed(1)} GB on disk)` : ""}
            </option>
          ))}
        </select>
      </Field>
      {detail && (
        <p className="micro muted" role="status">
          {detail}
        </p>
      )}
      <div>
        <button
          className="btn btn--ghost btn--sm"
          type="button"
          disabled={disabled || catalog.loading}
          onClick={() => setRevision((n) => n + 1)}
        >
          {catalog.loading ? "Checking installed models…" : "Refresh models"}
        </button>
      </div>
    </div>
  );
}
