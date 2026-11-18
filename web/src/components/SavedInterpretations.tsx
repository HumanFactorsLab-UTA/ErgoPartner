import { useEffect, useState } from "react";
import { get, post, useAsync } from "../lib/api";
import { clock } from "../lib/format";
import type { AssistantCitation } from "../lib/types";
import { ErrorState } from "./ui";
import { VisionModelSelect } from "./VisionModelSelect";

interface Evidence {
  frame: number;
  sec: number;
  camera: number;
  person: string | null;
}

interface Reading {
  id: string;
  model: string;
  modelDigest: string | null;
  createdAt: string;
  observations: { id: string; text: string; visibility: string; evidence: Evidence[] }[];
  limitations: string;
  followUp: string;
  selected: string[];
  stale: boolean;
}

type Readings = { interpretations: Reading[] };

export function SavedInterpretations({
  sessionId,
  onCitation,
  onSelectionPending,
}: {
  sessionId: string;
  onCitation: (citation: AssistantCitation) => void;
  onSelectionPending: (pending: boolean) => void;
}) {
  const path = `/api/sessions/${encodeURIComponent(sessionId)}/interpretations`;
  const loaded = useAsync((signal) => get<Readings>(path, signal), [path]);
  const [records, setRecords] = useState<Reading[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [model, setModel] = useState<string | null>(null);
  const [showModels, setShowModels] = useState(false);
  useEffect(() => {
    if (loaded.data) setRecords(loaded.data.interpretations);
  }, [loaded.data]);

  async function update(destination: string, body: unknown, optimistic?: Reading[]) {
    setBusy(true);
    setError("");
    if (optimistic) {
      setRecords(optimistic);
      onSelectionPending(true);
    }
    try {
      const result = await post<Readings>(destination, body);
      setRecords(result.interpretations);
    } catch (err) {
      if (optimistic) setRecords(records);
      setError(err instanceof Error ? err.message : "The interpretation could not be saved.");
    } finally {
      setBusy(false);
      if (optimistic) onSelectionPending(false);
    }
  }

  function toggleObservation(reading: Reading, id: string, include: boolean) {
    const selected = include
      ? [...reading.selected, id]
      : reading.selected.filter((key) => key !== id);
    return update(
      `${path}/${reading.id}/selection`,
      { observationId: id, include },
      records.map((record) => (record.id === reading.id ? { ...record, selected } : record)),
    );
  }

  return (
    <section className="interpretations card stack stack-4" aria-label="AI interpretation">
      <div className="workflow-heading">
        <h2>AI interpretation</h2>
        <div className="row row-2 wrap">
          <button
            type="button"
            className="btn"
            disabled={busy || loaded.loading || !!loaded.error}
            onClick={() => void update(path, { requestedModel: model })}
          >
            Generate interpretation
          </button>
          <button
            type="button"
            className="btn btn--ghost"
            disabled={busy}
            aria-expanded={showModels}
            onClick={() => setShowModels(!showModels)}
          >
            Choose model
          </button>
        </div>
      </div>
      {showModels && (
        <VisionModelSelect
          value={model}
          onChange={setModel}
          sessionId={sessionId}
          disabled={busy}
        />
      )}
      {model && !showModels && <p className="small">Selected model · {model}</p>}
      {loaded.error && <ErrorState message={loaded.error} onRetry={loaded.reload} />}
      {error && (
        <p role="alert" className="workflow-error">
          {error}
        </p>
      )}
      {busy && <p role="status">Updating the saved interpretation…</p>}
      {!records.length && !loaded.error && (
        <p className="small muted">
          {loaded.loading
            ? "Loading saved interpretations…"
            : "Generate a visual interpretation to review observations at their recorded timestamps."}
        </p>
      )}
      {records.map((reading, index) => (
        <details key={reading.id} open={index === 0} className="saved-reading">
          <summary>
            <strong>{reading.model}</strong> · {new Date(reading.createdAt).toLocaleString()}
            {reading.stale ? " · Earlier analysis" : ""}
          </summary>
          <div className="stack stack-4">
            {reading.observations.map((item) => (
              <article key={item.id} className="reading-observation">
                <p>{item.text}</p>
                <div className="reading-evidence">
                  {item.evidence.map((ref) => (
                    <button
                      type="button"
                      key={`${ref.camera}-${ref.frame}`}
                      className="reading-reference"
                      onClick={() =>
                        onCitation({
                          ...ref,
                          method: null,
                          region: null,
                          kind: "observed",
                          view: "raw",
                        })
                      }
                    >
                      {clock(ref.sec)} · Camera {ref.camera + 1} · Frame {ref.frame}
                    </button>
                  ))}
                </div>
                <div className="workflow-heading">
                  <span className="small muted">Visibility · {item.visibility}</span>
                  <label className="setup-check small">
                    <input
                      type="checkbox"
                      checked={reading.selected.includes(item.id)}
                      disabled={busy || reading.stale}
                      onChange={(event) =>
                        void toggleObservation(reading, item.id, event.target.checked)
                      }
                    />
                    Include in PDF report
                  </label>
                </div>
              </article>
            ))}
            <div>
              <h3>Limitations</h3>
              <p>{reading.limitations}</p>
            </div>
            <div>
              <h3>Follow-up</h3>
              <p>{reading.followUp}</p>
            </div>
            {reading.modelDigest && (
              <details>
                <summary>Model details</summary>
                <p className="small reading-digest">{reading.modelDigest}</p>
              </details>
            )}
          </div>
        </details>
      ))}
    </section>
  );
}
