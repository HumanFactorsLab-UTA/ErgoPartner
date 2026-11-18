import { useState } from "react";
import { IconPdf } from "./Icons";
import { ApiError, api } from "../lib/api";

/**
 * Render the session's PDF report, then hand it to the browser.
 *
 * Rendering runs a headless browser server-side and takes seconds, so the
 * request is not fire-and-forget: the button stays busy until the file exists,
 * and a failed render says why in place rather than opening a blank tab.
 */
export function ReportButton({
  sessionId,
  className = "btn btn--ghost",
  disabled = false,
}: {
  sessionId: string;
  className?: string;
  disabled?: boolean;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  return (
    <>
      <button
        type="button"
        className={className}
        disabled={busy || disabled}
        title="Render a PDF assessment report for this analysis"
        onClick={() => {
          setBusy(true);
          setError(null);
          api
            .generateReport(sessionId)
            .then(() => {
              window.open(api.reportDownloadUrl(sessionId), "_blank", "noopener");
            })
            .catch((cause: unknown) => {
              setError(
                cause instanceof ApiError || cause instanceof Error
                  ? cause.message
                  : "The report could not be generated",
              );
            })
            .finally(() => setBusy(false));
        }}
      >
        <IconPdf size={14} />
        {busy ? "Rendering…" : "PDF report"}
      </button>
      {error ? (
        <span className="micro" role="alert" style={{ color: "var(--critical-ink)" }}>
          {error}
        </span>
      ) : null}
    </>
  );
}
