"""Document builder — turns a completed analysis into a PDF assessment report.

Reads the published v4 result bundle for a session, fills an HTML template with
figures taken from it verbatim, and renders the PDF beside the results it came
from. Nothing here computes a score: every number is read from
``video_summary.json``, which the deterministic calculators wrote.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from ergonomics.reba import REBACalculator
from ergonomics.rula import RULACalculator
from paths import PROJECT_ROOT, brand_logo_uri
from reporting import svg_charts

LOGGER = logging.getLogger(__name__)

REPORT_FILENAME = "assessment_report.pdf"
TEMPLATE_PATH = PROJECT_ROOT / "reporting" / "templates"
GENERATOR_SCRIPT = TEMPLATE_PATH / "_pdf_generator.js"
PDF_TIMEOUT_SEC = 120

#: Keep the browser inside node_modules rather than the per-user cache. A
#: Microsoft Store Python sees a virtualized AppData\Local, so a browser
#: installed there by any other shell is invisible to the renderer we spawn.
BROWSERS_IN_PACKAGE = "0"
INSTALL_HINT = (
    "From the project root, run 'npm.cmd install --prefix web' in Windows "
    "PowerShell (or 'npm install --prefix web' on macOS/Linux). The project's "
    "postinstall step downloads Chromium automatically."
)


class ReportError(RuntimeError):
    """Something prevented report generation."""


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

MISSING = "Unavailable"


def _number(value: Any, digits: int = 1) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return MISSING
    if isinstance(value, int) or float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.{digits}f}"


def _percent(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return MISSING
    return f"{value * 100:.0f}%"


def _clock(seconds: Any) -> str:
    if isinstance(seconds, bool) or not isinstance(seconds, int | float):
        return MISSING
    minutes, remainder = divmod(int(round(seconds)), 60)
    return f"{minutes}:{remainder:02d}"


def _risk_class(band: str | None) -> str:
    return {
        "Negligible": "risk-low",
        "Low": "risk-low",
        "Acceptable": "risk-low",
        "Medium": "risk-medium",
        "Investigate": "risk-medium",
        "High": "risk-high",
        "Investigate Soon": "risk-high",
        "Very High": "risk-critical",
        "Investigate Now": "risk-critical",
    }.get(band or "", "")


def _status_class(status: str | None) -> str:
    """The analysis layer names bands by severity; the stylesheet by risk."""

    return {
        "good": "risk-low",
        "warning": "risk-medium",
        "serious": "risk-high",
        "critical": "risk-critical",
    }.get(status or "", "")


def _reba_band(score: Any) -> tuple[str, str, str]:
    entry = REBACalculator.RISK_LEVELS.get(score) if isinstance(score, int) else None
    if entry is None:
        return (MISSING, "", "")
    band, action = entry[0], entry[1]
    return (band, action, _risk_class(band))


def _rula_band(score: Any) -> tuple[str, str, str]:
    entry = RULACalculator.RISK_LEVELS.get(score) if isinstance(score, int) else None
    if entry is None:
        return (MISSING, "", "")
    band, action = entry[0], entry[1]
    return (band, action, _risk_class(band))


# ---------------------------------------------------------------------------
# Data gathering
# ---------------------------------------------------------------------------


def _study_context(session_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Participant and trial metadata, or empty mappings when unavailable.

    The study tree holds the recordings and the demographics; it can be absent
    or moved while the results remain. A report is still worth producing then,
    so a missing study degrades to blank metadata rather than a failure.
    """

    from api.main import STUDIES_ROOT
    from api.sessions import SessionError, find_context, find_trial, parse_session_id

    try:
        ref = parse_session_id(session_id)
        context = find_context(STUDIES_ROOT, f"{ref.study_folder}::{ref.participant_id}")
        trial = find_trial(context, ref.task_folder)
    except (SessionError, OSError) as exc:
        LOGGER.info("No study metadata for %s: %s", session_id, exc)
        return ({}, {})
    return (context, trial)


def _method_block(
    summary: dict[str, Any],
    completion: dict[str, Any],
    band: tuple[str, str, str],
) -> dict[str, Any]:
    peak = summary.get("maximum_score")
    return {
        "present": isinstance(peak, int),
        "peak": _number(peak),
        "peak_sec": _clock(summary.get("maximum_timestamp_sec")),
        "peak_frame": _number(summary.get("maximum_frame_number")),
        "typical": _number(summary.get("duration_weighted_mean")),
        "percentile": _number(summary.get("high_percentile")),
        "percentile_label": _number(summary.get("percentile")),
        "observations": _number(summary.get("observation_count")),
        "sustained": len(summary.get("sustained_high_risk_periods") or []),
        "band": band[0],
        "action": band[1],
        "class": band[2],
        "completion": _percent(completion.get("completion_rate")),
        "scored": _number(completion.get("scored_observations")),
        "total": _number(completion.get("total_observations")),
        "blocker": completion.get("primary_blocker"),
    }


def _rnle_block(rnle: dict[str, Any]) -> dict[str, Any]:
    index = rnle.get("lifting_index")
    withheld: list[str] = []
    for field in rnle.get("missing_inputs") or []:
        withheld.append(f"{field} was not supplied")
    for field in rnle.get("invalid_inputs") or []:
        withheld.append(f"{field} fell outside the equation's valid range")
    for field in rnle.get("unsupported_conditions") or []:
        withheld.append(str(field))

    multipliers = rnle.get("multipliers") or {}
    return {
        "present": isinstance(index, int | float),
        "index": _number(index, 2),
        "rwl": _number(rnle.get("recommended_weight_limit_kg"), 2),
        "risk": rnle.get("risk_level") or MISSING,
        "action": rnle.get("action_level") or "",
        "status": rnle.get("validity_status") or MISSING,
        "withheld": withheld,
        "multipliers": [
            {"name": name, "label": label, "value": _number(multipliers.get(name), 3)}
            for name, label in (
                ("HM", "Horizontal"),
                ("VM", "Vertical"),
                ("DM", "Distance"),
                ("AM", "Asymmetry"),
                ("FM", "Frequency"),
                ("CM", "Coupling"),
            )
        ],
    }


def _timelines(session_id: str) -> dict[str, dict[str, Any]]:
    """The per-method analysis payload the workspace draws, for the charts.

    Charting from this rather than from the summary keeps a figure in the report
    and the same figure on screen showing one number. A method that will not
    load is charted as absent instead of failing the whole report.
    """

    from api import analysis
    from api.sessions import SessionError

    payloads: dict[str, dict[str, Any]] = {}
    for method in ("reba", "rula"):
        try:
            payloads[method] = analysis.timeline(session_id, method)
        except (SessionError, KeyError, ValueError, OSError) as exc:
            LOGGER.warning("No %s timeline for %s: %s", method, session_id, exc)
    return payloads


def _episodes(timeline: dict[str, Any] | None, limit: int = 5) -> list[dict[str, Any]]:
    """The worst sustained stretches, each pointing at the frame that proves it."""

    if not timeline:
        return []
    method = str(timeline.get("method") or "")
    rows = []
    for episode in (timeline.get("episodes") or [])[:limit]:
        drivers = ", ".join(
            str(driver.get("label"))
            for driver in (episode.get("drivers") or [])[:3]
            if driver.get("label")
        )
        band = episode.get("band") or {}
        rows.append(
            {
                "method": method,
                "window": episode.get("timeRange") or MISSING,
                "duration": _number(episode.get("durationSec")),
                "peak": _number(episode.get("peakScore")),
                "peak_clock": _clock(episode.get("peakSec")),
                "peak_frame": _number(episode.get("peakFrame")),
                "band": band.get("label") or MISSING,
                "class": _status_class(band.get("status")),
                "drivers": drivers or MISSING,
            }
        )
    return rows


def _charts(timeline: dict[str, Any] | None) -> dict[str, Any]:
    if not timeline:
        return {}
    return {
        "timeline": svg_charts.score_timeline(timeline),
        "exposure": svg_charts.exposure_donut(timeline),
        "drivers": svg_charts.driver_bars(timeline),
    }


def _build_context(session_id: str) -> dict[str, Any]:
    from api.assessments import get_review
    from api.interpretations import for_report
    from api.sessions import load_summary, parse_session_id

    ref = parse_session_id(session_id)
    summary = load_summary(session_id)
    context, trial = _study_context(session_id)
    info = (context.get("participant") or {}).get("info") or {}

    completion = summary.get("method_completion") or {}
    reba = summary.get("reba_summary") or {}
    rula = summary.get("rula_summary") or {}
    quality = summary.get("data_quality") or {}
    rnle = summary.get("rnle") or {}
    payloads = _timelines(session_id)
    camera_count = quality.get("camera_count") or len(summary.get("video_metadata") or [])

    return {
        "review": get_review(session_id),
        "interpretations": for_report(session_id),
        "summaries": [data["verdict"] for data in payloads.values() if data and data.get("verdict")],
        "study_name": (context.get("study") or {}).get("name") or ref.study_folder,
        "task_name": trial.get("name") or ref.task_folder,
        "task_description": trial.get("description") or "",
        "primary_tool": str(trial.get("primary_tool") or "").upper(),
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "software_version": summary.get("software_version") or MISSING,
        "sampling_label": ref.sampling_label,
        "participant_id": context.get("participant_id") or ref.participant_id,
        "recorded_at": context.get("date_label") or MISSING,
        "age": info.get("age") or MISSING,
        "sex": info.get("sex") or MISSING,
        "height_cm": info.get("height_cm") or MISSING,
        "weight_kg": info.get("weight_kg") or MISSING,
        "dominant_hand": info.get("dominant_hand") or MISSING,
        "duration": _clock(summary.get("duration_sec")),
        "duration_sec": _number(summary.get("duration_sec")),
        "processed_frames": _number(summary.get("processed_frames")),
        "pose_frames": _number(summary.get("pose_detected_frames")),
        "camera_count": _number(camera_count),
        "camera_setup": ("Single camera" if camera_count == 1 else
                         f"{camera_count} cameras" if camera_count else "Camera count unavailable"),
        "calibration": summary.get("calibration_status") or MISSING,
        "reba": _method_block(
            reba, completion.get("reba") or {}, _reba_band(reba.get("maximum_score"))
        ),
        "rula": _method_block(
            rula, completion.get("rula") or {}, _rula_band(rula.get("maximum_score"))
        ),
        "rnle": _rnle_block(rnle),
        "warnings": [str(item) for item in (summary.get("warnings") or [])],
        "charts": {
            "reba": _charts(payloads.get("reba")),
            "rula": _charts(payloads.get("rula")),
            "multipliers": svg_charts.multiplier_bars(rnle.get("multipliers") or {}),
        },
        "episodes": _episodes(payloads.get("reba") or payloads.get("rula")),
    }


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


def _render_html(context: dict[str, Any]) -> str:
    if not TEMPLATE_PATH.is_dir():
        raise ReportError(f"Template directory not found at {TEMPLATE_PATH}")

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_PATH)),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["clock"] = _clock
    return env.get_template("report.html").render(brand_logo=brand_logo_uri(), **context)


# ---------------------------------------------------------------------------
# PDF generation via Playwright
# ---------------------------------------------------------------------------


def _find_node() -> str:
    try:
        result = subprocess.run(
            ["where", "node"], capture_output=True, text=True, timeout=5, check=False
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0].strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    for candidate in (
        r"C:\Program Files\nodejs\node.exe",
        r"C:\Program Files (x86)\nodejs\node.exe",
        "/usr/bin/node",
        "/usr/local/bin/node",
    ):
        if Path(candidate).exists():
            return candidate

    raise ReportError("Node.js was not found. Install it from nodejs.org and retry.")


def _decode(raw: bytes | None) -> str:
    return raw.decode("utf-8", errors="replace").strip() if raw else ""


def _generate_pdf(html_content: str, output_path: Path) -> None:
    node = _find_node()
    if not GENERATOR_SCRIPT.is_file():
        raise ReportError(f"PDF generator script is missing at {GENERATOR_SCRIPT}")

    node_modules = PROJECT_ROOT / "web" / "node_modules"
    if not (node_modules / "playwright").is_dir():
        raise ReportError(f"Playwright is not installed. {INSTALL_HINT}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, mode="w", encoding="utf-8"
    ) as handle:
        handle.write(html_content)
        tmp_html = handle.name

    try:
        result = subprocess.run(
            [node, str(GENERATOR_SCRIPT), tmp_html, str(output_path)],
            capture_output=True,
            timeout=PDF_TIMEOUT_SEC,
            cwd=str(PROJECT_ROOT / "web"),
            env={
                **os.environ,
                "NODE_PATH": str(node_modules),
                "PLAYWRIGHT_BROWSERS_PATH": BROWSERS_IN_PACKAGE,
            },
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ReportError(
            f"PDF rendering did not finish within {PDF_TIMEOUT_SEC} s"
        ) from exc
    except OSError as exc:
        raise ReportError(f"Could not run the PDF renderer: {exc}") from exc
    finally:
        try:
            os.unlink(tmp_html)
        except OSError:
            LOGGER.warning("Could not remove the temporary HTML at %s", tmp_html)

    if result.returncode != 0:
        detail = _decode(result.stderr) or _decode(result.stdout)
        LOGGER.error("Playwright PDF failed: %s", detail)
        if "Executable doesn't exist" in detail:
            raise ReportError(f"The Playwright browser is not downloaded. {INSTALL_HINT}")
        raise ReportError(f"PDF generation failed: {detail[:300]}")

    if not output_path.exists():
        raise ReportError("The renderer reported success but wrote no file")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def report_path(session_id: str) -> Path:
    """Where this session's report lives: beside the results it describes."""

    from api.sessions import resolve_result_dir

    return Path(resolve_result_dir(session_id)) / REPORT_FILENAME


def report_exists(session_id: str) -> Path | None:
    """Return the cached report for a session, or None."""

    path = report_path(session_id)
    return path if path.is_file() else None


def build_report(session_id: str) -> Path:
    """Render the PDF assessment report for one analysed session."""

    LOGGER.info("Building report for %s", session_id)
    output_path = report_path(session_id)
    _generate_pdf(_render_html(_build_context(session_id)), output_path)
    LOGGER.info(
        "Report generated: %s (%.1f KB)",
        output_path.name,
        output_path.stat().st_size / 1024,
    )
    return output_path


__all__ = [
    "REPORT_FILENAME",
    "ReportError",
    "build_report",
    "report_exists",
    "report_path",
]
