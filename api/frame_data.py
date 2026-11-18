"""Lossless, research-oriented access to frame-level analysis evidence.

The chart API deliberately peak-preserves and decimates long score series.  That
is the right trade-off for drawing a timeline, but it is not an acceptable data
source for a searchable table or an export.  This module therefore reads every
indexed JSONL observation and exposes:

* paginated, filterable frame rows;
* contiguous periods in every published RULA/REBA band, including unscored
  evidence gaps; and
* a stable, flattened CSV suitable for analysis outside the dashboard.

No posture is rescored here.  Published classifications come from the shared
``dashboard.viz_theme`` band definitions and every component comes directly
from the stored result trace.
"""

from __future__ import annotations

import csv
import io
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

from fastapi.responses import StreamingResponse

from api.sessions import SessionError, load_run_config, resolve_result_dir
from dashboard.result_store import JsonlResultStore, ResultIndexEntry, open_result_store
from dashboard.task_timeline import clock
from dashboard.viz_theme import RiskBand, method_bands, risk_band

METHODS = ("reba", "rula")
UNSCORED_LABEL = "Unscored"
UNSCORED_STATUS = "unscored"
DEFAULT_EXPORT_FILENAME = "frame_scores.csv"
DEFAULT_MAXIMUM_GAP_SEC = 0.25
MAX_PAGE_SIZE = 500

RULA_COMPONENTS = (
    "upper_arm",
    "lower_arm",
    "wrist",
    "wrist_twist",
    "table_a_posture",
    "muscle_use",
    "force_load",
    "score_a",
    "neck",
    "trunk",
    "legs",
    "table_b_posture",
    "score_b",
    "table_c",
)

REBA_COMPONENTS = (
    "trunk",
    "neck",
    "legs",
    "table_a",
    "load_force",
    "score_a",
    "upper_arm",
    "lower_arm",
    "wrist",
    "table_b",
    "coupling",
    "score_b",
    "table_c",
    "activity",
)

COMPONENTS = {"rula": RULA_COMPONENTS, "reba": REBA_COMPONENTS}

# Stable, useful measurements rather than every trace-only helper value.  The
# source JSONL remains the lossless artifact for callers needing the landmark
# maps or every raw condition input.
ANGLE_EXPORTS = (
    ("trunk_flexion_deg", "trunk_flexion"),
    ("trunk_side_bend_deg", "trunk_side_bend"),
    ("trunk_twist_deg", "trunk_twist"),
    ("neck_flexion_deg", "neck_flexion"),
    ("neck_side_bend_deg", "neck_side_bend"),
    ("neck_rotation_deg", "neck_rotation"),
    ("left_upper_arm_flexion_deg", "left_upper_arm_flexion"),
    ("left_upper_arm_abduction_deg", "left_upper_arm_abduction"),
    ("left_shoulder_elevation_deg", "left_shoulder_elevation"),
    ("left_forearm_flexion_deg", "left_forearm_flexion"),
    ("left_wrist_flexion_deg", "left_wrist_flexion"),
    ("left_wrist_deviation_deg", "left_wrist_deviation"),
    ("left_wrist_pronation_deg", "left_wrist_pronation"),
    ("left_knee_flexion_deg", "left_knee_flexion"),
    ("right_upper_arm_flexion_deg", "right_upper_arm_flexion"),
    ("right_upper_arm_abduction_deg", "right_upper_arm_abduction"),
    ("right_shoulder_elevation_deg", "right_shoulder_elevation"),
    ("right_forearm_flexion_deg", "right_forearm_flexion"),
    ("right_wrist_flexion_deg", "right_wrist_flexion"),
    ("right_wrist_deviation_deg", "right_wrist_deviation"),
    ("right_wrist_pronation_deg", "right_wrist_pronation"),
    ("right_knee_flexion_deg", "right_knee_flexion"),
)


def _check_method(method: str) -> str:
    key = str(method).strip().lower()
    if key not in METHODS:
        raise SessionError(f"Unknown method {method!r}; expected reba or rula")
    return key


def _store(session_id: str) -> JsonlResultStore:
    return open_result_store(Path(resolve_result_dir(session_id)) / "frame_results.jsonl")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _score(value: Any) -> int | None:
    number = _finite(value)
    if number is None or int(number) != number:
        return None
    return int(number)


def _normalise(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def _filter_tokens(value: str | Sequence[str] | None) -> frozenset[str]:
    if value is None:
        return frozenset()
    values: Iterable[Any] = (value,) if isinstance(value, str) else value
    tokens: set[str] = set()
    for item in values:
        tokens.update(_normalise(part) for part in str(item).split(",") if part.strip())
    return frozenset(token for token in tokens if token and token not in {"all", "*"})


def _band_payload(band: RiskBand | None) -> dict[str, Any] | None:
    if band is None:
        return None
    return {
        "label": band.label,
        "action": band.action,
        "status": band.status,
        "lower": band.lower,
        "upper": band.upper,
        "range": band.range_text,
        "describesRisk": band.describes_risk,
    }


def _band_definitions(method: str) -> list[dict[str, Any]]:
    return [_band_payload(band) for band in method_bands(method)]  # type: ignore[misc]


def _method_result(record: Mapping[str, Any], method: str) -> Mapping[str, Any]:
    return _mapping(record.get(method))


def _side_result(method_result: Mapping[str, Any], side: str) -> Mapping[str, Any]:
    return _mapping(method_result.get(side))


def _selected_result(method_result: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(method_result.get("selected"))


def _validity(method_result: Mapping[str, Any]) -> str:
    selected = _selected_result(method_result)
    selected_status = selected.get("validity_status")
    if selected_status:
        return str(selected_status)
    statuses = {
        str(value)
        for side in ("left", "right")
        if (value := _side_result(method_result, side).get("validity_status"))
    }
    if len(statuses) == 1:
        return next(iter(statuses))
    if statuses:
        return "bilateral_incomplete"
    return "unavailable"


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list | tuple | set):
        return []
    return sorted({str(item) for item in value if item is not None and str(item)})


def _evidence_lists(
    method_result: Mapping[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    selected = _selected_result(method_result)
    sources = [selected] if selected else [
        _side_result(method_result, "left"),
        _side_result(method_result, "right"),
    ]
    missing: set[str] = set()
    invalid: set[str] = set()
    warnings: set[str] = set()
    for source in sources:
        missing.update(_string_list(source.get("missing_inputs")))
        invalid.update(_string_list(source.get("invalid_inputs")))
        warnings.update(_string_list(source.get("warnings")))
    return sorted(missing), sorted(invalid), sorted(warnings)


def _numeric_mapping(value: Any) -> dict[str, int]:
    resolved: dict[str, int] = {}
    for name, raw in _mapping(value).items():
        number = _score(raw)
        if number is not None:
            resolved[str(name)] = number
    return resolved


def _numeric_angles(value: Any) -> dict[str, float]:
    resolved: dict[str, float] = {}
    for name, raw in _mapping(value).items():
        number = _finite(raw)
        if number is not None:
            resolved[str(name)] = number
    return resolved


def _side_payload(method_result: Mapping[str, Any], side: str, method: str) -> dict[str, Any]:
    result = _side_result(method_result, side)
    score = _score(result.get("final_score"))
    return {
        "score": score,
        "band": _band_payload(risk_band(score, method)),
        "validity": str(result.get("validity_status") or "unavailable"),
        "components": _numeric_mapping(result.get("intermediate_scores")),
        "missing": _string_list(result.get("missing_inputs")),
        "invalid": _string_list(result.get("invalid_inputs")),
    }


def _observation_row(entry: ResultIndexEntry, method: str) -> dict[str, Any]:
    record = entry.lightweight
    result = _method_result(record, method)
    selected = _selected_result(result)
    score = _score(selected.get("final_score"))
    band = risk_band(score, method)
    missing, invalid, warnings = _evidence_lists(result)
    quality = _mapping(record.get("pose_quality"))
    return {
        "frame": entry.frame_number,
        "timestampSec": entry.timestamp_sec,
        "clock": clock(entry.timestamp_sec),
        "person": entry.person_identifier,
        "score": score,
        "classification": band.label if band is not None else UNSCORED_LABEL,
        "status": band.status if band is not None else UNSCORED_STATUS,
        "band": _band_payload(band),
        "action": band.action if band is not None else None,
        "validity": _validity(result),
        "selectedSide": (
            str(result["selected_side"]) if result.get("selected_side") is not None else None
        ),
        "components": _numeric_mapping(selected.get("intermediate_scores")),
        "sides": {
            side: _side_payload(result, side, method)
            for side in ("left", "right")
        },
        "angles": _numeric_angles(record.get("angles")),
        "quality": {
            "meanVisibility": _finite(quality.get("mean_keypoint_visibility")),
            "status": quality.get("status"),
            "cameraView": quality.get("camera_view_status"),
            "wristSource": quality.get("wrist_data_source"),
            "warnings": _string_list(quality.get("warnings")),
        },
        "meanVisibility": _finite(quality.get("mean_keypoint_visibility")),
        "missing": missing,
        "invalid": invalid,
        "warnings": warnings,
    }


def _sort_key(entry: ResultIndexEntry) -> tuple[str, float, int, int]:
    return (
        entry.person_identifier or "",
        entry.timestamp_sec if entry.timestamp_sec is not None else float("inf"),
        entry.frame_number if entry.frame_number is not None else 2**63 - 1,
        entry.line_number,
    )


def _entries(session_id: str, person: str | None) -> list[ResultIndexEntry]:
    entries = [
        entry
        for entry in _store(session_id).entries
        if person is None or entry.person_identifier == person
    ]
    return sorted(entries, key=_sort_key)


def _matches(
    row: Mapping[str, Any],
    band_tokens: frozenset[str],
    status_tokens: frozenset[str],
) -> bool:
    if band_tokens and _normalise(row.get("classification")) not in band_tokens:
        return False
    if not status_tokens:
        return True
    # ``status`` primarily means the shared visual severity.  Accepting a
    # validity-status token as well makes the same endpoint useful for finding
    # incomplete evidence without adding another nearly identical query field.
    candidates = {
        _normalise(row.get("status")),
        _normalise(row.get("validity")),
    }
    return bool(candidates & status_tokens)


def observations_payload(
    session_id: str,
    method: str,
    band: str | Sequence[str] | None = None,
    status: str | Sequence[str] | None = None,
    page: int = 1,
    limit: int = 100,
    person: str | None = None,
) -> dict[str, Any]:
    """Return a complete, non-decimated page of frame-level evidence."""

    key = _check_method(method)
    if isinstance(page, bool) or page < 1:
        raise ValueError("page must be a positive integer")
    if isinstance(limit, bool) or limit < 1 or limit > MAX_PAGE_SIZE:
        raise ValueError(f"limit must be between 1 and {MAX_PAGE_SIZE}")
    band_tokens = _filter_tokens(band)
    status_tokens = _filter_tokens(status)
    rows = [
        row
        for entry in _entries(session_id, person)
        if _matches(
            row := _observation_row(entry, key),
            band_tokens,
            status_tokens,
        )
    ]
    total = len(rows)
    start = (page - 1) * limit
    selected = rows[start : start + limit]
    return {
        "sessionId": session_id,
        "method": key.upper(),
        "person": person,
        "filters": {
            "band": sorted(band_tokens),
            "status": sorted(status_tokens),
        },
        "page": page,
        "limit": limit,
        "count": len(selected),
        "total": total,
        "totalPages": (total + limit - 1) // limit,
        "pages": (total + limit - 1) // limit,
        "hasNext": start + len(selected) < total,
        "bands": _band_definitions(key),
        "rows": selected,
    }


def _configured_maximum_gap(session_id: str, requested: float | None) -> float:
    if requested is not None:
        value = _finite(requested)
    else:
        config = load_run_config(session_id)
        temporal = _mapping(config.get("temporal_aggregation"))
        value = _finite(temporal.get("maximum_gap_sec"))
    if value is None:
        return DEFAULT_MAXIMUM_GAP_SEC
    if value <= 0:
        raise ValueError("maximum_gap_sec must be positive")
    return value


def _dwell_times(rows: Sequence[Mapping[str, Any]], maximum_gap_sec: float) -> list[float]:
    timestamps = [_finite(row.get("timestampSec")) for row in rows]
    gaps = [
        second - first
        for first, second in zip(timestamps, timestamps[1:], strict=False)
        if first is not None
        and second is not None
        and 0 < second - first <= maximum_gap_sec
    ]
    trailing = median(gaps) if gaps else 0.0
    dwell: list[float] = []
    for index, timestamp in enumerate(timestamps):
        if timestamp is None:
            dwell.append(0.0)
        elif index + 1 == len(timestamps):
            dwell.append(float(trailing))
        else:
            following = timestamps[index + 1]
            delta = following - timestamp if following is not None else 0.0
            dwell.append(float(delta) if 0 < delta <= maximum_gap_sec else 0.0)
    return dwell


def _contiguous(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    maximum_gap_sec: float,
) -> bool:
    first = _finite(previous.get("timestampSec"))
    second = _finite(current.get("timestampSec"))
    return (
        previous.get("classification") == current.get("classification")
        and first is not None
        and second is not None
        and 0 < second - first <= maximum_gap_sec
    )


def _period_payload(
    run: Sequence[Mapping[str, Any]],
    dwell: Sequence[float],
    method: str,
    index: int,
) -> dict[str, Any]:
    first = run[0]
    last = run[-1]
    scored = [row for row in run if _score(row.get("score")) is not None]
    peak = (
        max(
            scored,
            key=lambda row: (
                _score(row.get("score")) or 0,
                -(_finite(row.get("timestampSec")) or 0.0),
            ),
        )
        if scored
        else None
    )
    start_sec = _finite(first.get("timestampSec"))
    last_sec = _finite(last.get("timestampSec"))
    duration = float(sum(dwell))
    end_sec = last_sec + dwell[-1] if last_sec is not None else None
    band = risk_band(_score(peak.get("score")) if peak else None, method)
    return {
        "index": index,
        "method": method.upper(),
        "person": first.get("person"),
        "classification": first.get("classification"),
        "status": first.get("status"),
        "band": _band_payload(band),
        "startSec": None if start_sec is None else round(start_sec, 6),
        "endSec": None if end_sec is None else round(end_sec, 6),
        "durationSec": round(duration, 6),
        "startFrame": first.get("frame"),
        "endFrame": last.get("frame"),
        "representativeFrame": peak.get("frame") if peak else first.get("frame"),
        "peakScore": peak.get("score") if peak else None,
        # Unscored periods still need an evidence frame the UI can open.  It is
        # the first unavailable observation rather than a fabricated score peak.
        "peakFrame": peak.get("frame") if peak else first.get("frame"),
        "peakSec": peak.get("timestampSec") if peak else first.get("timestampSec"),
        "observations": len(run),
        "scored": bool(peak),
    }


def risk_periods_payload(
    session_id: str,
    method: str,
    band: str | Sequence[str] | None = None,
    status: str | Sequence[str] | None = None,
    person: str | None = None,
    maximum_gap_sec: float | None = None,
) -> dict[str, Any]:
    """Return chronological same-band runs, including unscored evidence gaps."""

    key = _check_method(method)
    maximum_gap = _configured_maximum_gap(session_id, maximum_gap_sec)
    band_tokens = _filter_tokens(band)
    status_tokens = _filter_tokens(status)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in _entries(session_id, person):
        row = _observation_row(entry, key)
        grouped[str(row.get("person") or "")].append(row)

    periods: list[dict[str, Any]] = []
    for person_key in sorted(grouped):
        rows = grouped[person_key]
        dwell = _dwell_times(rows, maximum_gap)
        start = 0
        for index in range(1, len(rows) + 1):
            if index < len(rows) and _contiguous(rows[index - 1], rows[index], maximum_gap):
                continue
            run = rows[start:index]
            periods.append(
                _period_payload(
                    run,
                    dwell[start:index],
                    key,
                    len(periods) + 1,
                )
            )
            start = index

    filtered = [
        period
        for period in periods
        if _matches(period, band_tokens, status_tokens)
    ]
    recording_duration = max(
        (
            float(period["endSec"])
            for period in periods
            if period.get("endSec") is not None
        ),
        default=0.0,
    )
    return {
        "sessionId": session_id,
        "method": key.upper(),
        "person": person,
        "filters": {
            "band": sorted(band_tokens),
            "status": sorted(status_tokens),
        },
        "maximumGapSec": maximum_gap,
        "count": len(filtered),
        "unfilteredCount": len(periods),
        "durationSec": round(recording_duration, 6),
        "classifiedDurationSec": round(
            sum(float(period["durationSec"]) for period in filtered),
            6,
        ),
        "bands": _band_definitions(key),
        "periods": filtered,
    }


def _csv_method_fields(method: str) -> list[str]:
    fields = [
        f"{method}_status",
        f"{method}_score",
        f"{method}_classification",
        f"{method}_action",
        f"{method}_band_status",
        f"{method}_selected_side",
        f"{method}_selection_reason",
        f"{method}_left_score",
        f"{method}_left_classification",
        f"{method}_left_status",
        f"{method}_right_score",
        f"{method}_right_classification",
        f"{method}_right_status",
        f"{method}_missing_inputs",
        f"{method}_invalid_inputs",
        f"{method}_warnings",
    ]
    for source in ("selected", "left", "right"):
        fields.extend(f"{method}_{source}_{component}" for component in COMPONENTS[method])
    return fields


CSV_FIELDS = tuple(
    [
        "software_version",
        "frame_number",
        "timestamp_sec",
        "person_identifier",
        "calibration_status",
        "pose_mean_visibility",
        "pose_quality_status",
        "camera_view_status",
        "wrist_data_source",
        "pose_warnings",
        "task_conditions_true",
        "task_condition_sources",
    ]
    + _csv_method_fields("rula")
    + _csv_method_fields("reba")
    + [name for name, _ in ANGLE_EXPORTS]
)


def _joined(value: Any) -> str:
    return ";".join(_string_list(value))


def _task_condition_text(record: Mapping[str, Any]) -> tuple[str, str]:
    conditions = _mapping(record.get("task_conditions"))
    values = _mapping(conditions.get("values"))
    sources = _mapping(conditions.get("sources"))
    true_values = ";".join(sorted(str(name) for name, value in values.items() if value is True))
    source_values = ";".join(
        f"{name}={sources[name]}"
        for name in sorted(sources)
        if sources[name] is not None
    )
    return true_values, source_values


def _csv_method_values(record: Mapping[str, Any], method: str) -> dict[str, Any]:
    result = _method_result(record, method)
    selected = _selected_result(result)
    score = _score(selected.get("final_score"))
    band = risk_band(score, method)
    missing, invalid, warnings = _evidence_lists(result)
    values: dict[str, Any] = {
        f"{method}_status": _validity(result),
        f"{method}_score": score,
        f"{method}_classification": band.label if band else UNSCORED_LABEL,
        f"{method}_action": band.action if band else "",
        f"{method}_band_status": band.status if band else UNSCORED_STATUS,
        f"{method}_selected_side": result.get("selected_side"),
        f"{method}_selection_reason": result.get("selection_reason"),
        f"{method}_missing_inputs": ";".join(missing),
        f"{method}_invalid_inputs": ";".join(invalid),
        f"{method}_warnings": ";".join(warnings),
    }
    for side in ("left", "right"):
        side_result = _side_result(result, side)
        side_score = _score(side_result.get("final_score"))
        side_band = risk_band(side_score, method)
        values[f"{method}_{side}_score"] = side_score
        values[f"{method}_{side}_classification"] = (
            side_band.label if side_band else UNSCORED_LABEL
        )
        values[f"{method}_{side}_status"] = (
            side_result.get("validity_status") or "unavailable"
        )
    component_sources = {
        "selected": selected,
        "left": _side_result(result, "left"),
        "right": _side_result(result, "right"),
    }
    for source, source_result in component_sources.items():
        components = _numeric_mapping(source_result.get("intermediate_scores"))
        for component in COMPONENTS[method]:
            values[f"{method}_{source}_{component}"] = components.get(component)
    return values


def _csv_row(entry: ResultIndexEntry) -> dict[str, Any]:
    record = entry.lightweight
    quality = _mapping(record.get("pose_quality"))
    angles = _mapping(record.get("angles"))
    conditions, condition_sources = _task_condition_text(record)
    row: dict[str, Any] = {
        "software_version": record.get("software_version"),
        "frame_number": entry.frame_number,
        "timestamp_sec": entry.timestamp_sec,
        "person_identifier": entry.person_identifier,
        "calibration_status": record.get("calibration_status"),
        "pose_mean_visibility": _finite(quality.get("mean_keypoint_visibility")),
        "pose_quality_status": quality.get("status"),
        "camera_view_status": quality.get("camera_view_status"),
        "wrist_data_source": quality.get("wrist_data_source"),
        "pose_warnings": _joined(quality.get("warnings")),
        "task_conditions_true": conditions,
        "task_condition_sources": condition_sources,
    }
    row.update(_csv_method_values(record, "rula"))
    row.update(_csv_method_values(record, "reba"))
    for export_name, source_name in ANGLE_EXPORTS:
        row[export_name] = _finite(angles.get(source_name))
    return row


def _csv_chunk(writer: csv.DictWriter[str], buffer: io.StringIO) -> str:
    value = buffer.getvalue()
    buffer.seek(0)
    buffer.truncate(0)
    return value


def iter_enriched_csv(
    session_id: str,
    *,
    person: str | None = None,
    method: str | None = None,
    band: str | Sequence[str] | None = None,
    status: str | Sequence[str] | None = None,
) -> Iterator[str]:
    """Yield a stable CSV without retaining the complete export in memory."""

    band_tokens = _filter_tokens(band)
    status_tokens = _filter_tokens(status)
    key = _check_method(method) if method is not None else None
    if key is None and (band_tokens or status_tokens):
        raise ValueError("method is required when filtering an export by band or status")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    yield _csv_chunk(writer, buffer)
    for entry in _entries(session_id, person):
        if key is not None:
            observation = _observation_row(entry, key)
            if not _matches(observation, band_tokens, status_tokens):
                continue
        writer.writerow(_csv_row(entry))
        yield _csv_chunk(writer, buffer)


def _safe_filename(session_id: str, filename: str | None) -> str:
    requested = filename or f"{session_id}_{DEFAULT_EXPORT_FILENAME}"
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(requested).name).strip("._")
    if not name:
        name = DEFAULT_EXPORT_FILENAME
    if not name.lower().endswith(".csv"):
        name += ".csv"
    return name


def csv_export_response(
    session_id: str,
    *,
    person: str | None = None,
    method: str | None = None,
    band: str | Sequence[str] | None = None,
    status: str | Sequence[str] | None = None,
    filename: str | None = None,
) -> StreamingResponse:
    """Return a streaming FastAPI response for the enriched frame-score CSV."""

    # Validate before response headers are sent.  Generator functions otherwise
    # defer this work until the first body chunk, which would turn a bad query
    # into a broken 200 response rather than a normal API error.
    if method is not None:
        _check_method(method)
    elif _filter_tokens(band) or _filter_tokens(status):
        raise ValueError("method is required when filtering an export by band or status")
    safe_name = _safe_filename(session_id, filename)
    return StreamingResponse(
        iter_enriched_csv(
            session_id,
            person=person,
            method=method,
            band=band,
            status=status,
        ),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}"',
            "Cache-Control": "no-store",
        },
    )


# Route code can use the more domain-specific aliases without coupling itself to
# the implementation names.
frame_scores_csv_iter = iter_enriched_csv
frame_scores_csv_response = csv_export_response


__all__ = [
    "CSV_FIELDS",
    "MAX_PAGE_SIZE",
    "csv_export_response",
    "frame_scores_csv_iter",
    "frame_scores_csv_response",
    "iter_enriched_csv",
    "observations_payload",
    "risk_periods_payload",
]
