"""Structured JSON Lines, CSV, JSON summary, and Markdown report writers."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, TextIO


def serializable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return serializable(value.to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return serializable(asdict(value))  # type: ignore[arg-type]
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass
    return value


class ResultWriter:
    """Stream frame results without retaining video frames in memory."""

    CSV_HEADERS = (
        "frame_number",
        "timestamp_sec",
        "person_identifier",
        "input_source",
        "rula_status",
        "rula_score",
        "rula_selected_side",
        "rula_missing_inputs",
        "reba_status",
        "reba_score",
        "reba_selected_side",
        "reba_missing_inputs",
        "pose_mean_visibility",
        "calibration_status",
    )

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl: TextIO | None = None
        self._csv_file: TextIO | None = None
        self._csv: csv.DictWriter[str] | None = None

    def __enter__(self):
        self._jsonl = (self.output_dir / "frame_results.jsonl").open("w", encoding="utf-8")
        self._csv_file = (self.output_dir / "frame_results.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._csv = csv.DictWriter(self._csv_file, fieldnames=self.CSV_HEADERS)
        self._csv.writeheader()
        return self

    @staticmethod
    def _missing(bilateral) -> str:
        values = set()
        for result in (bilateral.left, bilateral.right):
            values.update(result.missing_inputs)
            values.update(result.invalid_inputs)
        return ";".join(sorted(values))

    def write_frame(self, record: dict[str, Any]) -> None:
        if self._jsonl is None or self._csv is None:
            raise RuntimeError("ResultWriter must be opened with a context manager")
        payload = serializable(record)
        self._jsonl.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        rula = record["rula"]
        reba = record["reba"]
        selected_rula = rula.selected
        selected_reba = reba.selected
        self._csv.writerow(
            {
                "frame_number": record["frame_number"],
                "timestamp_sec": f"{record['timestamp_sec']:.6f}",
                "person_identifier": record["person_identifier"],
                "input_source": record["input_source"],
                "rula_status": (selected_rula or rula.left).validity_status,
                "rula_score": selected_rula.final_score if selected_rula else "",
                "rula_selected_side": rula.selected_side or "",
                "rula_missing_inputs": self._missing(rula),
                "reba_status": (selected_reba or reba.left).validity_status,
                "reba_score": selected_reba.final_score if selected_reba else "",
                "reba_selected_side": reba.selected_side or "",
                "reba_missing_inputs": self._missing(reba),
                "pose_mean_visibility": record["pose_quality"].get("mean_keypoint_visibility", ""),
                "calibration_status": record["calibration_status"],
            }
        )

    def __exit__(self, exc_type, exc, traceback):
        if self._jsonl:
            self._jsonl.close()
        if self._csv_file:
            self._csv_file.close()


def _completion_lines(completion: dict[str, Any]) -> list[str]:
    """Render why each method did or did not reach a score.

    A zero observation count on its own does not tell an ergonomist whether the
    subject was never detected or a single input was missing, so the blocking
    inputs the calculators already recorded are listed in impact order.
    """

    if not completion:
        return []
    lines = ["## Method completion", ""]
    for entry in completion.values():
        if not isinstance(entry, dict):
            continue
        method = entry.get("method", "Method")
        total = entry.get("total_observations") or 0
        scored = entry.get("scored_observations") or 0
        rate = entry.get("completion_rate")
        if not total:
            lines.append(f"- **{method}**: no pose observations were available to assess.")
            continue
        share = f" ({rate:.0%})" if isinstance(rate, (int, float)) else ""
        lines.append(f"- **{method}**: scored {scored:,} of {total:,} observations{share}.")
        blocking = [item for item in entry.get("blocking_inputs", []) if isinstance(item, dict)]
        measurements = [item for item in blocking if item.get("category") == "pose_measurement"]
        task_inputs = [item for item in blocking if item.get("category") != "pose_measurement"]
        # Measurements the pipeline could not produce are listed individually
        # because each one is a defect to investigate; outstanding task
        # conditions are counted, because they are simply not entered yet.
        for item in measurements:
            name = item.get("name", "unknown")
            count = item.get("blocked_observations") or 0
            item_share = item.get("share_of_blocked")
            suffix = f" ({item_share:.0%} of blocked)" if isinstance(item_share, float) else ""
            lines.append(
                f"  - unavailable pose measurement `{name}` on {count:,} observations{suffix}"
            )
        if task_inputs:
            names = ", ".join(f"`{item.get('name', 'unknown')}`" for item in task_inputs[:6])
            more = f", and {len(task_inputs) - 6} more" if len(task_inputs) > 6 else ""
            lines.append(
                f"  - {len(task_inputs)} task condition(s) not yet supplied: {names}{more}"
            )
    lines.append("")
    return lines


def write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    output_dir = Path(output_dir)
    payload = serializable(summary)
    with (output_dir / "video_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    rula = payload["rula_summary"]
    reba = payload["reba_summary"]
    niosh = payload["rnle"]
    completion = payload.get("method_completion") or {}
    lines = [
        "# Ergonomic Video Assessment",
        "",
        f"- Input: `{payload['input_source']}`",
        f"- Processed frames: {payload['processed_frames']}",
        f"- Frames with a detected pose: {payload['pose_detected_frames']}",
        f"- Valid bilateral RULA observations: {rula['observation_count']}",
        f"- Valid bilateral REBA observations: {reba['observation_count']}",
        f"- Calibration: {payload['calibration_status']}",
        f"- Temporal aggregation: {', '.join(rula['aggregation_methods'])}",
        "",
        "## RULA",
        "",
        f"- Worst score: {rula['maximum_score']}",
        f"- Worst timestamp: {rula['maximum_timestamp_sec']}",
        f"- {rula['percentile']:g}th percentile: {rula['high_percentile']}",
        "",
        "## REBA",
        "",
        f"- Worst score: {reba['maximum_score']}",
        f"- Worst timestamp: {reba['maximum_timestamp_sec']}",
        f"- {reba['percentile']:g}th percentile: {reba['high_percentile']}",
        "",
        "## Revised NIOSH Lifting Equation",
        "",
        f"- Validity status: {niosh['validity_status']}",
        f"- RWL (kg): {niosh['recommended_weight_limit_kg']}",
        f"- Lifting Index: {niosh['lifting_index']}",
        f"- Missing inputs: {', '.join(niosh['missing_inputs']) or 'none'}",
        f"- Unsupported conditions: {', '.join(niosh['unsupported_conditions']) or 'none'}",
        "",
    ]
    lines.extend(_completion_lines(completion))
    lines.extend(
        [
            "## Limitations and warnings",
            "",
        ]
    )
    warnings = payload.get("warnings", [])
    lines.extend(f"- {warning}" for warning in warnings)
    if not warnings:
        lines.append("- None reported.")
    (output_dir / "ASSESSMENT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
