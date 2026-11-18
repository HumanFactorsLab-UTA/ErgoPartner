"""Capture and compare pipeline baselines over real study footage.

Phase 2 of the improvement plan deliberately changes computed angles, so a
change in the numbers is expected. What must not happen is a change nobody
noticed. This module pins the current output of the whole decode-to-score path
over fixed frame ranges, and reports exactly which measurements moved and by how
much.

The footage lives outside the repository, so every entry point degrades to a
skip when it is absent rather than failing. Set ``ERGO_STUDY_ROOT`` to point at
a different studies directory.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT = Path(__file__).resolve().parents[2]
BASELINE_PATH = Path(__file__).resolve().parent / "baseline_snapshot.json"

DEFAULT_STUDY_ROOT = PROJECT.parent / "studies"
TRIAL = Path("study_20260212_125912") / "participant_001" / "task1_niosh_lifting" / "raw"

# A window with a reliably detected subject, away from the multi-person
# activity at the start of the recording.
START_FRAME = 300
END_FRAME = 399

# Angles are compared to this tolerance. Repeated runs on one machine reproduce
# exactly, so anything above floating-point noise is a real behavioural change.
ANGLE_TOLERANCE_DEG = 1e-6

# Task conditions an ergonomist would supply. Fixed here so the baseline
# exercises the scoring path rather than only the incomplete path.
OBSERVED_TASK_INPUTS: dict[str, Any] = {
    "load_kg": 8.0,
    "coupling": "fair",
    "shock_or_rapid_force": False,
    "static_posture_over_one_minute": False,
    "repeated_more_than_four_per_minute": True,
    "rapid_large_posture_change_or_unstable_base": False,
    "seated": False,
    "legs_weight_bearing": "bilateral",
    "legs_supported_and_balanced": True,
    "trunk_twisted": False,
    "trunk_side_bent": False,
    "trunk_twisted_or_side_flexed": False,
    "neck_twisted": False,
    "neck_side_bent": False,
    "neck_twisted_or_side_flexed": False,
    "rula_load_type": "static_or_repeated",
    "left_upper_arm_abducted": False,
    "right_upper_arm_abducted": False,
    "left_upper_arm_abducted_or_rotated": False,
    "right_upper_arm_abducted_or_rotated": False,
    "left_shoulder_raised": False,
    "right_shoulder_raised": False,
    "left_arm_supported_or_leaning": False,
    "right_arm_supported_or_leaning": False,
    "left_arm_supported_or_gravity_assisted": False,
    "right_arm_supported_or_gravity_assisted": False,
    "left_lower_arm_across_midline_or_out_to_side": False,
    "right_lower_arm_across_midline_or_out_to_side": False,
    "left_wrist_deviated": False,
    "right_wrist_deviated": False,
    "left_wrist_twist_at_end_range": False,
    "right_wrist_twist_at_end_range": False,
    "left_wrist_deviated_or_twisted": False,
    "right_wrist_deviated_or_twisted": False,
}


@dataclass(frozen=True)
class Scenario:
    """One reproducible pipeline configuration over a fixed frame range."""

    name: str
    cameras: tuple[str, ...]
    supply_task_inputs: bool
    allow_pose_only_wrist: bool
    description: str


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="single_camera_shipped_config",
        cameras=("cam_1.mp4",),
        supply_task_inputs=False,
        allow_pose_only_wrist=False,
        description="config.yaml as distributed; pins the current incomplete-result behaviour",
    ),
    Scenario(
        name="single_camera_scored",
        cameras=("cam_1.mp4",),
        supply_task_inputs=True,
        allow_pose_only_wrist=True,
        description="task conditions supplied so RULA reaches a score; pins the scoring path",
    ),
    Scenario(
        name="multi_camera_scored",
        cameras=("cam_0.mp4", "cam_1.mp4", "cam_2.mp4"),
        supply_task_inputs=True,
        allow_pose_only_wrist=True,
        description="three synchronized cameras; pins multi-view fusion and selection",
    ),
)


def study_root() -> Path:
    return Path(os.environ.get("ERGO_STUDY_ROOT", str(DEFAULT_STUDY_ROOT)))


def footage_available(scenario: Scenario) -> bool:
    base = study_root() / TRIAL
    return all((base / camera).is_file() for camera in scenario.cameras)


def missing_footage_reason() -> str:
    return (
        f"Baseline footage not found under {study_root() / TRIAL}. "
        "Set ERGO_STUDY_ROOT to the studies directory to enable these checks."
    )


def build_config(scenario: Scenario) -> dict[str, Any]:
    """Return the effective configuration for one scenario."""

    config = yaml.safe_load((PROJECT / "config.yaml").read_text(encoding="utf-8"))
    # Pin sampling off: these checks are about which source frames are
    # decoded and reported, which the shipped target_fps would change.
    config["video"]["target_fps"] = None
    config["video"]["start_frame"] = START_FRAME
    config["video"]["end_frame"] = END_FRAME
    if scenario.supply_task_inputs:
        config["task_inputs"].update(OBSERVED_TASK_INPUTS)
    if scenario.allow_pose_only_wrist:
        config["quality"]["allow_pose_only_wrist"] = True
    return config


def _frame_digest(record: dict[str, Any]) -> dict[str, Any]:
    """Reduce one frame record to the values a regression must protect."""

    def method(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        side = payload.get("selected") or payload.get("left") or {}
        return {
            "final_score": side.get("final_score"),
            "validity_status": side.get("validity_status"),
            "selected_side": payload.get("selected_side"),
            "missing_inputs": sorted(side.get("missing_inputs") or []),
            "intermediate_scores": side.get("intermediate_scores") or {},
        }

    angles = {
        name: (round(value, 6) if isinstance(value, (int, float)) else value)
        for name, value in sorted((record.get("angles") or {}).items())
        if not isinstance(value, dict)
    }
    return {
        "frame_number": record.get("frame_number"),
        "timestamp_sec": round(float(record.get("timestamp_sec", 0.0)), 6),
        "person_identifier": record.get("person_identifier"),
        "angles": angles,
        "rula": method(record.get("rula")),
        "reba": method(record.get("reba")),
    }


def run_scenario(scenario: Scenario) -> dict[str, Any]:
    """Execute one scenario and return its comparable digest."""

    from pipeline.video_processor import VideoProcessor

    base = study_root() / TRIAL
    videos = [str(base / camera) for camera in scenario.cameras]
    output = Path(tempfile.mkdtemp(prefix=f"baseline_{scenario.name}_"))
    shutil.rmtree(output, ignore_errors=True)
    try:
        VideoProcessor(build_config(scenario)).process(videos, str(output))
        summary = json.loads((output / "video_summary.json").read_text(encoding="utf-8"))
        records = [
            json.loads(line)
            for line in (output / "frame_results.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        completion = summary.get("method_completion") or {}
        return {
            "description": scenario.description,
            "cameras": list(scenario.cameras),
            "counts": {
                "processed_frames": summary.get("processed_frames"),
                "pose_detected_frames": summary.get("pose_detected_frames"),
                "pose_observations": summary.get("pose_observations"),
                "valid_rula_observations": summary.get("valid_rula_observations"),
                "valid_reba_observations": summary.get("valid_reba_observations"),
            },
            "method_completion": {
                name: {
                    "scored_observations": entry.get("scored_observations"),
                    "total_observations": entry.get("total_observations"),
                    "primary_blocker": entry.get("primary_blocker"),
                }
                for name, entry in completion.items()
                if isinstance(entry, dict)
            },
            "frames": [_frame_digest(record) for record in records],
        }
    finally:
        shutil.rmtree(output, ignore_errors=True)


def capture(scenarios: tuple[Scenario, ...] = SCENARIOS) -> dict[str, Any]:
    """Run every available scenario and return a complete snapshot."""

    return {
        "start_frame": START_FRAME,
        "end_frame": END_FRAME,
        "scenarios": {
            scenario.name: run_scenario(scenario)
            for scenario in scenarios
            if footage_available(scenario)
        },
    }


def load_baseline() -> dict[str, Any] | None:
    if not BASELINE_PATH.is_file():
        return None
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def save_baseline(snapshot: dict[str, Any]) -> None:
    BASELINE_PATH.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def compare_scenario(expected: dict[str, Any], actual: dict[str, Any]) -> Iterator[str]:
    """Yield a human-readable difference for every value that moved."""

    for key, want in (expected.get("counts") or {}).items():
        got = (actual.get("counts") or {}).get(key)
        if want != got:
            yield f"counts.{key}: baseline {want} -> now {got}"

    expected_frames = {frame["frame_number"]: frame for frame in expected.get("frames", [])}
    actual_frames = {frame["frame_number"]: frame for frame in actual.get("frames", [])}
    only_baseline = sorted(set(expected_frames) - set(actual_frames))
    only_now = sorted(set(actual_frames) - set(expected_frames))
    if only_baseline:
        yield f"frames present in baseline but not now: {only_baseline[:10]}"
    if only_now:
        yield f"frames present now but not in baseline: {only_now[:10]}"

    # Angle drift is summarized per measurement rather than per frame, so a
    # systematic shift reads as one line instead of a hundred.
    drift: dict[str, list[float]] = {}
    score_changes: list[str] = []
    for number in sorted(set(expected_frames) & set(actual_frames)):
        want_frame, got_frame = expected_frames[number], actual_frames[number]
        for name, want_value in (want_frame.get("angles") or {}).items():
            got_value = (got_frame.get("angles") or {}).get(name)
            if isinstance(want_value, (int, float)) and isinstance(got_value, (int, float)):
                delta = abs(float(want_value) - float(got_value))
                if delta > ANGLE_TOLERANCE_DEG:
                    drift.setdefault(name, []).append(delta)
            elif want_value != got_value:
                drift.setdefault(name, [])
        for method in ("rula", "reba"):
            want_score = (want_frame.get(method) or {}).get("final_score")
            got_score = (got_frame.get(method) or {}).get("final_score")
            if want_score != got_score:
                score_changes.append(f"frame {number} {method}: {want_score} -> {got_score}")

    for name, deltas in sorted(drift.items()):
        if deltas:
            worst = max(deltas)
            mean = sum(deltas) / len(deltas)
            yield (
                f"angle {name}: changed on {len(deltas)} frame(s), "
                f"mean {mean:.4f} deg, max {worst:.4f} deg"
            )
        else:
            yield f"angle {name}: availability changed"
    if score_changes:
        yield f"scores changed on {len(score_changes)} frame/method pair(s)"
        for line in score_changes[:10]:
            yield f"  {line}"


def compare(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    """Return every difference between two snapshots."""

    differences: list[str] = []
    expected_scenarios = expected.get("scenarios", {})
    actual_scenarios = actual.get("scenarios", {})
    for name in sorted(set(expected_scenarios) & set(actual_scenarios)):
        for line in compare_scenario(expected_scenarios[name], actual_scenarios[name]):
            differences.append(f"[{name}] {line}")
    for name in sorted(set(expected_scenarios) - set(actual_scenarios)):
        differences.append(f"[{name}] scenario missing from the current run")
    return differences
