"""Select the blind validation frames and build the three-panel composites.

Design
------
Sixty frames are drawn under a two-stratum design:

* **sustained** -- the REBA band holds across a window either side of the frame, so
  the label describes a posture the participant actually held;
* **transient** -- the band does not hold, so the label describes an instant.

Both strata are balanced across the three risk bands.  Reporting them separately is
the point: agreement on the sustained stratum measures whether the pipeline reads
posture the way an expert does, and the gap between the strata measures what
frame-to-frame pose instability costs.  Collapsing the two -- which is what a design
stratified on band alone does -- charges every transient to the assessment logic.

Each selected frame is written as one horizontal composite of its three
synchronised camera views, carrying no text, score or identifier.  The mapping back
to the source data lives only in ``ground_truth_key.csv``.

Usage
-----
    python -m validation.sample_frames --no-images      # solve and inspect the design
    python -m validation.sample_frames                  # full run, writes composites
    python -m validation.sample_frames --seed 12345     # different draw, same design
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from validation import allocate
from validation.pool import (
    DEFAULT_STABILITY_HALF_WINDOW,
    RISK_BANDS,
    PoolError,
    build_pool,
    camera_references,
    load_frame_details,
)

LOGGER = logging.getLogger("validation.sample_frames")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "validation_sample"

STRATA: tuple[str, ...] = ("sustained", "transient")

#: Frames per (stratum, band).  Sustained carries the larger share because it is the
#: primary estimate; transient needs only enough to size the instability effect.
DEFAULT_QUOTA: dict[str, dict[str, int]] = {
    "sustained": {"Low": 12, "Medium": 12, "High": 12},
    "transient": {"Low": 8, "Medium": 8, "High": 8},
}

MIN_FRAMES_PER_PARTICIPANT = 2
MAX_FRAMES_PER_PARTICIPANT = 4
#: Hard cap on frames drawn from a single recording.
MAX_FRAMES_PER_RECORDING = 3

#: Preferred spacing between two frames taken from the same recording.  A preference,
#: not a rule -- it is applied by farthest-point selection, which always returns the
#: requested count.
MIN_SECONDS_BETWEEN_FRAMES = 8.0

RANDOM_SEED = 20260801

COMPOSITE_HEIGHT_PX = 720
PANEL_SEPARATOR_PX = 6
PANEL_SEPARATOR_COLOR = (0, 0, 0)
JPEG_QUALITY = 95
COMPOSITE_NAME_TEMPLATE = "frame_{index:03d}.jpg"
CAMERA_ROLE_ORDER = ("front", "side", "iso")

GROUND_TRUTH_FILENAME = "ground_truth_key.csv"
MANIFEST_FILENAME = "sampling_manifest.json"

ALLOW_FRAME_SEEK = False
SEEK_MIN_GAP_FRAMES = 240


class SamplingError(RuntimeError):
    """Raised when the sample cannot be produced."""


def cell_name(stratum: str, band: str) -> str:
    return f"{stratum}:{band}"


def split_cell(cell: str) -> tuple[str, str]:
    stratum, band = cell.split(":", 1)
    return stratum, band


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _pick_spread(
    candidates: pd.DataFrame, count: int, already: list[float], rng: random.Random
) -> list[int]:
    """Choose ``count`` row labels spread as far apart in time as possible.

    Farthest-point selection on the timestamp axis: the first pick is random among
    those clear of ``already``, and each subsequent pick maximises the minimum
    distance to everything chosen so far.
    """

    timestamps = candidates["timestamp_sec"].to_dict()
    labels = list(timestamps)
    rng.shuffle(labels)
    if count >= len(labels):
        return labels

    reference = list(already)
    clear = [
        label
        for label in labels
        if all(abs(timestamps[label] - other) >= MIN_SECONDS_BETWEEN_FRAMES for other in reference)
    ]
    chosen = [rng.choice(clear) if clear else rng.choice(labels)]
    reference.append(timestamps[chosen[0]])

    while len(chosen) < count:
        remaining = [label for label in labels if label not in chosen]
        best_label, best_distance = remaining[0], -1.0
        for label in remaining:
            distance = min(abs(timestamps[label] - other) for other in reference)
            if distance > best_distance:
                best_label, best_distance = label, distance
        chosen.append(best_label)
        reference.append(timestamps[best_label])
    return chosen


def select_frames(
    pool: pd.DataFrame, quota: dict[str, dict[str, int]], seed: int
) -> tuple[pd.DataFrame, allocate.Allocation, dict[str, Any]]:
    """Solve the design and materialise it into concrete frames."""

    rng = random.Random(seed)
    pool = pool.copy()
    pool["stratum"] = pool["sustained"].map({True: "sustained", False: "transient"})
    pool["cell"] = [cell_name(s, b) for s, b in zip(pool["stratum"], pool["band"], strict=True)]

    cell_quota: dict[str, int] = {}
    for stratum in STRATA:
        for band in RISK_BANDS:
            value = quota.get(stratum, {}).get(band, 0)
            if value:
                cell_quota[cell_name(stratum, band)] = value
    if not cell_quota:
        raise SamplingError("every quota is zero; nothing to sample")

    total_frames = sum(cell_quota.values())
    participants = sorted(pool["participant"].unique())
    tasks = sorted(pool["task"].unique(), key=lambda name: (pool.loc[pool["task"] == name, "task_index"].iloc[0], name))
    cell_types = list(cell_quota)

    base, remainder = divmod(total_frames, len(tasks))
    task_target = {task: base + (1 if index < remainder else 0) for index, task in enumerate(tasks)}

    availability = {
        key: int(size) for key, size in pool.groupby(["participant", "task", "cell"]).size().items()
    }

    supply = {cell: sum(v for k, v in availability.items() if k[2] == cell) for cell in cell_types}
    LOGGER.info("Requested quotas: %s", cell_quota)
    LOGGER.info("Available supply: %s", supply)
    for cell, wanted in cell_quota.items():
        if supply.get(cell, 0) < wanted:
            raise SamplingError(
                f"cell {cell!r} needs {wanted} frames but the pool holds only {supply.get(cell, 0)}. "
                f"Adjust --quota, or widen/narrow --stability-window."
            )

    solved = allocate.solve(
        availability=availability,
        participants=participants,
        tasks=tasks,
        cell_types=cell_types,
        cell_quota=cell_quota,
        task_target=task_target,
        min_per_participant=MIN_FRAMES_PER_PARTICIPANT,
        max_per_participant=MAX_FRAMES_PER_PARTICIPANT,
        pair_cap=MAX_FRAMES_PER_RECORDING,
        seed=seed,
    )

    selected_labels: list[int] = []
    per_recording: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (participant, task, cell), count in sorted(solved.counts.items()):
        candidates = pool[
            (pool["participant"] == participant) & (pool["task"] == task) & (pool["cell"] == cell)
        ]
        labels = _pick_spread(candidates, count, per_recording[(participant, task)], rng)
        selected_labels.extend(labels)
        per_recording[(participant, task)].extend(candidates.loc[labels, "timestamp_sec"].tolist())

    selection = pool.loc[selected_labels].copy()
    if len(selection) != total_frames:
        raise SamplingError(f"internal error: selected {len(selection)}, expected {total_frames}")

    # Inverse inclusion probability, so agreement can be re-weighted to the
    # population the frames were drawn from instead of the balanced design.
    population = pool["cell"].value_counts().to_dict()
    sampled = selection["cell"].value_counts().to_dict()
    selection["cell_population"] = selection["cell"].map(population).astype(int)
    selection["cell_sampled"] = selection["cell"].map(sampled).astype(int)
    selection["sampling_weight"] = selection["cell_population"] / selection["cell_sampled"]
    selection["sampling_weight"] = selection["sampling_weight"] / selection["sampling_weight"].mean()

    stats = {
        "population_by_cell": {str(k): int(v) for k, v in population.items()},
        "sampled_by_cell": {str(k): int(v) for k, v in sampled.items()},
        "pool_frames": int(len(pool)),
    }
    return selection.reset_index(drop=True), solved, stats


def anonymise(selection: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Shuffle, then name sequentially.

    Shuffling is what makes the names carry no information: unshuffled, file order
    would encode participant, task and band, and a rater could infer the answer from
    the sequence without looking at the image.
    """

    rng = random.Random(seed + 1)
    order = list(range(len(selection)))
    rng.shuffle(order)
    shuffled = selection.iloc[order].reset_index(drop=True)
    shuffled["frame_id"] = [
        COMPOSITE_NAME_TEMPLATE.format(index=index).replace(".jpg", "")
        for index in range(1, len(shuffled) + 1)
    ]
    shuffled["image_filename"] = [f"{name}.jpg" for name in shuffled["frame_id"]]
    return shuffled


# ---------------------------------------------------------------------------
# Pipeline detail lookup
# ---------------------------------------------------------------------------

_REBA_COMPONENTS = ("trunk", "neck", "legs", "upper_arm", "lower_arm", "wrist")
_RULA_COMPONENTS = ("upper_arm", "lower_arm", "wrist", "wrist_twist", "neck", "trunk", "legs")

#: Subscores a rater cannot judge from a still frame: load and coupling are task facts,
#: and activity / muscle use depend on what happens before and after the instant shown.
#: They are handed to the rater as givens so that the only thing free to differ between
#: expert and pipeline is the postural judgement.
_REBA_GIVENS = ("load_force", "coupling", "activity")
_RULA_GIVENS = ("force_load", "muscle_use")


def attach_pipeline_detail(selection: pd.DataFrame) -> pd.DataFrame:
    """Attach per-component pipeline scores and camera references to each frame.

    The CSV carries only the final scores.  Component-level comparison -- which body
    segment the disagreement lives in -- needs the intermediate scores, and correct
    compositing needs the per-camera source frame numbers; both are in the JSONL.
    """

    selection = selection.copy()
    for column in (
        *[f"pipeline_reba_{name}" for name in _REBA_COMPONENTS],
        *[f"pipeline_rula_{name}" for name in _RULA_COMPONENTS],
        *[f"given_reba_{name}" for name in _REBA_GIVENS],
        *[f"given_rula_{name}" for name in _RULA_GIVENS],
        "pipeline_reba_final",
        "pipeline_rula_final",
        "pipeline_reba_side",
        "assumed_load_kg",
        "assumed_coupling",
        "assumed_shock_or_rapid_force",
        "reba_missing_inputs",
    ):
        selection[column] = pd.Series([None] * len(selection), dtype="object")
    selection["camera_refs"] = pd.Series([None] * len(selection), dtype="object")

    for (jsonl_path, ), group in selection.groupby(["results_jsonl"]):
        wanted = set(group["frame_number"].astype(int))
        records = load_frame_details(Path(jsonl_path), wanted)
        for label, row in group.iterrows():
            record = records.get(int(row["frame_number"]))
            if record is None:
                continue
            reba = (record.get("reba") or {}).get("selected") or {}
            rula = (record.get("rula") or {}).get("selected") or {}
            intermediate_reba = reba.get("intermediate_scores") or {}
            intermediate_rula = rula.get("intermediate_scores") or {}
            for name in _REBA_COMPONENTS:
                selection.at[label, f"pipeline_reba_{name}"] = intermediate_reba.get(name)
            for name in _RULA_COMPONENTS:
                selection.at[label, f"pipeline_rula_{name}"] = intermediate_rula.get(name)
            for name in _REBA_GIVENS:
                selection.at[label, f"given_reba_{name}"] = intermediate_reba.get(name)
            for name in _RULA_GIVENS:
                selection.at[label, f"given_rula_{name}"] = intermediate_rula.get(name)
            selection.at[label, "pipeline_reba_final"] = reba.get("final_score")
            selection.at[label, "pipeline_rula_final"] = rula.get("final_score")
            selection.at[label, "pipeline_reba_side"] = (record.get("reba") or {}).get("selected_side")
            measurements = reba.get("raw_measurements") or {}
            selection.at[label, "assumed_load_kg"] = measurements.get("load_kg")
            selection.at[label, "assumed_coupling"] = measurements.get("coupling")
            selection.at[label, "assumed_shock_or_rapid_force"] = measurements.get("shock_or_rapid_force")
            selection.at[label, "reba_missing_inputs"] = ";".join(reba.get("missing_inputs") or [])
            selection.at[label, "camera_refs"] = camera_references(record)
    return selection


# ---------------------------------------------------------------------------
# Image extraction
# ---------------------------------------------------------------------------


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SamplingError(
            "OpenCV is required to write composites; install opencv-python or use --no-images"
        ) from exc
    return cv2


def _read_frames(video_path: Path, frame_numbers: list[int], allow_seek: bool) -> dict[int, Any]:
    """Decode specific source frame indices from one video.

    Indices are the decode indices the pipeline recorded, so sequential decoding
    reproduces them exactly.  Seeking is opt-in and verified, falling back to
    sequential decoding when the container reports a position it did not honour.
    """

    cv2 = _require_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SamplingError(f"OpenCV could not open video: {video_path}")
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

    images: dict[int, Any] = {}
    position = 0
    seek_enabled = allow_seek
    try:
        for target in sorted(frame_numbers):
            if target < position:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                position = 0
            if seek_enabled and target - position > SEEK_MIN_GAP_FRAMES:
                capture.set(cv2.CAP_PROP_POS_FRAMES, target)
                reported = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
                if reported == target:
                    position = target
                else:
                    LOGGER.warning(
                        "Seeking unreliable for %s (asked %d, landed %d); decoding sequentially",
                        video_path.name,
                        target,
                        reported,
                    )
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    position = 0
                    seek_enabled = False
            while position < target:
                if not capture.grab():
                    LOGGER.error("Video %s ended before frame %d", video_path.name, target)
                    return images
                position += 1
            ok, image = capture.read()
            if not ok:
                LOGGER.error("Could not decode frame %d of %s", target, video_path.name)
                return images
            position += 1
            images[target] = image
    finally:
        capture.release()
    return images


def _stitch(panels: list[Any], height: int = COMPOSITE_HEIGHT_PX) -> tuple[Any, list[list[int]]]:
    """Resize panels to a common height and concatenate horizontally, drawing nothing.

    Also returns each panel's ``[x_start, x_end]`` in the composite.  Cameras do not
    necessarily share an aspect ratio, so the panels are not thirds; the scoring app
    needs the real boundaries to be able to enlarge one view at a time.
    """

    cv2 = _require_cv2()
    import numpy as np

    resized = []
    for panel in panels:
        source_height, source_width = panel.shape[:2]
        width = max(1, int(round(source_width * height / source_height)))
        interpolation = cv2.INTER_AREA if source_height > height else cv2.INTER_CUBIC
        resized.append(cv2.resize(panel, (width, height), interpolation=interpolation))

    bounds: list[list[int]] = []
    offset = 0
    pieces: list[Any] = []
    separator = (
        np.full((height, PANEL_SEPARATOR_PX, 3), PANEL_SEPARATOR_COLOR, dtype=resized[0].dtype)
        if PANEL_SEPARATOR_PX > 0
        else None
    )
    for index, panel in enumerate(resized):
        if index and separator is not None:
            pieces.append(separator)
            offset += PANEL_SEPARATOR_PX
        pieces.append(panel)
        bounds.append([offset, offset + panel.shape[1]])
        offset += panel.shape[1]
    return cv2.hconcat(pieces), bounds


def write_composites(
    selection: pd.DataFrame, output_dir: Path, allow_seek: bool = ALLOW_FRAME_SEEK
) -> pd.DataFrame:
    """Write one clean three-panel composite per selected frame.

    Each video is opened once and read in ascending frame order, so a recording is
    decoded at most once regardless of how many frames it contributes.
    """

    cv2 = _require_cv2()
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # video path -> source frame numbers needed from it
    plan: dict[str, set[int]] = defaultdict(set)
    for _, row in selection.iterrows():
        for reference in row["camera_refs"] or []:
            plan[str(reference["source_path"])].add(int(reference["source_frame_number"]))
    if not plan:
        raise SamplingError(
            "no camera references were resolved; the JSONL files are required to build composites"
        )

    LOGGER.info("Decoding %d video file(s) for %d frame(s)", len(plan), len(selection))
    decoded: dict[str, dict[int, Any]] = {}
    for index, (video_path, numbers) in enumerate(sorted(plan.items()), start=1):
        LOGGER.info("[%d/%d] %s (%d frame(s))", index, len(plan), Path(video_path).name, len(numbers))
        try:
            decoded[video_path] = _read_frames(Path(video_path), sorted(numbers), allow_seek)
        except SamplingError as exc:
            LOGGER.error("%s", exc)
            decoded[video_path] = {}

    statuses: list[str] = []
    geometry: list[str] = []
    widths: list[Any] = []
    for _, row in selection.iterrows():
        references = row["camera_refs"] or []
        panels = []
        for reference in references:
            frames = decoded.get(str(reference["source_path"]), {})
            panels.append(frames.get(int(reference["source_frame_number"])))
        if len(panels) != 3 or any(panel is None for panel in panels):
            LOGGER.error(
                "Missing camera view(s) for %s/%s frame %s; composite not written",
                row["participant"],
                row["task"],
                row["frame_number"],
            )
            statuses.append("failed_missing_view")
            geometry.append("")
            widths.append(None)
            continue
        composite, bounds = _stitch(panels)
        destination = images_dir / str(row["image_filename"])
        if not cv2.imwrite(str(destination), composite, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
            LOGGER.error("Could not write %s", destination)
            statuses.append("failed_write")
            geometry.append("")
            widths.append(None)
            continue
        statuses.append("ok")
        geometry.append(json.dumps(bounds))
        widths.append(int(composite.shape[1]))

    result = selection.copy()
    result["extraction_status"] = statuses
    result["panel_bounds"] = geometry
    result["composite_width"] = widths
    return result


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

KEY_COLUMNS = (
    "frame_id",
    "image_filename",
    "participant",
    "task",
    "task_index",
    "study",
    "fps_variant",
    "frame_number",
    "timestamp_sec",
    "stratum",
    "band",
    "band_run_frames",
    "band_run_sec",
    "frames_to_band_change",
    "sampling_weight",
    "cell_population",
    "cell_sampled",
    "reba_score",
    "rula_score",
    "pipeline_reba_final",
    "pipeline_rula_final",
    "pipeline_reba_side",
    *[f"pipeline_reba_{name}" for name in _REBA_COMPONENTS],
    *[f"pipeline_rula_{name}" for name in _RULA_COMPONENTS],
    *[f"given_reba_{name}" for name in _REBA_GIVENS],
    *[f"given_rula_{name}" for name in _RULA_GIVENS],
    "assumed_load_kg",
    "assumed_coupling",
    "assumed_shock_or_rapid_force",
    "reba_missing_inputs",
    "reba_status",
    "rula_status",
    "pose_mean_visibility",
    "results_csv",
)


def export_ground_truth(selection: pd.DataFrame, output_dir: Path) -> Path:
    """Write the un-blinded key linking each composite back to its source data."""

    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [name for name in KEY_COLUMNS if name in selection.columns]
    for extra in ("extraction_status", "panel_bounds", "composite_width"):
        if extra in selection.columns:
            columns.append(extra)
    table = selection[columns].sort_values("frame_id").reset_index(drop=True)
    destination = output_dir / GROUND_TRUTH_FILENAME
    table.to_csv(destination, index=False)
    LOGGER.info("Wrote %s", destination)
    return destination


def build_report(selection: pd.DataFrame, solved: allocate.Allocation) -> dict[str, Any]:
    """Summarise the realised design so it can be verified at a glance."""

    participant_counts = selection["participant"].value_counts().sort_index()
    cross = (
        selection.pivot_table(
            index="task", columns="cell", values="frame_number", aggfunc="count", fill_value=0
        )
        .astype(int)
    )
    return {
        "total_frames": int(len(selection)),
        "by_stratum": {k: int(v) for k, v in selection["stratum"].value_counts().items()},
        "by_band": {k: int(v) for k, v in selection["band"].value_counts().items()},
        "by_cell": {k: int(v) for k, v in selection["cell"].value_counts().items()},
        "by_task": {k: int(v) for k, v in selection["task"].value_counts().items()},
        "task_target": solved.task_target,
        "task_by_cell": {task: row.to_dict() for task, row in cross.iterrows()},
        "by_participant": {str(k): int(v) for k, v in participant_counts.items()},
        "participants_covered": int(participant_counts.size),
        "recordings_used": solved.recordings_used,
        "recordings_contributing_multiple": solved.recordings_doubled,
        "band_run_sec": {
            "min": float(selection["band_run_sec"].min()),
            "median": float(selection["band_run_sec"].median()),
            "max": float(selection["band_run_sec"].max()),
        },
        "notes": solved.notes,
    }


def export_manifest(
    report: dict[str, Any], settings: dict[str, Any], provenance: dict[str, Any],
    stats: dict[str, Any], output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "settings": settings,
        "pool_provenance": provenance,
        "sampling_statistics": stats,
        "design": report,
        "camera_panel_order": list(CAMERA_ROLE_ORDER),
        "composite_height_px": COMPOSITE_HEIGHT_PX,
    }
    destination = output_dir / MANIFEST_FILENAME
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s", destination)
    return destination


def print_report(report: dict[str, Any]) -> None:
    print("\nRealised design")
    print("---------------")
    print(f"  total frames   : {report['total_frames']}")
    print(f"  by stratum     : {report['by_stratum']}")
    print(f"  by band        : {report['by_band']}")
    print(f"  by cell        : {report['by_cell']}")
    print(f"  by task        : {report['by_task']}   (target {report['task_target']})")
    counts = report["by_participant"]
    print(
        f"  participants   : {report['participants_covered']} covered, "
        f"min {min(counts.values())}, max {max(counts.values())} frames each"
    )
    print(
        f"  recordings     : {report['recordings_used']} used, "
        f"{report['recordings_contributing_multiple']} contributing more than one frame"
    )
    run = report["band_run_sec"]
    print(f"  band run (sec) : min {run['min']:.1f}, median {run['median']:.1f}, max {run['max']:.1f}")
    for note in report["notes"]:
        print(f"  NOTE: {note}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_quota(text: str | None) -> dict[str, dict[str, int]]:
    if not text:
        return {k: dict(v) for k, v in DEFAULT_QUOTA.items()}
    quota: dict[str, dict[str, int]] = {stratum: {} for stratum in STRATA}
    for item in text.split(","):
        key, _, value = item.strip().partition("=")
        stratum, _, band = key.strip().partition(":")
        if stratum not in STRATA or band not in RISK_BANDS:
            raise SamplingError(f"bad quota term {item!r}; expected e.g. sustained:Low=12")
        quota[stratum][band] = int(value)
    return quota


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--fps", default=None,
        help="processing variant to sample from; default picks the most complete one",
    )
    parser.add_argument(
        "--quota", default=None,
        help="override quotas, e.g. 'sustained:Low=12,sustained:Medium=12,...'",
    )
    parser.add_argument(
        "--stability-window", type=int, default=DEFAULT_STABILITY_HALF_WINDOW,
        help="half-window in samples; a frame is sustained when its band holds either side",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--no-images", action="store_true", help="solve the design without decoding video")
    parser.add_argument(
        "--allow-seek", action="store_true", default=ALLOW_FRAME_SEEK,
        help="use verified keyframe seeking instead of sequential decoding (faster)",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )

    try:
        quota = parse_quota(args.quota)
        pool, provenance = build_pool(
            args.results_dir, args.fps, stability_half_window=args.stability_window
        )
        selection, solved, stats = select_frames(pool, quota, args.seed)
        selection = anonymise(selection, args.seed)
        selection = attach_pipeline_detail(selection)
        if not args.no_images:
            selection = write_composites(selection, args.output_dir, args.allow_seek)
    except (PoolError, SamplingError, allocate.AllocationError) as exc:
        LOGGER.error("%s", exc)
        return 1

    report = build_report(selection, solved)
    export_ground_truth(selection, args.output_dir)
    export_manifest(
        report,
        {
            "results_dir": str(args.results_dir),
            "output_dir": str(args.output_dir),
            "fps_variant": provenance["fps_variant"],
            "quota": quota,
            "seed": args.seed,
            "stability_half_window": args.stability_window,
            "min_frames_per_participant": MIN_FRAMES_PER_PARTICIPANT,
            "max_frames_per_participant": MAX_FRAMES_PER_PARTICIPANT,
            "max_frames_per_recording": MAX_FRAMES_PER_RECORDING,
            "min_seconds_between_frames": MIN_SECONDS_BETWEEN_FRAMES,
            "images_written": not args.no_images,
            "frame_seek_allowed": bool(args.allow_seek),
        },
        provenance,
        stats,
        args.output_dir,
    )
    print_report(report)

    failures = []
    if "extraction_status" in selection.columns:
        failures = selection.loc[selection["extraction_status"] != "ok", "frame_id"].tolist()
    if failures:
        LOGGER.error(
            "%d composite(s) could not be written: %s. Re-run with the same --seed to "
            "reproduce the identical selection once the source video issue is fixed.",
            len(failures),
            ", ".join(failures[:10]),
        )
        return 1
    print(f"\nOutput directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
