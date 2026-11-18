"""Candidate pool construction for the blind expert validation sample.

The pool is every per-frame REBA assessment that is a legitimate thing to put in
front of a human rater.  Two properties distinguish it from a naive scan of the
results tree, and both exist because of what the data turned out to look like:

*Archived runs are excluded.*  ``batch_process.py`` retires a superseded run by
renaming its directory to ``.<task>.archive-<tag>``.  A plain ``rglob`` picks those
up and, because the directory name still parses as a task, they arrive downstream
as extra tasks competing for the task quota.

*Every frame carries a stability flag.*  A REBA band assigned to a single frame is
not the same evidence as one that holds for a second.  Measured on this dataset,
the rare bands are overwhelmingly the former: in ``task3_rula_screwdriving`` the
median run of consecutive "High" frames is one frame, and no "High" frame there
survives smoothing against its own neighbours.  Sampling that stratifies on band
alone therefore selects pose-estimation transients for exactly the cells it has
the most trouble filling.  Stability is recorded per frame so the design can
sample the two kinds deliberately instead of confounding them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ergonomics.reba import REBACalculator

LOGGER = logging.getLogger(__name__)

FPS_VARIANTS: tuple[str, ...] = ("native", "10fps", "15fps", "30fps")
RISK_BANDS: tuple[str, ...] = ("Low", "Medium", "High")

#: ``*_status`` values that count as a usable assessment (ergonomics/common.py).
ACCEPTED_STATUSES = frozenset({"valid", "valid_with_limitations"})

#: Collapse the calculator's five REBA risk levels onto the three reporting bands.
#: Keyed by the level names in ``REBACalculator.RISK_LEVELS`` so that a change there
#: raises instead of silently re-banding the study.
_BAND_COLLAPSE = {
    "Negligible": "Low",
    "Low": "Low",
    "Medium": "Medium",
    "High": "High",
    "Very High": "High",
}

#: A frame is "sustained" when its band is unchanged across this many samples either
#: side of it.  At 10 fps, 2 means the band holds for 0.5 s.  Chosen because it is the
#: weakest criterion that still excludes single-frame artefacts, which is what the
#: rare bands are made of; the realised run length is exported per frame so the
#: threshold can be varied in sensitivity analysis rather than taken on faith.
DEFAULT_STABILITY_HALF_WINDOW = 2

#: Guard against a window that spans a decoding gap rather than real elapsed time.
STABILITY_MAX_WINDOW_SLACK = 2.5

REQUIRED_COLUMNS = (
    "frame_number",
    "timestamp_sec",
    "reba_status",
    "reba_score",
    "rula_status",
    "rula_score",
    "pose_mean_visibility",
    "input_source",
)


class PoolError(RuntimeError):
    """Raised when the results tree cannot yield a usable candidate pool."""


def reba_band_lookup() -> dict[int, str]:
    """Map every REBA final score to a three-way band, derived from the calculator."""

    lookup: dict[int, str] = {}
    for score, level in REBACalculator.RISK_LEVELS.items():
        risk_name = level[0]
        if risk_name not in _BAND_COLLAPSE:
            raise PoolError(
                f"REBACalculator.RISK_LEVELS defines risk level {risk_name!r}, which "
                f"validation/pool.py does not know how to band. Update _BAND_COLLAPSE."
            )
        lookup[int(score)] = _BAND_COLLAPSE[risk_name]
    return lookup


REBA_BAND = reba_band_lookup()


def band_of(score: float | None) -> str | None:
    """Band a single REBA score, or ``None`` when it is missing or out of range."""

    if score is None or (isinstance(score, float) and not np.isfinite(score)):
        return None
    return REBA_BAND.get(int(round(score)))


def task_index(task_folder: str) -> int:
    """Extract the numeric index from folder names such as ``task2_reba_stocking``."""

    digits = ""
    for character in task_folder:
        if character.isdigit():
            digits += character
        elif digits:
            break
    return int(digits) if digits else 0


def _is_archived(parts: tuple[str, ...]) -> bool:
    """Archived runs are hidden by a leading dot on some path component."""

    return any(part.startswith(".") for part in parts)


def discover_result_tables(base_results_dir: Path, fps_variant: str | None) -> list[dict[str, Any]]:
    """Locate live ``frame_results.csv`` files and recover their study keys."""

    if not base_results_dir.is_dir():
        raise PoolError(f"results directory does not exist: {base_results_dir}")

    tables: list[dict[str, Any]] = []
    archived = 0
    for csv_path in sorted(base_results_dir.rglob("frame_results.csv")):
        parts = csv_path.relative_to(base_results_dir).parts
        if _is_archived(parts):
            archived += 1
            continue
        if len(parts) == 5:
            variant, study, participant, task = parts[0], parts[1], parts[2], parts[3]
        elif len(parts) == 4:
            # batch_process.py writes without the FPS prefix; treat that as native.
            variant, study, participant, task = "native", parts[0], parts[1], parts[2]
        else:
            LOGGER.debug("Ignoring unexpected results path: %s", csv_path)
            continue
        if variant not in FPS_VARIANTS:
            LOGGER.debug("Ignoring unrecognised FPS variant %r at %s", variant, csv_path)
            continue
        if fps_variant is not None and variant != fps_variant:
            continue
        tables.append(
            {
                "csv_path": csv_path,
                "jsonl_path": csv_path.with_name("frame_results.jsonl"),
                "fps_variant": variant,
                "study": study,
                "participant": participant,
                "task": task,
            }
        )
    if archived:
        LOGGER.info("Skipped %d archived result table(s)", archived)
    return tables


def summarise_variants(base_results_dir: Path) -> dict[str, dict[str, int]]:
    """Report coverage per FPS variant, used to pick or validate the variant."""

    recordings: dict[str, int] = {}
    participants: dict[str, set[str]] = {}
    for table in discover_result_tables(base_results_dir, None):
        variant = str(table["fps_variant"])
        recordings[variant] = recordings.get(variant, 0) + 1
        participants.setdefault(variant, set()).add(str(table["participant"]))
    return {
        variant: {"recordings": count, "participants": len(participants[variant])}
        for variant, count in recordings.items()
    }


def choose_variant(base_results_dir: Path, requested: str | None) -> str:
    """Resolve the FPS variant, preferring the most complete one when unspecified."""

    summary = summarise_variants(base_results_dir)
    if not summary:
        raise PoolError(f"no live frame_results.csv found under {base_results_dir}")
    if requested is not None:
        if requested not in summary:
            available = ", ".join(
                f"{name} ({info['participants']}p/{info['recordings']}rec)"
                for name, info in sorted(summary.items())
            )
            raise PoolError(
                f"fps variant {requested!r} has no live results under {base_results_dir}. "
                f"Available: {available}"
            )
        return requested
    best = max(summary.items(), key=lambda kv: (kv[1]["participants"], kv[1]["recordings"]))
    LOGGER.info(
        "Auto-selected fps variant %r (%d participants, %d recordings); others: %s",
        best[0],
        best[1]["participants"],
        best[1]["recordings"],
        {k: v for k, v in summary.items() if k != best[0]} or "none",
    )
    return best[0]


def _resolve_camera_videos(input_source: Any) -> tuple[Path, ...]:
    """Split the recorded ``input_source`` into existing camera video paths."""

    if not isinstance(input_source, str) or not input_source.strip():
        return ()
    candidates = tuple(Path(part.strip()) for part in input_source.split(" | ") if part.strip())
    return candidates if all(path.is_file() for path in candidates) else ()


def _mark_stability(
    frame: pd.DataFrame, half_window: int, expected_step_sec: float | None
) -> pd.DataFrame:
    """Flag sustained frames and record the run length each frame belongs to.

    ``sustained`` means the band is identical across ``half_window`` samples either
    side.  The window must also span roughly the elapsed time it should, so a gap in
    the decoded sequence cannot masquerade as a held posture.
    """

    frame = frame.sort_values("frame_number").reset_index(drop=True)
    bands = frame["band"].to_numpy(dtype=object)
    times = frame["timestamp_sec"].to_numpy(dtype=float)
    count = len(frame)

    sustained = np.zeros(count, dtype=bool)
    # A frame within half_window of either end of the recording, or spanning a decoding
    # gap, cannot be judged either way.  It is not transient -- it is unverifiable, and
    # calling it transient would put frames from the middle of long stable runs into the
    # stratum that exists to isolate pose instability.
    verifiable = np.zeros(count, dtype=bool)
    if half_window <= 0:
        sustained[:] = True
        verifiable[:] = True
    else:
        span_limit = None
        if expected_step_sec:
            span_limit = 2 * half_window * expected_step_sec * STABILITY_MAX_WINDOW_SLACK
        for index in range(count):
            low, high = index - half_window, index + half_window + 1
            if low < 0 or high > count:
                continue
            if span_limit is not None and (times[high - 1] - times[low]) > span_limit:
                continue
            verifiable[index] = True
            window = bands[low:high]
            if window[0] is not None and all(value == window[0] for value in window):
                sustained[index] = True

    # Samples until the band next differs, in either direction.  This is what actually
    # decides the stratum, and unlike the run length it cannot look like it contradicts
    # it: a frame can sit inside a minute-long run and still have a spike two samples away.
    to_change = np.full(count, np.nan)
    for index in range(count):
        for distance in range(1, count):
            before = index - distance
            after = index + distance
            differs = (before >= 0 and bands[before] != bands[index]) or (
                after < count and bands[after] != bands[index]
            )
            if differs:
                to_change[index] = distance
                break
            if before < 0 and after >= count:
                break

    # Length of the contiguous same-band run containing each frame.
    run_id = np.zeros(count, dtype=int)
    if count:
        changes = [False] + [bands[i] != bands[i - 1] for i in range(1, count)]
        run_id = np.cumsum(changes)
    run_frames = pd.Series(run_id).map(pd.Series(run_id).value_counts()).to_numpy()
    run_seconds = np.zeros(count, dtype=float)
    for identifier in np.unique(run_id):
        mask = run_id == identifier
        window_times = times[mask]
        run_seconds[mask] = float(window_times.max() - window_times.min()) if mask.sum() > 1 else 0.0

    frame["sustained"] = sustained
    frame["stability_verifiable"] = verifiable
    frame["frames_to_band_change"] = to_change
    frame["band_run_frames"] = run_frames
    frame["band_run_sec"] = run_seconds
    return frame


def build_pool(
    base_results_dir: Path,
    fps_variant: str | None = None,
    stability_half_window: int = DEFAULT_STABILITY_HALF_WINDOW,
    min_pose_visibility: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load every frame that is a legitimate candidate, with a stability flag.

    Only frames the pipeline itself scored with REBA are admitted.  Admitting a
    frame on its RULA score instead -- as an "either assessment is usable" rule would
    -- rescues precisely the frames REBA rejected for ``insufficient_pose_quality``
    and labels them with a different instrument, leaving the stratification variable
    meaning two different things on 3% of the sample.

    Returns the pool and a provenance dict describing how it was filtered.
    """

    variant = choose_variant(base_results_dir, fps_variant)
    tables = discover_result_tables(base_results_dir, variant)
    if not tables:
        raise PoolError(f"no live results for fps variant {variant!r} under {base_results_dir}")

    frames: list[pd.DataFrame] = []
    missing_video: list[str] = []
    raw_rows = 0
    for table in tables:
        try:
            raw = pd.read_csv(table["csv_path"])
        except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            LOGGER.warning("Could not read %s: %s", table["csv_path"], exc)
            continue
        if raw.empty:
            continue
        missing_columns = [name for name in REQUIRED_COLUMNS if name not in raw.columns]
        if missing_columns:
            LOGGER.warning("%s is missing columns %s; skipped", table["csv_path"], missing_columns)
            continue
        raw_rows += len(raw)

        videos = _resolve_camera_videos(raw["input_source"].iloc[0])
        if len(videos) != 3:
            missing_video.append(f"{table['participant']}/{table['task']}")
            continue

        data = pd.DataFrame(
            {
                "study": table["study"],
                "participant": table["participant"],
                "task": table["task"],
                "task_index": task_index(table["task"]),
                "fps_variant": table["fps_variant"],
                "frame_number": pd.to_numeric(raw["frame_number"], errors="coerce"),
                "timestamp_sec": pd.to_numeric(raw["timestamp_sec"], errors="coerce"),
                "reba_score": pd.to_numeric(raw["reba_score"], errors="coerce"),
                "rula_score": pd.to_numeric(raw["rula_score"], errors="coerce"),
                "reba_status": raw["reba_status"].astype("string"),
                "rula_status": raw["rula_status"].astype("string"),
                "reba_selected_side": raw.get("reba_selected_side", pd.Series(dtype="string")).astype("string"),
                "pose_mean_visibility": pd.to_numeric(raw["pose_mean_visibility"], errors="coerce"),
            }
        )
        data["results_csv"] = str(table["csv_path"])
        data["results_jsonl"] = str(table["jsonl_path"])
        frames.append(data)

    if missing_video:
        LOGGER.warning(
            "Skipped %d recording(s) without 3 resolvable camera videos: %s",
            len(missing_video),
            ", ".join(missing_video[:6]),
        )
    if not frames:
        raise PoolError("no recording survived video resolution; check input_source paths")

    pool = pd.concat(frames, ignore_index=True)
    counts = {"raw_rows": int(raw_rows), "after_video_resolution": int(len(pool))}

    pool = pool[pool["reba_status"].isin(ACCEPTED_STATUSES) & pool["reba_score"].notna()]
    counts["reba_scored"] = int(len(pool))
    pool = pool[pool["frame_number"].notna() & pool["timestamp_sec"].notna()]
    counts["with_frame_and_timestamp"] = int(len(pool))

    if min_pose_visibility is not None:
        before = len(pool)
        pool = pool[pool["pose_mean_visibility"].isna() | (pool["pose_mean_visibility"] >= min_pose_visibility)]
        counts["after_visibility_filter"] = int(len(pool))
        if len(pool) == before:
            LOGGER.warning(
                "min_pose_visibility=%s removed no frames (observed minimum %.3f). "
                "MediaPipe visibility is self-reported and near-saturated here, so it is "
                "not an image-quality gate; do not report it as one.",
                min_pose_visibility,
                float(pool["pose_mean_visibility"].min()) if len(pool) else float("nan"),
            )

    pool = pool.assign(band=[band_of(score) for score in pool["reba_score"].tolist()])
    pool = pool[pool["band"].notna()]
    counts["banded"] = int(len(pool))
    pool = pool.drop_duplicates(subset=["study", "participant", "task", "frame_number"])
    pool["frame_number"] = pool["frame_number"].astype(int)

    if pool.empty:
        raise PoolError("every frame was filtered out; relax the thresholds")

    step = float(pool.groupby(["participant", "task"])["timestamp_sec"].diff().median())
    marked = [
        _mark_stability(group, stability_half_window, step)
        for _, group in pool.groupby(["participant", "task"], sort=True)
    ]
    pool = pd.concat(marked, ignore_index=True)

    unverifiable = int((~pool["stability_verifiable"]).sum())
    if unverifiable:
        LOGGER.info(
            "Dropped %d frame(s) at recording edges or across decoding gaps: their stability "
            "cannot be judged, and treating them as transient would mix stable postures into "
            "the stratum that measures instability",
            unverifiable,
        )
    pool = pool[pool["stability_verifiable"]].reset_index(drop=True)
    counts["stability_verifiable"] = int(len(pool))
    if pool.empty:
        raise PoolError("no frame had a complete stability window; lower --stability-window")

    provenance = {
        "fps_variant": variant,
        "recordings": int(pool.groupby(["participant", "task"]).ngroups),
        "participants": int(pool["participant"].nunique()),
        "tasks": sorted(pool["task"].unique()),
        "filter_counts": counts,
        "stability_half_window": stability_half_window,
        "median_sample_step_sec": step,
        "sustained_frames": int(pool["sustained"].sum()),
        "transient_frames": int((~pool["sustained"]).sum()),
        "band_prevalence": {
            band: int(count) for band, count in pool["band"].value_counts().items()
        },
        "recordings_without_three_videos": missing_video,
    }
    LOGGER.info(
        "Pool: %d frames from %d recordings / %d participants (%d sustained, %d transient)",
        len(pool),
        provenance["recordings"],
        provenance["participants"],
        provenance["sustained_frames"],
        provenance["transient_frames"],
    )
    return pool.reset_index(drop=True), provenance


def load_frame_details(
    jsonl_path: Path, frame_numbers: set[int]
) -> dict[int, dict[str, Any]]:
    """Pull the per-frame record for specific frames out of a results JSONL.

    The JSONL carries what the CSV does not: the REBA/RULA component scores the
    pipeline arrived at, the task conditions it assumed, and an explicit per-camera
    frame reference.  Only the requested frames are retained -- these files run to
    tens of megabytes each.
    """

    found: dict[int, dict[str, Any]] = {}
    if not jsonl_path.is_file():
        LOGGER.warning("No JSONL beside the CSV: %s", jsonl_path)
        return found
    try:
        with jsonl_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                number = record.get("frame_number")
                if number in frame_numbers:
                    found[int(number)] = record
                    if len(found) == len(frame_numbers):
                        break
    except (OSError, UnicodeError) as exc:
        LOGGER.warning("Could not read %s: %s", jsonl_path, exc)
    missing = frame_numbers - set(found)
    if missing:
        LOGGER.warning("%s did not contain frames %s", jsonl_path, sorted(missing)[:10])
    return found


def camera_references(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return per-camera decode references for a frame, ordered front, side, iso.

    Using the reference the pipeline recorded removes the need to infer which file is
    which camera from sort order, and carries the per-camera source frame number so
    the composite is built from the frames that were actually scored.
    """

    references = record.get("camera_frame_references") or []
    ordered: list[dict[str, Any]] = []
    by_role = {str(ref.get("camera_role")): ref for ref in references if isinstance(ref, dict)}
    for role in ("front", "side", "iso"):
        if role in by_role:
            ordered.append(by_role[role])
    if len(ordered) == len(references) and ordered:
        return ordered
    # Fall back to declared camera_index order when roles are absent or unexpected.
    return sorted(
        (ref for ref in references if isinstance(ref, dict)),
        key=lambda ref: ref.get("camera_index", 0),
    )
