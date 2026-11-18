"""Choosing which frames the vision model actually looks at.

A vision model's frame budget is its scarcest resource: a 110-second trial holds
834 scored frames and a scene pass can afford a dozen. How those twelve are
chosen decides what the model can possibly notice.

Sampling uniformly in *time* is the obvious approach and, for this domain, a
weak one. These are repetitive tasks -- fifteen near-identical stocking cycles --
so evenly spaced frames cluster onto whatever posture the cycle spends longest
in, and a posture that occupies four seconds of the trial may get no frame at
all.

This module samples in *posture* instead. The pipeline already measured 22 joint
angles per frame, which makes the recording a point cloud, and "which twelve
frames best represent this cloud" is the k-medoid problem. It is solved here with
the greedy facility-location rule: repeatedly take the frame that most reduces
the total distance from every other frame to its nearest chosen one. That is
submodular, so greedy is within (1 - 1/e) of optimal, and it is deterministic --
no seed, no iteration to convergence, nothing to record for reproducibility.

The moments the deterministic analysis already calls important are forced into
the set first, because coverage is not a reason to omit the worst frame of the
trial.

:func:`coverage` reports the result against time-spaced sampling of the same size
rather than asserting the improvement. An early draft of this module used
farthest-point sampling, and that measurement is what showed it was optimising
for outliers instead of representativeness.

Nothing here estimates a posture. Every coordinate is a measurement the
deterministic pipeline already wrote.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from api import analysis as analysis_api
from api.sessions import SessionError
from dashboard.cache import ttl_cache

LOGGER = logging.getLogger(__name__)

#: The angles that describe a working posture. Wrist pronation is excluded on
#: purpose: ``core/posture_conditions.py`` records that it is measured without an
#: anatomical zero, so its magnitude is not comparable between frames and it
#: would dominate a distance built from raw degrees.
POSTURE_AXES: tuple[str, ...] = (
    "trunk_flexion",
    "trunk_side_bend",
    "trunk_twist",
    "neck_flexion",
    "neck_side_bend",
    "neck_rotation",
    "left_upper_arm_flexion",
    "left_upper_arm_abduction",
    "left_shoulder_elevation",
    "left_forearm_flexion",
    "left_wrist_flexion",
    "left_wrist_deviation",
    "right_upper_arm_flexion",
    "right_upper_arm_abduction",
    "right_shoulder_elevation",
    "right_forearm_flexion",
    "right_wrist_flexion",
    "right_wrist_deviation",
    "left_knee_flexion",
    "right_knee_flexion",
)

#: A full pairwise distance matrix is O(n^2) in memory. Native-rate trials can
#: run to thousands of frames, so the candidate pool is thinned evenly above this
#: before the matrix is built. Still two orders of magnitude above any budget.
MAX_CANDIDATES = 1200


@dataclass(frozen=True, slots=True)
class Sample:
    """One frame chosen for the model to look at, and why it was chosen."""

    frame: int
    sec: float
    score: int | None
    reason: str

    def payload(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "sec": round(self.sec, 2),
            "score": self.score,
            "reason": self.reason,
        }


@ttl_cache(ttl=180, copy_result=False)
def posture_cloud(session_id: str) -> tuple[list[int], list[float], Any]:
    """(frame numbers, timestamps, standardised angle matrix) for a session."""

    tracked = next(iter(analysis_api.people(session_id)), None)
    records = analysis_api._records(session_id, tracked)
    frames: list[int] = []
    times: list[float] = []
    rows: list[list[float]] = []

    for record in records:
        number = record.get("frame_number")
        stamp = record.get("timestamp_sec")
        angles = record.get("angles") or {}
        if not isinstance(number, int) or not isinstance(stamp, int | float):
            continue
        vector: list[float] = []
        for axis in POSTURE_AXES:
            value = angles.get(axis)
            if isinstance(value, bool) or not isinstance(value, int | float):
                break
            number_value = float(value)
            if not math.isfinite(number_value):
                break
            vector.append(number_value)
        else:
            frames.append(int(number))
            times.append(float(stamp))
            rows.append(vector)

    if not rows:
        return [], [], np.empty((0, 0))

    matrix = np.asarray(rows, dtype=np.float64)
    if len(frames) > MAX_CANDIDATES:
        keep = np.linspace(0, len(frames) - 1, MAX_CANDIDATES).astype(int)
        frames = [frames[i] for i in keep]
        times = [times[i] for i in keep]
        matrix = matrix[keep]

    # Standardise per axis. Without it, knee flexion swinging through 90 degrees
    # drowns out a wrist deviation of 20, and the selection describes the legs of
    # a task whose risk is in the shoulder.
    spread = matrix.std(axis=0)
    spread[spread < 1e-9] = 1.0
    return frames, times, (matrix - matrix.mean(axis=0)) / spread


def _distance_matrix(cloud: Any) -> Any:
    """Squared euclidean distances between every pair of frames."""

    square = np.einsum("ij,ij->i", cloud, cloud)
    distances = square[:, None] + square[None, :] - 2.0 * (cloud @ cloud.T)
    np.maximum(distances, 0.0, out=distances)
    return distances


def representative_frames(
    session_id: str, budget: int, *, holdout: tuple[int, int] | None = None
) -> list[Sample]:
    """Frames chosen to represent the postures the recording actually contains.

    Seeded with the moments the deterministic analysis already calls important --
    the worst frame, the calmest frame, each episode peak -- then filled by
    greedy facility-location over the standardised angle space.

    ``holdout`` is ``(fold, folds)`` and withholds every ``folds``-th candidate
    starting at ``fold``, so each pass selects from a different slice of the
    recording. That is what makes the agreement measured in
    :mod:`assistant.consistency` mean anything: an earlier version rotated
    tie-breaking instead, which was too weak to change the greedy trajectory and
    handed all three passes the identical twelve frames. Agreement between
    identical prompts measures decoder noise, not whether the footage supports a
    reading.

    The forced seeds survive every fold. A pass that has not seen the worst
    moment of the trial is not a second opinion about it.
    """

    budget = max(1, int(budget))
    try:
        timeline = analysis_api.timeline(session_id, "reba")
        upper_limb = analysis_api.timeline(session_id, "rula")
    except SessionError:
        return []

    frames, times, cloud = posture_cloud(session_id)
    points = timeline["series"]["points"] or upper_limb["series"]["points"]

    if not frames:
        # Incomplete angle sets and withheld scores can still have readable pixels.
        if not points:
            points = [{"f": row["frame_number"], "t": row["timestamp_sec"]}
                      for row in analysis_api._records(session_id, timeline.get("person"))
                      if isinstance(row.get("frame_number"), int)
                      and isinstance(row.get("timestamp_sec"), (int, float))]
        indices = np.linspace(0, len(points) - 1, min(len(points), budget)).astype(int)
        return [Sample(points[i]["f"], points[i]["t"], points[i].get("s"),
                       "time-spaced (no angle data)") for i in indices]

    index_of = {frame: position for position, frame in enumerate(frames)}
    scores = {int(p["f"]): p.get("s") for p in points}

    chosen: list[int] = []
    reasons: dict[int, str] = {}

    def take(frame: Any, reason: str) -> bool:
        """Snap to the nearest surviving candidate; the pool may have been thinned."""

        if len(chosen) >= budget or not isinstance(frame, int) or frame in reasons:
            return False
        if frame not in index_of:
            nearby = min(frames, key=lambda candidate: abs(candidate - frame))
            if abs(nearby - frame) > 25 or nearby in reasons:
                return False
            frame = nearby
        chosen.append(frame)
        reasons[frame] = reason
        return True

    verdict = timeline["verdict"]
    take(verdict.get("peakFrame"), "worst scored moment")
    take(upper_limb["verdict"].get("peakFrame"), "RULA peak")
    for episode in timeline["episodes"][:3]:
        take(episode.get("peakFrame"), f"peak of episode {episode.get('index')}")
    if points:
        calmest = min(points, key=lambda p: (p.get("s") or 0, p["t"]))
        take(int(calmest["f"]), "lowest scored moment")
    if not chosen:
        take(frames[0], "start of recording")

    # Withhold this pass's fold from the candidates the greedy fill may use.
    withheld = np.zeros(len(frames), dtype=bool)
    if holdout is not None:
        fold, folds = holdout
        if folds > 1:
            withheld = (np.arange(len(frames)) % folds) == (fold % folds)

    distances = _distance_matrix(cloud)
    nearest = distances[:, [index_of[frame] for frame in chosen]].min(axis=1)

    # Greedy facility location: take the frame that most reduces the total
    # distance from every frame to its nearest chosen one. Submodular, so greedy
    # is within (1 - 1/e) of the optimal set of this size.
    while len(chosen) < budget and len(chosen) < len(frames):
        improved = np.minimum(nearest[None, :], distances)
        gain = nearest.sum() - improved.sum(axis=1)
        gain[withheld] = -np.inf
        for frame in reasons:
            gain[index_of[frame]] = -np.inf
        best = int(np.argmax(gain))
        if not np.isfinite(gain[best]):
            break
        take(frames[best], "represents postures no other chosen frame covers")
        nearest = np.minimum(nearest, distances[best])

    chosen.sort()
    return [
        Sample(frame, times[index_of[frame]], scores.get(frame), reasons[frame])
        for frame in chosen
    ]


def coverage(session_id: str, picked: list[int]) -> float:
    """How well a chosen set represents the recording, against time spacing.

    The mean distance from every scored frame to its nearest chosen frame,
    divided into the same figure for the same number of evenly time-spaced
    frames. Above 1 means the selection represents the posture range better than
    sampling the clock would. This is the module's whole claim, so it is measured
    and reported rather than asserted.
    """

    frames, _, cloud = posture_cloud(session_id)
    if not frames or not picked:
        return 0.0
    index_of = {frame: position for position, frame in enumerate(frames)}
    positions = sorted({index_of[frame] for frame in picked if frame in index_of})
    if not positions:
        return 0.0

    distances = _distance_matrix(cloud)
    uniform = np.linspace(0, len(frames) - 1, len(positions)).astype(int).tolist()
    achieved = float(distances[:, positions].min(axis=1).mean())
    baseline = float(distances[:, sorted(set(uniform))].min(axis=1).mean())
    if achieved <= 1e-12:
        return 1.0
    return round(baseline / achieved, 3)


__all__ = [
    "MAX_CANDIDATES",
    "POSTURE_AXES",
    "Sample",
    "coverage",
    "posture_cloud",
    "representative_frames",
]
