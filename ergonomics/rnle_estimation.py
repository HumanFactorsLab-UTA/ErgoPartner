"""Estimate lifting geometry and frequency from tracked hand and body landmarks.

Recorded stature provides a scale for the model's approximate world coordinates.
The resulting distances are anthropometric estimates. Manual inputs take
priority in the pipeline. Load mass and shift duration cannot be observed from
a short recording. The estimator records sources and any duration assumption
so that the reviewer can check them before interpreting the RNLE output.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

# Landmarks used for the lifting geometry.
LEFT_EYE, RIGHT_EYE = 2, 5
LEFT_EAR, RIGHT_EAR = 7, 8
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_INDEX, RIGHT_INDEX = 19, 20

#: NIOSH's valid input ranges. Outside these the equation is not defined, so a
#: derived value that falls outside is reported with a specific diagnosis rather
#: than silently clamped into range.
VALID_RANGES: dict[str, tuple[float, float]] = {
    "horizontal_cm": (25.0, 63.0),
    "vertical_cm": (0.0, 175.0),
    "travel_distance_cm": (25.0, 175.0),
    "asymmetry_deg": (0.0, 135.0),
    "frequency_lifts_per_min": (0.2, 15.0),
}
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

#: Ear height as a fraction of standing stature, from standard adult
#: anthropometry. Used to convert the model's ankle-to-ear span into an implied
#: full stature, since no landmark sits at the top of the head.
EAR_HEIGHT_FRACTION_OF_STATURE = 0.936


@dataclass(frozen=True)
class LiftEstimationPolicy:
    """Thresholds governing lift detection and the scale estimate."""

    #: A hand rise smaller than this is reaching or adjusting, not a lift.
    minimum_lift_travel_cm: float = 25.0
    #: Samples either side of a turning point used to confirm it is one.
    turning_point_window: int = 2
    #: Reject a stature correction outside this band. A factor far from one
    #: means the landmarks, the recorded height, or the units are wrong, and a
    #: silently rescaled result would be worse than no result.
    minimum_scale_factor: float = 0.6
    maximum_scale_factor: float = 1.6
    #: Fall back to this when no participant height is on file. The pose model's
    #: world landmarks are already approximately metric for an average adult, so
    #: a factor of one is the honest no-information default.
    default_scale_factor: float = 1.0
    #: Duration band assumed when the shift length is unknown. RNLE's bands are
    #: short (<=1 h), moderate (1-2 h), and long (2-8 h); the middle band is the
    #: least-committal choice and is always reported as assumed.
    assumed_duration: str = "moderate"
    #: Coupling assumed when the handles were not described. NIOSH's own
    #: convention when grip quality is unrecorded.
    assumed_coupling: str = "fair"

    def __post_init__(self) -> None:
        if self.minimum_lift_travel_cm <= 0:
            raise ValueError("minimum_lift_travel_cm must be positive")
        if not 0 < self.minimum_scale_factor < self.maximum_scale_factor:
            raise ValueError("require 0 < minimum_scale_factor < maximum_scale_factor")
        if self.assumed_duration not in {"short", "moderate", "long"}:
            raise ValueError("assumed_duration must be short, moderate, or long")


@dataclass(frozen=True)
class LiftSample:
    """One assessed frame's landmarks, in canonical model-world metres."""

    timestamp_sec: float
    frame_number: int
    landmarks: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectedLift:
    """One hand-height excursion identified as a lift."""

    origin_frame: int
    destination_frame: int
    origin_sec: float
    destination_sec: float
    horizontal_cm: float
    vertical_cm: float
    travel_distance_cm: float
    asymmetry_deg: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin_frame": self.origin_frame,
            "destination_frame": self.destination_frame,
            "origin_sec": round(self.origin_sec, 3),
            "destination_sec": round(self.destination_sec, 3),
            "horizontal_cm": round(self.horizontal_cm, 1),
            "vertical_cm": round(self.vertical_cm, 1),
            "travel_distance_cm": round(self.travel_distance_cm, 1),
            "asymmetry_deg": round(self.asymmetry_deg, 1),
        }


@dataclass
class RNLEEstimate:
    """Estimated RNLE task inputs, with what supported each one."""

    horizontal_cm: float | None = None
    vertical_cm: float | None = None
    travel_distance_cm: float | None = None
    asymmetry_deg: float | None = None
    frequency_lifts_per_min: float | None = None
    duration: str | None = None
    coupling: str | None = None

    lifts: list[DetectedLift] = field(default_factory=list)
    scale_factor: float | None = None
    scale_basis: str | None = None
    #: Field name -> "derived" or "assumed".
    sources: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def lift_count(self) -> int:
        return len(self.lifts)

    def as_input_fields(self) -> dict[str, Any]:
        """The subset that can be handed straight to :class:`NIOSHInput`."""

        return {
            "horizontal_cm": self.horizontal_cm,
            "vertical_cm": self.vertical_cm,
            "travel_distance_cm": self.travel_distance_cm,
            "asymmetry_deg": self.asymmetry_deg,
            "frequency_lifts_per_min": self.frequency_lifts_per_min,
            "duration": self.duration,
            "coupling": self.coupling,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.as_input_fields())
        payload.update(
            {
                "lift_count": self.lift_count,
                "lifts": [item.to_dict() for item in self.lifts],
                "scale_factor": self.scale_factor,
                "scale_basis": self.scale_basis,
                "sources": dict(self.sources),
                "notes": list(self.notes),
                "distance_measurement_source": "anthropometric_estimate",
            }
        )
        return payload


def _point(sample: LiftSample, index: int) -> np.ndarray | None:
    value = sample.landmarks.get(index)
    if value is None:
        return None
    array = np.asarray(value, dtype=float)[:3]
    return array if np.all(np.isfinite(array)) else None


def _midpoint(sample: LiftSample, left: int, right: int) -> np.ndarray | None:
    first, second = _point(sample, left), _point(sample, right)
    if first is None or second is None:
        return None
    return (first + second) / 2.0


def estimate_scale_factor(
    samples: Sequence[LiftSample],
    participant_height_cm: float | None,
    policy: LiftEstimationPolicy | None = None,
) -> tuple[float, str]:
    """Metres-to-real-metres correction from the participant's recorded stature.

    Returns the factor and a human-readable basis. Falls back to 1.0 whenever the
    recorded height is missing or the implied correction is implausible, because
    a wrong scale silently corrupts every distance downstream.
    """

    policy = policy or LiftEstimationPolicy()
    if not participant_height_cm or participant_height_cm <= 0:
        return policy.default_scale_factor, (
            "No participant height on file; distances use the pose model's own metric scale."
        )

    statures: list[float] = []
    for sample in samples:
        head = _midpoint(sample, LEFT_EAR, RIGHT_EAR)
        if head is None:
            head = _midpoint(sample, LEFT_EYE, RIGHT_EYE)
        floor = _floor_height(sample)
        if head is None or floor is None:
            continue
        span_cm = (float(head[1]) - floor) * 100.0
        if span_cm > 50.0:
            statures.append(span_cm / EAR_HEIGHT_FRACTION_OF_STATURE)

    if not statures:
        return policy.default_scale_factor, (
            "Stature could not be measured from the landmarks; the model's metric scale is used."
        )

    # The tallest frames are the upright ones; a subject bent over a load reads
    # short. The 90th percentile takes an upright frame without taking an outlier.
    model_stature_cm = float(np.percentile(statures, 90))
    factor = participant_height_cm / model_stature_cm
    if not policy.minimum_scale_factor <= factor <= policy.maximum_scale_factor:
        return policy.default_scale_factor, (
            f"Implied stature correction of {factor:.2f} is outside the plausible range; "
            f"the model's own metric scale is used instead. The pose model read "
            f"{model_stature_cm:.0f} cm against a recorded {participant_height_cm:.0f} cm."
        )
    return float(factor), (
        f"Scaled by {factor:.3f}: recorded stature {participant_height_cm:.0f} cm against "
        f"{model_stature_cm:.0f} cm implied by the pose model."
    )


def _floor_height(sample: LiftSample) -> float | None:
    """Height of the supporting surface, in model metres.

    The lowest foot landmark is the best available floor: the ankle sits several
    centimetres above it, which would bias every hand height.
    """

    candidates = [
        _point(sample, index)
        for index in (
            LEFT_HEEL,
            RIGHT_HEEL,
            LEFT_FOOT_INDEX,
            RIGHT_FOOT_INDEX,
            LEFT_ANKLE,
            RIGHT_ANKLE,
        )
    ]
    heights = [float(point[1]) for point in candidates if point is not None]
    return min(heights) if heights else None


def _hand_midpoint(sample: LiftSample) -> np.ndarray | None:
    """Midpoint of the hand grasps.

    RNLE measures to the grasp itself, so the knuckles are the right landmark and
    the wrist is a fallback. The difference is only a few centimetres, but H is
    divided into 25 to form the horizontal multiplier, so it is not free.
    """

    knuckles = _midpoint(sample, LEFT_INDEX, RIGHT_INDEX)
    return knuckles if knuckles is not None else _midpoint(sample, LEFT_WRIST, RIGHT_WRIST)


def _forward_axis(sample: LiftSample) -> np.ndarray | None:
    """The subject's anterior direction, projected into the transverse plane.

    RNLE measures asymmetry from the sagittal plane of the *neutral* body
    position, which is defined by where the torso points rather than by where
    the camera stands.

    Two details matter, and getting either wrong produces nonsense:

    **The feet define the reference, not the torso.** RNLE's asymmetry angle is
    measured from the sagittal plane of the neutral body position -- where the
    subject would face standing upright -- and explicitly not from a twisted or
    flexed trunk. Using ``shoulders - hips`` as the superior axis fails exactly
    where it is needed: at the origin of a floor lift the trunk is near
    horizontal, so that vector points forward, ``cross(superior, lateral)``
    points at the floor, and its transverse projection is pure noise. Measured on
    real lifting footage that produced asymmetry angles scattered across the full
    1-135 degree range. The ankle line and world vertical are stable through the
    whole lift.

    **Handedness follows ``AngleCalculator``'s pelvis frame**, anterior =
    ``cross(superior, lateral)``. Deriving the sign independently inverted it in
    an earlier revision, putting every forward reach at 180 degrees.
    """

    lateral = None
    left_ankle, right_ankle = _point(sample, LEFT_ANKLE), _point(sample, RIGHT_ANKLE)
    if left_ankle is not None and right_ankle is not None:
        lateral = right_ankle - left_ankle
    if lateral is None or np.linalg.norm(lateral) < 1e-6:
        left_hip, right_hip = _point(sample, LEFT_HIP), _point(sample, RIGHT_HIP)
        if left_hip is None or right_hip is None:
            return None
        lateral = right_hip - left_hip
    if np.linalg.norm(lateral) < 1e-6:
        return None

    anterior = np.cross(np.array([0.0, 1.0, 0.0]), lateral)
    flat = np.array([anterior[0], anterior[2]])
    norm = np.linalg.norm(flat)
    if norm < 1e-6:
        return None
    return flat / norm


class RNLEEstimator:
    """Derive RNLE task inputs from a lifting sequence."""

    def __init__(self, policy: LiftEstimationPolicy | None = None) -> None:
        self.policy = policy or LiftEstimationPolicy()

    def estimate(
        self,
        samples: Sequence[LiftSample],
        *,
        participant_height_cm: float | None = None,
        duration: str | None = None,
        coupling: str | None = None,
    ) -> RNLEEstimate:
        """Estimate every RNLE input this footage can support."""

        ordered = sorted(samples, key=lambda item: (item.timestamp_sec, item.frame_number))
        result = RNLEEstimate()

        scale, basis = estimate_scale_factor(ordered, participant_height_cm, self.policy)
        result.scale_factor = scale
        result.scale_basis = basis

        # Duration and coupling are task facts, not geometry.
        if duration is not None:
            result.duration, result.sources["duration"] = duration, "observed"
        else:
            result.duration = self.policy.assumed_duration
            result.sources["duration"] = "assumed"
            result.notes.append(
                f"Work duration assumed to be '{self.policy.assumed_duration}'. How long the "
                "job runs is a property of the shift and cannot be read from a recording; "
                "correct it if the task runs materially longer or shorter."
            )
        if coupling is not None:
            result.coupling, result.sources["coupling"] = coupling, "observed"
        else:
            result.coupling = self.policy.assumed_coupling
            result.sources["coupling"] = "assumed"
            result.notes.append(
                f"Hand coupling assumed to be '{self.policy.assumed_coupling}', the NIOSH "
                "convention when grip quality is unrecorded."
            )

        lifts = self._detect_lifts(ordered, scale)
        result.lifts = lifts
        if not lifts:
            result.notes.append(
                "No lift was detected: the hands never rose by the "
                f"{self.policy.minimum_lift_travel_cm:.0f} cm that separates a lift from "
                "reaching. RNLE describes lifting tasks and is not reported for this trial."
            )
            return result

        span_sec = ordered[-1].timestamp_sec - ordered[0].timestamp_sec
        # Median across lifts: representative of the task rather than of its one
        # worst repetition, and unaffected by a single mis-detected excursion.
        result.horizontal_cm = float(np.median([item.horizontal_cm for item in lifts]))
        result.vertical_cm = float(np.median([item.vertical_cm for item in lifts]))
        result.travel_distance_cm = float(np.median([item.travel_distance_cm for item in lifts]))
        result.asymmetry_deg = float(np.median([item.asymmetry_deg for item in lifts]))
        for name in ("horizontal_cm", "vertical_cm", "travel_distance_cm", "asymmetry_deg"):
            result.sources[name] = "derived"

        frequency = self._frequency(lifts, span_sec)
        if frequency is not None:
            result.frequency_lifts_per_min = frequency
            result.sources["frequency_lifts_per_min"] = "derived"
            result.notes.append(
                f"{len(lifts)} lift(s) detected over {span_sec:.1f} s, giving "
                f"{frequency:.1f} lifts per minute. Frequency assumes this sample is "
                "representative of the sustained rate."
            )

        # Last, so that every derived value is present to be checked.
        self._note_out_of_range(result)
        return result

    @staticmethod
    def _note_out_of_range(result: RNLEEstimate) -> None:
        """Say plainly when a derived value falls outside the equation's range.

        Two very different things produce an out-of-range value, and the reader
        has to be able to tell them apart: the task genuinely sits outside what
        RNLE covers, or the estimate is wrong. The note names both, because this
        module cannot distinguish them.

        Horizontal distance is the one to watch. It is a Euclidean norm that
        includes the depth axis, and depth is the weakest axis of a monocular
        pose estimate, so noise there can only push H up, never down.
        """

        for name, (low, high) in VALID_RANGES.items():
            value = getattr(result, name, None)
            if value is None or low <= value <= high:
                continue
            result.sources[name] = "derived_out_of_range"
            detail = (
                "Depth is the least reliable axis of a monocular pose estimate and it can "
                "only inflate this distance, so treat a high value with suspicion and "
                "measure it directly."
                if name == "horizontal_cm"
                else "Confirm it against the task."
            )
            result.notes.append(
                f"Estimated {name.replace('_', ' ')} of {value:.1f} is outside the RNLE valid "
                f"range of {low:g} to {high:g}. Either the task falls outside what the equation "
                f"covers, or the estimate is off. {detail} RNLE will report the task as "
                "unsupported until this value is entered or corrected."
            )

    @staticmethod
    def _frequency(lifts: Sequence[DetectedLift], span_sec: float) -> float | None:
        """Lifts per minute, measured from the interval between them.

        Dividing the lift count by the clip length undercounts, because a
        turning point cannot be confirmed at the very first or last frame and
        the recording rarely starts and ends mid-cycle. The interval between
        consecutive lift origins is unaffected by either edge.
        """

        if len(lifts) >= 2:
            origins = [item.origin_sec for item in lifts]
            period = (origins[-1] - origins[0]) / (len(origins) - 1)
            if period > 0:
                return float(60.0 / period)
        if len(lifts) == 1 and span_sec > 0:
            # One lift sets a lower bound on the rate, nothing more.
            return float(1.0 / (span_sec / 60.0))
        return None

    # ------------------------------------------------------------------

    def _detect_lifts(self, samples: Sequence[LiftSample], scale: float) -> list[DetectedLift]:
        """Find hand-height excursions large enough to be lifts.

        A lift runs from a local minimum of the hand height (the origin, where
        the load is grasped) to the following local maximum (the destination).
        Only excursions taller than the policy's minimum count, which excludes
        reaching, adjusting, and pose noise.
        """

        usable = []
        for sample in samples:
            hands = _hand_midpoint(sample)
            floor = _floor_height(sample)
            if hands is None or floor is None:
                continue
            usable.append((sample, float(hands[1]) - floor))
        if len(usable) < 5:
            return []

        heights = np.array([height for _, height in usable], dtype=float)
        minimum_travel_m = self.policy.minimum_lift_travel_cm / 100.0 / max(scale, 1e-6)

        turning = self._turning_points(heights)
        lifts: list[DetectedLift] = []
        for position in range(len(turning) - 1):
            start_index, start_kind = turning[position]
            end_index, end_kind = turning[position + 1]
            if start_kind != "min" or end_kind != "max":
                continue
            travel = heights[end_index] - heights[start_index]
            if travel < minimum_travel_m:
                continue
            origin_sample = usable[start_index][0]
            lift = self._measure(origin_sample, usable[end_index][0], travel, scale)
            if lift is not None:
                lifts.append(lift)
        return lifts

    def _turning_points(self, heights: np.ndarray) -> list[tuple[int, str]]:
        """Alternating local minima and maxima of the hand-height signal."""

        window = max(1, self.policy.turning_point_window)
        points: list[tuple[int, str]] = []
        for index in range(window, len(heights) - window):
            neighbourhood = heights[index - window : index + window + 1]
            if heights[index] == neighbourhood.min() and heights[index] < neighbourhood.max():
                kind = "min"
            elif heights[index] == neighbourhood.max() and heights[index] > neighbourhood.min():
                kind = "max"
            else:
                continue
            # Keep the sequence alternating, holding the most extreme of any run.
            if points and points[-1][1] == kind:
                previous = points[-1][0]
                better = (
                    heights[index] < heights[previous]
                    if kind == "min"
                    else heights[index] > heights[previous]
                )
                if better:
                    points[-1] = (index, kind)
                continue
            points.append((index, kind))
        return points

    def _measure(
        self,
        origin: LiftSample,
        destination: LiftSample,
        travel_m: float,
        scale: float,
    ) -> DetectedLift | None:
        """Measure H, V, D and A for one lift, at its origin as RNLE requires."""

        hands = _hand_midpoint(origin)
        ankles = _midpoint(origin, LEFT_ANKLE, RIGHT_ANKLE)
        floor = _floor_height(origin)
        if hands is None or ankles is None or floor is None:
            return None

        # H: floor-plane distance from the ankle midpoint to the hand midpoint.
        offset = np.array([hands[0] - ankles[0], hands[2] - ankles[2]])
        horizontal_cm = float(np.linalg.norm(offset)) * 100.0 * scale
        # V: hand height above the supporting surface at the origin.
        vertical_cm = (float(hands[1]) - floor) * 100.0 * scale
        travel_cm = travel_m * 100.0 * scale

        # A: angle in the transverse plane between the subject's forward
        # direction and the direction from the ankles to the hands.
        asymmetry_deg = 0.0
        forward = _forward_axis(origin)
        reach = float(np.linalg.norm(offset))
        if forward is not None and reach > 1e-6:
            cosine = float(np.dot(offset / reach, forward))
            asymmetry_deg = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

        return DetectedLift(
            origin_frame=origin.frame_number,
            destination_frame=destination.frame_number,
            origin_sec=origin.timestamp_sec,
            destination_sec=destination.timestamp_sec,
            horizontal_cm=horizontal_cm,
            vertical_cm=max(vertical_cm, 0.0),
            travel_distance_cm=travel_cm,
            asymmetry_deg=min(asymmetry_deg, 135.0),
        )


def estimate_rnle_inputs(
    samples: Sequence[LiftSample],
    *,
    participant_height_cm: float | None = None,
    duration: str | None = None,
    coupling: str | None = None,
    policy: LiftEstimationPolicy | None = None,
) -> RNLEEstimate:
    """Convenience wrapper for one-off estimation."""

    return RNLEEstimator(policy).estimate(
        samples,
        participant_height_cm=participant_height_cm,
        duration=duration,
        coupling=coupling,
    )


def policy_from_config(config: dict[str, Any] | None) -> LiftEstimationPolicy:
    """Build a policy from the ``rnle_estimation`` config block."""

    defaults = LiftEstimationPolicy()
    if not config:
        return defaults
    if not isinstance(config, dict):
        raise ValueError("rnle_estimation must be a mapping")
    known = set(vars(defaults))
    unknown = sorted(set(config) - known)
    if unknown:
        raise ValueError(f"unknown rnle estimation settings: {unknown}; available: {sorted(known)}")
    values: dict[str, Any] = {}
    for name, value in config.items():
        values[name] = value if isinstance(value, str) else float(value)
    return replace(defaults, **values)
