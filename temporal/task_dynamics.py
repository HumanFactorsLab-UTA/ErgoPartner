"""Derive the time-dependent RULA/REBA task conditions from a pose sequence.

``core/posture_conditions.py`` answers the conditions that a single frame can
settle. The remainder are properties of the *task over time* and cannot be seen
in any one frame:

* whether a posture is held for more than a minute (RULA and REBA muscle use);
* whether the action repeats more than four times a minute (same);
* whether there are rapid, large posture changes or an unstable base
  (REBA activity score);
* whether the subject is seated (REBA leg score);
* how the legs bear weight, and whether they are supported and balanced.

Each of these is a measurement over the sequence, not an operator observation,
so this module derives them from the frames the pipeline already decoded.

The same three rules as the per-frame deriver apply: unknown never becomes
false, every threshold is declared in :class:`TaskDynamicsPolicy` with its
reasoning, and each answer carries the evidence it was based on.

Pure module: numpy only, no video, model, or dashboard imports.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

# Landmark ids used for the leg and base-of-support geometry.
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

#: Conditions this module can supply.
DERIVABLE_CONDITIONS = frozenset(
    {
        "static_posture_over_one_minute",
        "repeated_more_than_four_per_minute",
        "rapid_large_posture_change_or_unstable_base",
        "seated",
        "legs_weight_bearing",
        "legs_supported_and_balanced",
        "rula_load_type",
    }
)

#: Angles scanned for stillness and for the dominant cyclic signal. Trunk and
#: upper arm carry most task motion; neck and knee catch the cases where the
#: work is done by looking down or squatting.
MOTION_SIGNALS = (
    "trunk_flexion",
    "neck_flexion",
    "left_upper_arm_flexion",
    "right_upper_arm_flexion",
    "left_knee_flexion",
    "right_knee_flexion",
)


@dataclass(frozen=True)
class TaskDynamicsPolicy:
    """Thresholds that turn motion over time into method conditions."""

    # --- Static posture ---
    #: RULA and REBA both say "held longer than one minute".
    static_window_sec: float = 60.0
    #: Peak-to-peak angular movement below which a window counts as held. Pose
    #: noise on a stationary subject runs a few degrees; 10 deg clears it while
    #: still calling a genuinely still posture static.
    static_range_deg: float = 10.0

    # --- Repetition ---
    #: Both methods use "more than 4 times per minute".
    repetition_per_minute: float = 4.0
    #: A cycle must swing at least this far to count, so that noise around the
    #: mean is not counted as repeated work.
    minimum_cycle_amplitude_deg: float = 15.0
    #: Fraction of the signal's amplitude used as crossing hysteresis.
    crossing_hysteresis_fraction: float = 0.25

    # --- Activity / stability ---
    #: Angular speed marking a rapid postural change.
    #:
    #: Measured on the *smoothed* signal. Raw per-frame angles jitter by 10-30
    #: degrees between neighbouring frames, and at the shipped 10 fps a 10 degree
    #: jitter is already 100 degrees per second -- which was the previous
    #: threshold, so this condition fired on 100% of frames of a bench task and
    #: added a REBA activity point to every one of them. Smoothing removes the
    #: jitter and 150 deg/s then describes a genuinely fast swing.
    rapid_change_deg_per_sec: float = 150.0
    #: Frames of median filtering applied before the speed is measured.
    speed_smoothing_frames: int = 5
    #: Rapid motion must occupy at least this share of the assessed frames
    #: before the task as a whole is called rapid. A single lunge in an
    #: otherwise steady task is not "rapid large range changes in posture".
    rapid_change_minimum_fraction: float = 0.15
    #: Stance-width variation, as a fraction of leg length, above which the base
    #: of support is treated as unstable.
    unstable_base_stance_variation: float = 0.25

    # --- Seated ---
    #: (hip_y - knee_y) / (hip_y - ankle_y). Standing sits near 0.5 because the
    #: knee is midway down the leg; sitting collapses it toward 0 as the thigh
    #: comes up to horizontal. Normalising by leg length makes this independent
    #: of the subject's height.
    seated_thigh_drop_ratio: float = 0.30
    #: Share of frames that must satisfy the ratio before the trial is seated.
    seated_minimum_fraction: float = 0.60

    # --- Leg weight bearing ---
    #: Ankle height difference, as a fraction of leg length, marking one foot
    #: clearly off the supporting surface.
    single_support_ankle_offset: float = 0.10
    #: Share of frames on one foot that makes the stance unilateral rather than
    #: an incidental weight shift.
    unilateral_minimum_fraction: float = 0.35
    #: Alternating support transitions per minute above which the subject is
    #: walking rather than standing.
    walking_transitions_per_minute: float = 8.0

    def __post_init__(self) -> None:
        if self.static_window_sec <= 0 or self.repetition_per_minute < 0:
            raise ValueError("static window must be positive and repetition rate non-negative")
        if not 0 < self.crossing_hysteresis_fraction < 1:
            raise ValueError("crossing_hysteresis_fraction must be in (0, 1)")
        for name in ("seated_minimum_fraction", "unilateral_minimum_fraction"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")


@dataclass(frozen=True)
class PostureSample:
    """One assessed frame's measurements, in acquisition order."""

    timestamp_sec: float
    frame_number: int
    #: Measured angles by name; a missing or None entry is simply unavailable.
    angles: dict[str, float | None] = field(default_factory=dict)
    #: Canonical model-world landmarks in metres, hip-centred, Y up.
    landmarks: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class DynamicsEvidence:
    """Why one time-dependent condition took the value it did."""

    condition: str
    value: Any
    basis: str
    measured: float | None = None
    threshold: float | None = None
    units: str | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "condition": self.condition,
            "value": self.value,
            "basis": self.basis,
            "measured": self.measured,
            "threshold": self.threshold,
        }
        if self.units:
            payload["units"] = self.units
        if self.note:
            payload["note"] = self.note
        return payload


@dataclass
class TaskDynamics:
    """Time-dependent conditions derived for one assessed interval."""

    values: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, DynamicsEvidence] = field(default_factory=dict)
    #: Detected cycle rate, retained for the dashboard's movement view.
    cycles_per_minute: float | None = None
    dominant_signal: str | None = None
    observed_duration_sec: float = 0.0

    @property
    def available(self) -> dict[str, Any]:
        return {name: value for name, value in self.values.items() if value is not None}

    @property
    def unavailable(self) -> list[str]:
        return sorted(name for name, value in self.values.items() if value is None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": dict(self.values),
            "unavailable": self.unavailable,
            "cycles_per_minute": self.cycles_per_minute,
            "dominant_signal": self.dominant_signal,
            "observed_duration_sec": self.observed_duration_sec,
            "evidence": {name: item.to_dict() for name, item in self.evidence.items()},
        }


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _series(samples: Sequence[PostureSample], name: str) -> tuple[np.ndarray, np.ndarray]:
    """Timestamps and values for one angle, skipping unmeasured frames."""

    times: list[float] = []
    values: list[float] = []
    for sample in samples:
        value = _finite(sample.angles.get(name))
        if value is None:
            continue
        times.append(float(sample.timestamp_sec))
        values.append(value)
    return np.asarray(times, dtype=float), np.asarray(values, dtype=float)


def _median_filter(values: np.ndarray, window: int) -> np.ndarray:
    """Median-filter a series, so a spike does not read as motion."""

    if window <= 1 or values.size < window:
        return values
    half = window // 2
    return np.array(
        [
            float(np.median(values[max(0, index - half) : index + half + 1]))
            for index in range(values.size)
        ]
    )


def _longest_still_interval(times: np.ndarray, values: np.ndarray, range_deg: float) -> float:
    """Longest span over which the signal's peak-to-peak stays within a band.

    Two pointers with monotonic maximum/minimum deques, so the whole scan is
    linear in the number of samples. The naive version recomputes the window
    extremes on every step and becomes unusable on a few thousand frames.
    """

    from collections import deque

    maxima: deque[int] = deque()
    minima: deque[int] = deque()
    start = 0
    longest = 0.0

    for end in range(times.size):
        while maxima and values[maxima[-1]] <= values[end]:
            maxima.pop()
        maxima.append(end)
        while minima and values[minima[-1]] >= values[end]:
            minima.pop()
        minima.append(end)

        while values[maxima[0]] - values[minima[0]] > range_deg:
            start += 1
            if maxima[0] < start:
                maxima.popleft()
            if minima[0] < start:
                minima.popleft()

        if end >= start:
            longest = max(longest, float(times[end] - times[start]))
    return longest


def _leg_length(sample: PostureSample) -> float | None:
    """Hip-to-ankle distance, used to normalise leg geometry by body size."""

    lengths = []
    for hip_id, ankle_id in ((LEFT_HIP, LEFT_ANKLE), (RIGHT_HIP, RIGHT_ANKLE)):
        hip = sample.landmarks.get(hip_id)
        ankle = sample.landmarks.get(ankle_id)
        if hip is None or ankle is None:
            continue
        span = abs(float(np.asarray(hip)[1]) - float(np.asarray(ankle)[1]))
        if span > 1e-3:
            lengths.append(span)
    return float(np.mean(lengths)) if lengths else None


class TaskDynamicsAnalyzer:
    """Derive time-dependent task conditions from an ordered pose sequence."""

    def __init__(self, policy: TaskDynamicsPolicy | None = None) -> None:
        self.policy = policy or TaskDynamicsPolicy()

    # ------------------------------------------------------------------

    def analyze(self, samples: Sequence[PostureSample]) -> TaskDynamics:
        """Derive every supportable condition for the assessed interval."""

        ordered = sorted(samples, key=lambda item: (item.timestamp_sec, item.frame_number))
        result = TaskDynamics()
        if len(ordered) < 2:
            for name in DERIVABLE_CONDITIONS:
                result.values[name] = None
                result.evidence[name] = DynamicsEvidence(
                    condition=name,
                    value=None,
                    basis="insufficient_samples",
                    note="At least two assessed frames are required.",
                )
            return result

        duration = float(ordered[-1].timestamp_sec - ordered[0].timestamp_sec)
        result.observed_duration_sec = duration

        static = self._static_posture(ordered, duration, result)
        repeated = self._repetition(ordered, duration, result)
        self._activity(ordered, result)
        seated = self._seated(ordered, result)
        self._legs(ordered, seated, result)
        self._load_type(static, repeated, result)
        return result

    # ------------------------------------------------------------------
    # Static posture
    # ------------------------------------------------------------------

    def _static_posture(
        self, samples: Sequence[PostureSample], duration: float, result: TaskDynamics
    ) -> bool | None:
        """True when some angle stays within its stillness band for a full window."""

        policy = self.policy
        window = policy.static_window_sec
        longest_still = 0.0
        measured_any = False

        for name in MOTION_SIGNALS:
            times, values = _series(samples, name)
            if times.size < 2:
                continue
            measured_any = True
            longest_still = max(
                longest_still, _longest_still_interval(times, values, policy.static_range_deg)
            )

        if not measured_any:
            result.values["static_posture_over_one_minute"] = None
            result.evidence["static_posture_over_one_minute"] = DynamicsEvidence(
                condition="static_posture_over_one_minute",
                value=None,
                basis="no_measured_angles",
                note="No angle was measurable across the interval.",
            )
            return None

        if longest_still >= window:
            value: bool | None = True
            note = None
        elif duration < window and longest_still >= duration - 1e-6:
            # The subject held still for the whole clip, but the clip is shorter
            # than a minute, so whether it continued past the recording is not
            # something this footage can answer.
            value = None
            note = (
                f"The posture was held for the entire {duration:.1f} s interval, which is "
                f"shorter than the {window:.0f} s the methods ask about. Extend the recording "
                "or confirm the hold time directly."
            )
        else:
            value = False
            note = None

        result.values["static_posture_over_one_minute"] = value
        result.evidence["static_posture_over_one_minute"] = DynamicsEvidence(
            condition="static_posture_over_one_minute",
            value=value,
            basis="longest_interval_within_stillness_band",
            measured=longest_still,
            threshold=window,
            units="seconds",
            note=note,
        )
        return value

    # ------------------------------------------------------------------
    # Repetition
    # ------------------------------------------------------------------

    def _dominant_signal(
        self, samples: Sequence[PostureSample]
    ) -> tuple[str | None, np.ndarray, np.ndarray]:
        """The angle carrying the largest swing, which is the task's motion."""

        best_name: str | None = None
        best_amplitude = 0.0
        best: tuple[np.ndarray, np.ndarray] = (np.array([]), np.array([]))
        for name in MOTION_SIGNALS:
            times, values = _series(samples, name)
            if times.size < 4:
                continue
            # Smoothed before the comparison and before counting cycles.
            # Unsmoothed, landmark jitter is itself an oscillation: on a bench
            # task the noisiest signal won the amplitude contest and produced 48
            # "cycles" per minute, which is head jitter rather than task
            # repetition.
            smoothed = _median_filter(values, self.policy.speed_smoothing_frames)
            # Robust amplitude: the central 90% of the signal, so one bad frame
            # does not win the comparison.
            amplitude = float(np.percentile(smoothed, 95) - np.percentile(smoothed, 5))
            if amplitude > best_amplitude:
                best_amplitude, best_name, best = amplitude, name, (times, smoothed)
        return best_name, best[0], best[1]

    def _repetition(
        self, samples: Sequence[PostureSample], duration: float, result: TaskDynamics
    ) -> bool | None:
        policy = self.policy
        name, times, values = self._dominant_signal(samples)

        if name is None or duration <= 0:
            result.values["repeated_more_than_four_per_minute"] = None
            result.evidence["repeated_more_than_four_per_minute"] = DynamicsEvidence(
                condition="repeated_more_than_four_per_minute",
                value=None,
                basis="no_usable_motion_signal",
                note="No angle supplied enough samples to detect cycles.",
            )
            return None

        amplitude = float(np.percentile(values, 95) - np.percentile(values, 5))
        result.dominant_signal = name

        if amplitude < policy.minimum_cycle_amplitude_deg:
            # The subject barely moved: no repetition, and that is a real answer.
            result.cycles_per_minute = 0.0
            result.values["repeated_more_than_four_per_minute"] = False
            result.evidence["repeated_more_than_four_per_minute"] = DynamicsEvidence(
                condition="repeated_more_than_four_per_minute",
                value=False,
                basis=f"{name} amplitude below cycle threshold",
                measured=amplitude,
                threshold=policy.minimum_cycle_amplitude_deg,
                units="degrees",
            )
            return False

        rate, cycles = self._cycle_rate(times, values, amplitude)
        result.cycles_per_minute = float(rate)
        value = bool(rate > policy.repetition_per_minute)
        result.values["repeated_more_than_four_per_minute"] = value
        result.evidence["repeated_more_than_four_per_minute"] = DynamicsEvidence(
            condition="repeated_more_than_four_per_minute",
            value=value,
            basis=f"upward mean crossings of {name}",
            measured=float(rate),
            threshold=policy.repetition_per_minute,
            units="cycles per minute",
            note=(
                f"{cycles} cycle(s) detected over {duration:.1f} s. Cycles are a motion "
                "signal, not confirmed task repetitions."
            ),
        )
        return value

    def _cycle_rate(
        self, times: np.ndarray, values: np.ndarray, amplitude: float
    ) -> tuple[float, int]:
        """Cycle rate per minute, measured from the signal's own period.

        Counting crossings and dividing by the clip length miscounts whenever
        the recording does not start and end at the same phase: two full cycles
        of a signal that starts at its mean yields one detected crossing, an
        undercount of half. Measuring the interval *between* crossings avoids
        that -- consecutive crossings of the mean are half a period apart
        regardless of where the recording happened to begin.

        The hysteresis band keeps noise around the mean from registering as
        extra crossings.

        Returns:
            (cycles per minute, number of whole cycles observed).
        """

        centre = float(np.mean(values))
        band = amplitude * self.policy.crossing_hysteresis_fraction / 2.0

        crossings: list[float] = []
        state: str | None = None
        for time, value in zip(times, values, strict=True):
            if value > centre + band:
                current = "above"
            elif value < centre - band:
                current = "below"
            else:
                continue
            if state is not None and current != state:
                crossings.append(float(time))
            state = current

        if len(crossings) < 2:
            return 0.0, 0
        # Consecutive crossings are half a period apart.
        half_period = (crossings[-1] - crossings[0]) / (len(crossings) - 1)
        if half_period <= 0:
            return 0.0, 0
        rate = 60.0 / (2.0 * half_period)
        observed = (len(crossings) - 1) // 2
        return float(rate), int(observed)

    # ------------------------------------------------------------------
    # Activity: rapid change and base stability
    # ------------------------------------------------------------------

    def _activity(self, samples: Sequence[PostureSample], result: TaskDynamics) -> None:
        policy = self.policy
        fastest = 0.0
        rapid_fraction = 0.0
        measured = False

        for name in ("trunk_flexion", "left_upper_arm_flexion", "right_upper_arm_flexion"):
            times, values = _series(samples, name)
            if times.size < 3:
                continue
            gaps = np.diff(times)
            usable = gaps > 1e-6
            if not np.any(usable):
                continue
            measured = True
            # Smoothed first: otherwise this measures landmark jitter, not motion.
            smoothed = _median_filter(values, policy.speed_smoothing_frames)
            speed = np.abs(np.diff(smoothed)[usable] / gaps[usable])
            fastest = max(fastest, float(np.max(speed)))
            rapid_fraction = max(
                rapid_fraction,
                float(np.mean(speed > policy.rapid_change_deg_per_sec)),
            )

        stance_variation = self._stance_variation(samples)

        if not measured and stance_variation is None:
            result.values["rapid_large_posture_change_or_unstable_base"] = None
            result.evidence["rapid_large_posture_change_or_unstable_base"] = DynamicsEvidence(
                condition="rapid_large_posture_change_or_unstable_base",
                value=None,
                basis="no_measured_motion_or_stance",
            )
            return

        rapid = measured and rapid_fraction >= policy.rapid_change_minimum_fraction
        unstable = (
            stance_variation is not None
            and stance_variation > policy.unstable_base_stance_variation
        )
        value = bool(rapid or unstable)
        basis = "rapid posture change" if rapid else "stance variation"
        if rapid and unstable:
            basis = "rapid posture change and stance variation"
        elif not value:
            basis = "neither rapid change nor stance variation"

        result.values["rapid_large_posture_change_or_unstable_base"] = value
        result.evidence["rapid_large_posture_change_or_unstable_base"] = DynamicsEvidence(
            condition="rapid_large_posture_change_or_unstable_base",
            value=value,
            basis=basis,
            measured=fastest if measured else None,
            threshold=policy.rapid_change_deg_per_sec,
            units="degrees per second (peak)",
            note=(
                None
                if stance_variation is None
                else f"Stance width varied by {stance_variation:.2f} of leg length."
            ),
        )

    def _stance_variation(self, samples: Sequence[PostureSample]) -> float | None:
        """Spread of the foot separation, normalised by leg length.

        Measured along the subject's lateral axis only. Taking the full
        horizontal distance pulls in the depth axis, which is the least reliable
        component of a monocular pose estimate, and the noise there dominates:
        on a subject standing at a bench the full-distance version reported the
        stance varying by 1.06 of a leg length, which is not a thing a standing
        person does. That single bad number set both the unstable-base condition
        and the legs-not-supported condition, adding two REBA penalties to every
        frame of the trial.
        """

        widths: list[float] = []
        for sample in samples:
            left = sample.landmarks.get(LEFT_ANKLE)
            right = sample.landmarks.get(RIGHT_ANKLE)
            lateral = self._lateral_axis(sample)
            length = _leg_length(sample)
            if left is None or right is None or lateral is None or not length:
                continue
            offset = np.asarray(left, dtype=float)[:3] - np.asarray(right, dtype=float)[:3]
            separation = abs(float(np.dot(np.array([offset[0], offset[2]]), lateral)))
            widths.append(separation / length)
        if len(widths) < 3:
            return None
        # A median filter first, so one bad frame does not become the spread.
        smoothed = _median_filter(np.asarray(widths, dtype=float), 5)
        return float(np.percentile(smoothed, 95) - np.percentile(smoothed, 5))

    @staticmethod
    def _lateral_axis(sample: PostureSample) -> np.ndarray | None:
        """The subject's left-right direction in the horizontal plane."""

        left = sample.landmarks.get(LEFT_HIP)
        right = sample.landmarks.get(RIGHT_HIP)
        if left is None or right is None:
            return None
        offset = np.asarray(right, dtype=float)[:3] - np.asarray(left, dtype=float)[:3]
        flat = np.array([offset[0], offset[2]])
        norm = float(np.linalg.norm(flat))
        return flat / norm if norm > 1e-6 else None

    # ------------------------------------------------------------------
    # Seated
    # ------------------------------------------------------------------

    def _seated(self, samples: Sequence[PostureSample], result: TaskDynamics) -> bool | None:
        policy = self.policy
        ratios: list[float] = []
        for sample in samples:
            hip = self._midpoint(sample, LEFT_HIP, RIGHT_HIP)
            knee = self._midpoint(sample, LEFT_KNEE, RIGHT_KNEE)
            ankle = self._midpoint(sample, LEFT_ANKLE, RIGHT_ANKLE)
            if hip is None or knee is None or ankle is None:
                continue
            leg_span = hip[1] - ankle[1]
            if abs(leg_span) < 1e-3:
                continue
            ratios.append(float((hip[1] - knee[1]) / leg_span))

        if len(ratios) < 3:
            result.values["seated"] = None
            result.evidence["seated"] = DynamicsEvidence(
                condition="seated",
                value=None,
                basis="hip_knee_ankle_geometry_unavailable",
                note="Legs were not visible often enough to tell sitting from standing.",
            )
            return None

        array = np.asarray(ratios, dtype=float)
        fraction = float(np.mean(array < policy.seated_thigh_drop_ratio))
        value = bool(fraction >= policy.seated_minimum_fraction)
        result.values["seated"] = value
        result.evidence["seated"] = DynamicsEvidence(
            condition="seated",
            value=value,
            basis="thigh drop ratio (hip to knee over hip to ankle)",
            measured=float(np.median(array)),
            threshold=policy.seated_thigh_drop_ratio,
            units="ratio of leg length",
            note=f"{fraction * 100:.0f}% of frames matched a seated thigh position.",
        )
        return value

    @staticmethod
    def _midpoint(sample: PostureSample, left_id: int, right_id: int) -> np.ndarray | None:
        left = sample.landmarks.get(left_id)
        right = sample.landmarks.get(right_id)
        if left is None or right is None:
            return None
        pair = (np.asarray(left, dtype=float)[:3] + np.asarray(right, dtype=float)[:3]) / 2.0
        return pair if np.all(np.isfinite(pair)) else None

    # ------------------------------------------------------------------
    # Leg weight bearing
    # ------------------------------------------------------------------

    def _legs(
        self, samples: Sequence[PostureSample], seated: bool | None, result: TaskDynamics
    ) -> None:
        policy = self.policy
        offsets: list[float] = []
        times: list[float] = []
        for sample in samples:
            left = sample.landmarks.get(LEFT_ANKLE)
            right = sample.landmarks.get(RIGHT_ANKLE)
            length = _leg_length(sample)
            if left is None or right is None or not length:
                continue
            offsets.append((float(np.asarray(left)[1]) - float(np.asarray(right)[1])) / length)
            times.append(float(sample.timestamp_sec))

        if len(offsets) < 3:
            for name in ("legs_weight_bearing", "legs_supported_and_balanced"):
                result.values[name] = None
                result.evidence[name] = DynamicsEvidence(
                    condition=name,
                    value=None,
                    basis="ankle_geometry_unavailable",
                    note="Ankles were not visible often enough to judge weight bearing.",
                )
            return

        array = np.asarray(offsets, dtype=float)
        span = max(times[-1] - times[0], 1e-6)
        single_support = np.abs(array) > policy.single_support_ankle_offset
        single_fraction = float(np.mean(single_support))

        # Alternating support: sign changes of a clearly one-sided offset.
        signs = np.sign(array[single_support])
        transitions = int(np.count_nonzero(np.diff(signs) != 0)) if signs.size > 1 else 0
        transitions_per_minute = transitions / (span / 60.0)

        if seated:
            bearing = "bilateral"
            basis = "seated; legs are not the weight-bearing base"
        elif (
            single_fraction >= policy.unilateral_minimum_fraction
            and transitions_per_minute >= policy.walking_transitions_per_minute
        ):
            bearing = "walking"
            basis = f"{transitions} alternating support transitions"
        elif single_fraction >= policy.unilateral_minimum_fraction:
            bearing = "unilateral"
            basis = f"one foot unloaded in {single_fraction * 100:.0f}% of frames"
        else:
            bearing = "bilateral"
            basis = f"both feet loaded in {(1 - single_fraction) * 100:.0f}% of frames"

        result.values["legs_weight_bearing"] = bearing
        result.evidence["legs_weight_bearing"] = DynamicsEvidence(
            condition="legs_weight_bearing",
            value=bearing,
            basis=basis,
            measured=single_fraction,
            threshold=policy.unilateral_minimum_fraction,
            units="fraction of frames on one foot",
        )

        stance_variation = self._stance_variation(samples)
        unstable = (
            stance_variation is not None
            and stance_variation > policy.unstable_base_stance_variation
        )
        supported = bool(bearing == "bilateral" and not unstable)
        result.values["legs_supported_and_balanced"] = supported
        result.evidence["legs_supported_and_balanced"] = DynamicsEvidence(
            condition="legs_supported_and_balanced",
            value=supported,
            basis=f"weight bearing is {bearing}"
            + ("; stance varied" if unstable else "; stance steady"),
            measured=stance_variation,
            threshold=policy.unstable_base_stance_variation,
            units="stance variation as fraction of leg length",
        )

    # ------------------------------------------------------------------
    # RULA load type
    # ------------------------------------------------------------------

    def _load_type(self, static: bool | None, repeated: bool | None, result: TaskDynamics) -> None:
        """RULA's muscle-use category follows directly from hold and repeat."""

        if static is True or repeated is True:
            value: str | None = "static_or_repeated"
            basis = "posture held over a minute" if static else "action repeats over 4/min"
        elif static is False and repeated is False:
            value = "intermittent"
            basis = "posture neither held over a minute nor repeated over 4/min"
        else:
            value = None
            basis = "hold time or repetition rate is unknown"

        result.values["rula_load_type"] = value
        result.evidence["rula_load_type"] = DynamicsEvidence(
            condition="rula_load_type",
            value=value,
            basis=basis,
        )


def analyze_task_dynamics(
    samples: Sequence[PostureSample], policy: TaskDynamicsPolicy | None = None
) -> TaskDynamics:
    """Convenience wrapper for one-off analysis."""

    return TaskDynamicsAnalyzer(policy).analyze(samples)


def policy_from_config(config: dict[str, Any] | None) -> TaskDynamicsPolicy:
    """Build a policy from the ``task_dynamics`` config block."""

    defaults = TaskDynamicsPolicy()
    if not config:
        return defaults
    if not isinstance(config, dict):
        raise ValueError("task_dynamics must be a mapping")
    known = set(vars(defaults))
    unknown = sorted(set(config) - known)
    if unknown:
        raise ValueError(f"unknown task dynamics settings: {unknown}; available: {sorted(known)}")
    # Typed as Any because the policy mixes int and float fields; each value is
    # coerced to the field's own type below.
    values: dict[str, Any] = {
        name: int(value) if name.endswith("_frames") else float(value)
        for name, value in config.items()
    }
    return replace(defaults, **values)
