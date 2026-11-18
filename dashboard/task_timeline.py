"""When in the task the risk happened, and which body part drove it.

Every other summary in this project collapses the time axis. Maxima,
percentiles, band exposure, and range of motion all answer "how bad was this
trial" and none of them answer "when". An assessor watching a two-minute lifting
trial does not need to be told the worst REBA was 13; they need to be told the
worst REBA was 13 at 0:44, that it lasted two and a half seconds, that the trunk
drove it, and where to look.

This module produces that. It reads the per-frame RULA/REBA results the pipeline
already writes -- including the component scores each method builds its total
from -- and returns three things:

* a dwell-weighted score series, so "38% of the task was at High risk" means 38%
  of the *time* and not 38% of the samples;
* risk episodes: contiguous stretches at or above a chosen action level, each
  carrying its peak frame and the body part that was worst at that peak;
* per-body-part load, ranked by how much of the task each part actually spent
  away from its own neutral.

It calculates no new ergonomic score. Every number here is either a component
the published method itself produced, or a duration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from statistics import median
from typing import Any, NamedTuple

from dashboard.viz_theme import RiskBand, method_bands, risk_band

#: Sampling can be irregular, and a long gap must not be counted as dwell time.
#: Anything longer than this between two samples is excluded from the timeline's
#: duration totals, matching ``temporal.aggregation``'s treatment of gaps.
DEFAULT_MAXIMUM_GAP_SEC = 0.5

#: An episode shorter than this is a spike, not a stretch of work. One second is
#: the project's existing sustained-posture floor.
DEFAULT_MINIMUM_EPISODE_SEC = 0.6


class ComponentSpec(NamedTuple):
    """One scored component of one method, and how to read it.

    ``kind`` separates the two things a reviewer can act on. A ``posture``
    component is a body part -- change the workstation or the reach. A ``task``
    component is a condition of the job -- the load, the grip, the repetition
    rate. Both add points to the final score and both belong in an explanation,
    but they lead to different interventions, so they are never mixed into one
    ranked list.
    """

    component: str
    region: str
    minimum: int
    maximum: int
    sided: bool
    kind: str = "posture"


# Ranges are the published clamps in ergonomics/rula.py and ergonomics/reba.py.
# ``minimum`` is the score a neutral posture or an absent condition earns.
# Normalising against the raw maximum instead -- what the previous body map did
# -- made a perfectly neutral neck read as 1/6 of the way to critical, so every
# joint on every trial looked mildly implicated.
RULA_COMPONENTS: tuple[ComponentSpec, ...] = (
    ComponentSpec("trunk", "trunk", 1, 6, False),
    ComponentSpec("neck", "neck", 1, 6, False),
    ComponentSpec("legs", "legs", 1, 2, False),
    ComponentSpec("upper_arm", "shoulder", 1, 6, True),
    ComponentSpec("lower_arm", "elbow", 1, 3, True),
    ComponentSpec("wrist", "wrist", 1, 4, True),
    ComponentSpec("wrist_twist", "wrist_twist", 1, 2, False, "task"),
    ComponentSpec("muscle_use", "muscle_use", 0, 1, False, "task"),
    ComponentSpec("force_load", "force_load", 0, 3, False, "task"),
)
REBA_COMPONENTS: tuple[ComponentSpec, ...] = (
    ComponentSpec("trunk", "trunk", 1, 5, False),
    ComponentSpec("neck", "neck", 1, 3, False),
    ComponentSpec("legs", "legs", 1, 4, False),
    ComponentSpec("upper_arm", "shoulder", 1, 6, True),
    ComponentSpec("lower_arm", "elbow", 1, 2, True),
    ComponentSpec("wrist", "wrist", 1, 3, True),
    ComponentSpec("load_force", "load_force", 0, 3, False, "task"),
    ComponentSpec("coupling", "coupling", 0, 3, False, "task"),
    ComponentSpec("activity", "activity", 0, 3, False, "task"),
)

#: Task components are not body parts, so they need their own display names.
TASK_LABELS: Mapping[str, str] = {
    "wrist_twist": "Wrist twist",
    "muscle_use": "Static or repeated posture",
    "force_load": "Force and load",
    "load_force": "Force and load",
    "coupling": "Hand coupling",
    "activity": "Activity (static, repeated, unstable)",
}

POSTURE_COMPONENTS = {"trunk", "neck", "legs", "upper_arm", "lower_arm", "wrist"}

REGION_LABELS: Mapping[str, str] = {
    "neck": "Neck",
    "trunk": "Trunk",
    "left_shoulder": "Left shoulder",
    "right_shoulder": "Right shoulder",
    "left_elbow": "Left elbow",
    "right_elbow": "Right elbow",
    "left_wrist": "Left wrist",
    "right_wrist": "Right wrist",
    "legs": "Legs and balance",
}

#: The angle that most directly explains a raised component, so a driver can be
#: reported as "Trunk, scored 4 of 5, bent 68 degrees" rather than a bare number.
REGION_MEASUREMENTS: Mapping[str, tuple[str, ...]] = {
    "neck": ("neck_flexion",),
    "trunk": ("trunk_flexion",),
    "legs": ("left_knee_flexion", "right_knee_flexion"),
    "left_shoulder": ("left_upper_arm_flexion",),
    "right_shoulder": ("right_upper_arm_flexion",),
    "left_elbow": ("left_forearm_flexion",),
    "right_elbow": ("right_forearm_flexion",),
    "left_wrist": ("left_wrist_flexion",),
    "right_wrist": ("right_wrist_flexion",),
}


def components_for(method: str) -> tuple[ComponentSpec, ...]:
    return RULA_COMPONENTS if str(method).lower() == "rula" else REBA_COMPONENTS


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    number = _finite(value)
    return None if number is None else int(number)


def region_key(spec: ComponentSpec, side: str | None) -> str:
    """Resolve a component to a body region, taking the anatomical side."""

    if not spec.sided:
        return spec.region
    return f"{side}_{spec.region}" if side in ("left", "right") else f"left_{spec.region}"


def severity(score: int | None, spec: ComponentSpec) -> float | None:
    """Fraction of the component's own range above its neutral score.

    Zero means the method scored this part as neutral. One means it scored the
    worst the method allows.
    """

    if score is None:
        return None
    span = spec.maximum - spec.minimum
    if span <= 0:
        return 0.0
    return max(0.0, min(1.0, (float(score) - spec.minimum) / span))


def component_label(component: str, region: str) -> str:
    if component in TASK_LABELS:
        return TASK_LABELS[component]
    return REGION_LABELS.get(region, region.replace("_", " ").title())


# ----------------------------------------------------------------------
# What each component actually adds to the final score
# ----------------------------------------------------------------------


@lru_cache(maxsize=1)
def _rula_calculator() -> Any:
    # Imported lazily and held once: the ergonomic calculators are pure and
    # stateless, and this runs a few times per frame across a long trial.
    from ergonomics.rula import RULACalculator

    return RULACalculator()


@lru_cache(maxsize=1)
def _reba_calculator() -> Any:
    from ergonomics.reba import REBACalculator

    return REBACalculator()


def _rula_final(components: Mapping[str, int]) -> int | None:
    """Reassemble a RULA total from its components, using RULA's own tables."""

    calculator = _rula_calculator()
    try:
        table_a = calculator.lookup_table_a(
            components["upper_arm"],
            components["lower_arm"],
            components["wrist"],
            components["wrist_twist"],
        )
        table_b = calculator.lookup_table_b(
            components["neck"], components["trunk"], components["legs"]
        )
        muscle = components["muscle_use"]
        force = components["force_load"]
        return calculator.lookup_table_c(table_a + muscle + force, table_b + muscle + force)
    except (KeyError, ValueError, IndexError):
        return None


def _reba_final(components: Mapping[str, int]) -> int | None:
    """Reassemble a REBA total from its components, using REBA's own tables."""

    calculator = _reba_calculator()
    try:
        table_a = calculator.lookup_table_a(
            components["trunk"], components["neck"], components["legs"]
        )
        table_b = calculator.lookup_table_b(
            components["upper_arm"], components["lower_arm"], components["wrist"]
        )
        score_a = table_a + components["load_force"]
        score_b = table_b + components["coupling"]
        return calculator.lookup_table_c(score_a, score_b) + components["activity"]
    except (KeyError, ValueError, IndexError):
        return None


def recompute_final(components: Mapping[str, int], method: str) -> int | None:
    """Return the method's final score implied by these component scores.

    This exists so a component can be set back to neutral and the score
    recomputed, which is the only honest way to say what a body part contributed.
    It is not a second scoring implementation: it calls the same published
    lookup tables the pipeline used, and ``verify_reconstruction`` asserts it
    reproduces the stored totals exactly.
    """

    return _rula_final(components) if str(method).lower() == "rula" else _reba_final(components)


def points_from(components: Mapping[str, int], method: str, spec: ComponentSpec) -> int | None:
    """How many points the final score would drop if this component were neutral.

    This is the number the previous ranking was missing. Severity alone ranks a
    component that is pinned at the top of a two-value scale -- REBA's lower arm
    is 1 or 2 and nothing else -- above a trunk that swings through five levels,
    because both read as "at maximum". Neutralising the component and asking the
    method what changes gives the trunk its due and correctly gives no credit to
    a component the lookup table absorbs.
    """

    actual = recompute_final(components, method)
    if actual is None:
        return None
    neutral = dict(components)
    neutral[spec.component] = spec.minimum
    relieved = recompute_final(neutral, method)
    return None if relieved is None else max(0, actual - relieved)


def verify_reconstruction(
    records: Sequence[Mapping[str, Any]], method: str
) -> tuple[int, int]:
    """Return (matched, checked): how often the reassembled total equals the stored one."""

    key = str(method).lower()
    matched = 0
    checked = 0
    for record in records:
        result = _mapping(record.get(key))
        for side in ("left", "right"):
            side_result = _mapping(result.get(side))
            stored = _integer(side_result.get("final_score"))
            if stored is None:
                continue
            checked += 1
            if recompute_final(_component_scores(side_result), key) == stored:
                matched += 1
    return matched, checked


# ----------------------------------------------------------------------
# Score series
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScorePoint:
    """One scored observation, with the time it stands for."""

    method: str
    frame_number: int
    timestamp_sec: float
    score: int
    side: str | None
    dwell_sec: float
    components: Mapping[str, int]
    left_components: Mapping[str, int]
    right_components: Mapping[str, int]
    angles: Mapping[str, float]

    @property
    def band(self) -> RiskBand | None:
        return risk_band(self.score, self.method)


def _component_scores(side_result: Any) -> dict[str, int]:
    scores = _mapping(_mapping(side_result).get("intermediate_scores"))
    resolved: dict[str, int] = {}
    for name, value in scores.items():
        number = _integer(value)
        if number is not None:
            resolved[str(name)] = number
    return resolved


def score_series(
    records: Sequence[Mapping[str, Any]],
    method: str,
    *,
    maximum_gap_sec: float = DEFAULT_MAXIMUM_GAP_SEC,
) -> list[ScorePoint]:
    """Return the method's scored observations in time order, with dwell times.

    ``dwell_sec`` is how long each sample stands for: the interval to the next
    sample, capped at ``maximum_gap_sec`` so a break in tracking is not credited
    as time spent in a posture. The last sample takes the median dwell rather
    than zero, because a trial's final observation represents as much work as any
    other and dropping it biases every duration slightly short.
    """

    key = str(method).lower()
    rows: list[tuple[int, float, int, str | None, dict[str, Any]]] = []
    for record in records:
        result = _mapping(record.get(key))
        if not result:
            continue
        selected = _mapping(result.get("selected"))
        score = _integer(selected.get("final_score"))
        frame = _integer(record.get("frame_number"))
        timestamp = _finite(record.get("timestamp_sec"))
        if score is None or frame is None or timestamp is None:
            continue
        side = result.get("selected_side") or selected.get("side")
        rows.append(
            (
                frame,
                timestamp,
                score,
                str(side) if side is not None else None,
                {
                    "selected": _component_scores(selected),
                    "left": _component_scores(result.get("left")),
                    "right": _component_scores(result.get("right")),
                    "angles": {
                        str(name): number
                        for name, value in _mapping(record.get("angles")).items()
                        if (number := _finite(value)) is not None
                    },
                },
            )
        )
    rows.sort(key=lambda row: (row[1], row[0]))
    if not rows:
        return []

    gaps = [
        second[1] - first[1]
        for first, second in zip(rows, rows[1:], strict=False)
        if 0 < second[1] - first[1] <= maximum_gap_sec
    ]
    trailing_dwell = median(gaps) if gaps else 0.0

    points: list[ScorePoint] = []
    for index, (frame, timestamp, score, side, extra) in enumerate(rows):
        if index + 1 < len(rows):
            delta = rows[index + 1][1] - timestamp
            dwell = delta if 0 < delta <= maximum_gap_sec else 0.0
        else:
            dwell = trailing_dwell
        points.append(
            ScorePoint(
                method=key,
                frame_number=frame,
                timestamp_sec=timestamp,
                score=score,
                side=side,
                dwell_sec=float(dwell),
                components=extra["selected"],
                left_components=extra["left"],
                right_components=extra["right"],
                angles=extra["angles"],
            )
        )
    return points


# ----------------------------------------------------------------------
# Time spent at each action level
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BandShare:
    band: RiskBand
    observation_count: int
    duration_sec: float
    percent_of_scored_time: float


def band_shares(series: Sequence[ScorePoint], method: str) -> list[BandShare]:
    """Return time spent at each of the method's published action levels."""

    bands = method_bands(method)
    counts = dict.fromkeys(range(len(bands)), 0)
    durations = dict.fromkeys(range(len(bands)), 0.0)
    for point in series:
        # A score above the published scale still lands in the top band rather
        # than being dropped; an out-of-range value is a data fault to report,
        # not a reason to lose time out of the totals.
        index = next(
            (position for position, band in enumerate(bands) if point.score <= band.upper),
            len(bands) - 1,
        )
        counts[index] += 1
        durations[index] += point.dwell_sec
    total = sum(durations.values())
    return [
        BandShare(
            band=band,
            observation_count=counts[index],
            duration_sec=durations[index],
            percent_of_scored_time=(100.0 * durations[index] / total) if total > 0 else 0.0,
        )
        for index, band in enumerate(bands)
    ]


def scored_duration_sec(series: Sequence[ScorePoint]) -> float:
    return float(sum(point.dwell_sec for point in series))


def time_at_or_above(series: Sequence[ScorePoint], threshold: int) -> tuple[float, float]:
    """Return (seconds, percent of scored time) at or above a score."""

    total = scored_duration_sec(series)
    above = sum(point.dwell_sec for point in series if point.score >= threshold)
    return above, (100.0 * above / total if total > 0 else 0.0)


# ----------------------------------------------------------------------
# Drivers
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Driver:
    """One component's contribution to a score at one moment."""

    region: str
    label: str
    component: str
    kind: str
    component_score: int
    component_maximum: int
    severity: float
    points_added: int
    measurement: str | None
    value_deg: float | None

    @property
    def score_text(self) -> str:
        return f"{self.component_score} of {self.component_maximum}"

    @property
    def points_text(self) -> str:
        if self.points_added <= 0:
            return "no change to the total"
        return f"+{self.points_added} point" + ("" if self.points_added == 1 else "s")


def _supporting_angle(region: str, angles: Mapping[str, float]) -> tuple[str | None, float | None]:
    candidates = [
        (name, angles[name]) for name in REGION_MEASUREMENTS.get(region, ()) if name in angles
    ]
    if not candidates:
        return None, None
    name, value = max(candidates, key=lambda item: abs(item[1]))
    return name, value


def _scored_side(point: ScorePoint) -> tuple[str | None, Mapping[str, int]]:
    """The components behind the score that was actually reported.

    RULA and REBA are calculated for both arms and the worse one is reported, so
    the total on screen belongs to exactly one side. Explaining it means
    explaining that side's components; mixing the other arm's numbers into the
    same ranking would attribute points to a score nobody is looking at.
    """

    if point.components:
        return point.side, point.components
    if point.left_components:
        return "left", point.left_components
    return "right", point.right_components


def drivers_at(point: ScorePoint, method: str, *, limit: int = 3, kind: str | None = None) -> list[Driver]:
    """Rank what pushed this observation's score up, worst first.

    Ranking is by points added to the total, then by how far the component sits
    through its own range. A component the lookup table absorbs adds nothing and
    sorts below one that moves the total, however extreme it looks on its own.
    """

    side, scores = _scored_side(point)
    if not scores:
        return []
    found: list[Driver] = []
    for spec in components_for(method):
        if kind is not None and spec.kind != kind:
            continue
        value = scores.get(spec.component)
        if value is None:
            continue
        fraction = severity(value, spec)
        if fraction is None or fraction <= 0:
            continue
        region = region_key(spec, side) if spec.sided else spec.region
        measurement, angle = _supporting_angle(region, point.angles)
        found.append(
            Driver(
                region=region,
                label=component_label(spec.component, region),
                component=spec.component,
                kind=spec.kind,
                component_score=value,
                component_maximum=spec.maximum,
                severity=fraction,
                points_added=points_from(scores, method, spec) or 0,
                measurement=measurement,
                value_deg=angle,
            )
        )
    found.sort(key=lambda item: (-item.points_added, -item.severity, item.region))
    return found[:limit]


# ----------------------------------------------------------------------
# Episodes: the answer to "when"
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RiskEpisode:
    """A contiguous stretch of the task at or above an action level."""

    index: int
    method: str
    band: RiskBand
    start_sec: float
    end_sec: float
    duration_sec: float
    start_frame_number: int
    end_frame_number: int
    peak_score: int
    peak_frame_number: int
    peak_timestamp_sec: float
    observation_count: int
    drivers: tuple[Driver, ...]

    @property
    def peak_band(self) -> RiskBand | None:
        return risk_band(self.peak_score, self.method)

    @property
    def time_range_text(self) -> str:
        return f"{clock(self.start_sec)}–{clock(self.end_sec)}"


def clock(seconds: float | None) -> str:
    """Format a time as m:ss, which is how a reviewer scrubs a video."""

    if seconds is None or not math.isfinite(float(seconds)) or seconds < 0:
        return "—"
    total = int(round(float(seconds)))
    return f"{total // 60}:{total % 60:02d}"


def action_threshold(method: str, level: str = "serious") -> int:
    """The lowest score in the first band at or above a status severity.

    Defaulting to ``serious`` means RULA 5 and REBA 8, which are the same
    thresholds the pipeline already uses for sustained high-risk periods, so the
    episodes on the Results page and the periods in the summary agree.
    """

    order = ("good", "warning", "serious", "critical")
    wanted = order.index(level) if level in order else 2
    for band in method_bands(method):
        if order.index(band.status) >= wanted:
            return band.lower
    return method_bands(method)[-1].lower


def risk_episodes(
    series: Sequence[ScorePoint],
    method: str,
    *,
    threshold: int | None = None,
    minimum_duration_sec: float = DEFAULT_MINIMUM_EPISODE_SEC,
    maximum_gap_sec: float = DEFAULT_MAXIMUM_GAP_SEC,
    limit: int | None = None,
) -> list[RiskEpisode]:
    """Group the timeline into stretches at or above ``threshold``.

    An episode's duration is the sum of its samples' dwell times, so a single
    observation held for one sample interval is not reported as lasting zero
    seconds -- which is what happens when duration is taken as last minus first
    timestamp, and is why a brief spike at the top of the scale could vanish from
    a sustained-period list entirely.

    Episodes are returned worst first: peak score, then duration. The reviewer
    wants the worst moment at the top of the list, not the earliest.
    """

    cut = action_threshold(method) if threshold is None else int(threshold)
    runs: list[list[ScorePoint]] = []
    current: list[ScorePoint] = []
    for point in series:
        if point.score >= cut:
            if current and point.timestamp_sec - current[-1].timestamp_sec > maximum_gap_sec:
                runs.append(current)
                current = []
            current.append(point)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)

    episodes: list[RiskEpisode] = []
    for run in runs:
        duration = sum(point.dwell_sec for point in run)
        if duration < minimum_duration_sec:
            continue
        peak = max(run, key=lambda point: (point.score, -point.timestamp_sec))
        band = risk_band(peak.score, method)
        if band is None:
            continue
        episodes.append(
            RiskEpisode(
                index=0,
                method=str(method).upper(),
                band=band,
                start_sec=run[0].timestamp_sec,
                end_sec=run[-1].timestamp_sec + run[-1].dwell_sec,
                duration_sec=duration,
                start_frame_number=run[0].frame_number,
                end_frame_number=run[-1].frame_number,
                peak_score=peak.score,
                peak_frame_number=peak.frame_number,
                peak_timestamp_sec=peak.timestamp_sec,
                observation_count=len(run),
                drivers=tuple(drivers_at(peak, method)),
            )
        )
    episodes.sort(key=lambda item: (-item.peak_score, -item.duration_sec, item.start_sec))
    ranked = [
        RiskEpisode(
            index=position + 1,
            method=item.method,
            band=item.band,
            start_sec=item.start_sec,
            end_sec=item.end_sec,
            duration_sec=item.duration_sec,
            start_frame_number=item.start_frame_number,
            end_frame_number=item.end_frame_number,
            peak_score=item.peak_score,
            peak_frame_number=item.peak_frame_number,
            peak_timestamp_sec=item.peak_timestamp_sec,
            observation_count=item.observation_count,
            drivers=item.drivers,
        )
        for position, item in enumerate(episodes)
    ]
    return ranked if limit is None else ranked[:limit]


# ----------------------------------------------------------------------
# Body-part load: the answer to "where"
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BodyPartLoad:
    """What one component cost the score across the whole selection."""

    region: str
    label: str
    component: str
    kind: str
    component_maximum: int
    worst_score: int
    worst_severity: float
    mean_points: float
    worst_points: int
    contributing_duration_sec: float
    percent_of_scored_time: float
    raised_duration_sec: float
    percent_raised: float
    peak_frame_number: int | None
    peak_timestamp_sec: float | None
    measurement: str | None
    peak_value_deg: float | None
    observation_count: int

    @property
    def worst_score_text(self) -> str:
        return f"{self.worst_score} of {self.component_maximum}"


def body_part_load(
    series: Sequence[ScorePoint], method: str, *, kind: str | None = None
) -> list[BodyPartLoad]:
    """Rank components by the points they added to the score, over time.

    Ranking by "was this part ever non-neutral" -- what the previous body map did
    -- puts every part in the same red on any real trial, because over two
    minutes nearly everything leaves neutral once. Ranking by mean points added
    separates a trunk that is bent through the whole lift from a leg flag that
    was pinned at its maximum for the entire recording and, because the lookup
    table absorbs it, changed the total by nothing.

    ``mean_points`` is dwell-weighted across the whole selection including frames
    where the component was neutral, so it reads directly as "this part adds N
    points to the average score of this task".
    """

    total = scored_duration_sec(series)
    accumulator: dict[str, dict[str, Any]] = {}
    for point in series:
        side, scores = _scored_side(point)
        if not scores:
            continue
        for spec in components_for(method):
            if kind is not None and spec.kind != kind:
                continue
            value = scores.get(spec.component)
            if value is None:
                continue
            fraction = severity(value, spec)
            if fraction is None:
                continue
            points = points_from(scores, method, spec) or 0
            region = region_key(spec, side) if spec.sided else spec.region
            state = accumulator.setdefault(
                region,
                {
                    "component": spec.component,
                    "kind": spec.kind,
                    "maximum": spec.maximum,
                    "worst": value,
                    "worst_severity": fraction,
                    "worst_points": points,
                    "point_time": 0.0,
                    "contributing": 0.0,
                    "raised": 0.0,
                    "count": 0,
                    "peak_point": point,
                },
            )
            state["count"] += 1
            state["point_time"] += points * point.dwell_sec
            if points > 0:
                state["contributing"] += point.dwell_sec
            if fraction > 0:
                state["raised"] += point.dwell_sec
            state["worst_points"] = max(state["worst_points"], points)
            if value > state["worst"]:
                state["worst"] = value
                state["worst_severity"] = fraction
                state["peak_point"] = point

    loads: list[BodyPartLoad] = []
    for region, state in accumulator.items():
        peak_point: ScorePoint = state["peak_point"]
        measurement, angle = _supporting_angle(region, peak_point.angles)
        loads.append(
            BodyPartLoad(
                region=region,
                label=component_label(state["component"], region),
                component=state["component"],
                kind=state["kind"],
                component_maximum=state["maximum"],
                worst_score=state["worst"],
                worst_severity=state["worst_severity"],
                mean_points=(state["point_time"] / total) if total > 0 else 0.0,
                worst_points=state["worst_points"],
                contributing_duration_sec=state["contributing"],
                percent_of_scored_time=(
                    100.0 * state["contributing"] / total if total > 0 else 0.0
                ),
                raised_duration_sec=state["raised"],
                percent_raised=(100.0 * state["raised"] / total) if total > 0 else 0.0,
                peak_frame_number=peak_point.frame_number,
                peak_timestamp_sec=peak_point.timestamp_sec,
                measurement=measurement,
                peak_value_deg=angle,
                observation_count=state["count"],
            )
        )
    loads.sort(key=lambda item: (-item.mean_points, -item.worst_points, item.region))
    return loads


# ----------------------------------------------------------------------
# The verdict
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Verdict:
    """The one paragraph a reviewer should be able to read and act on."""

    method: str
    scored: bool
    peak_score: int | None
    peak_band: RiskBand | None
    peak_frame_number: int | None
    peak_timestamp_sec: float | None
    typical_score: float | None
    typical_band: RiskBand | None
    scored_duration_sec: float
    time_at_action_sec: float
    percent_at_action: float
    action_threshold: int
    episode_count: int
    leading_region: str | None
    leading_label: str | None
    leading_points: float | None
    leading_task_label: str | None
    leading_task_points: float | None
    headline: str
    detail: str


def verdict(
    series: Sequence[ScorePoint],
    method: str,
    *,
    episodes: Sequence[RiskEpisode] | None = None,
    loads: Sequence[BodyPartLoad] | None = None,
) -> Verdict:
    """Return the plain-language summary for one method.

    Two numbers carry it. The peak says how bad it got; the share of time at or
    above the action threshold says whether that was one awkward reach or the way
    the job is done. A trial that touches REBA 13 once is a different problem
    from one that sits above 8 for a third of its duration, and reporting only
    the maximum -- which is what the Results page did -- cannot tell them apart.
    """

    upper = str(method).upper()
    threshold = action_threshold(method)
    if not series:
        return Verdict(
            method=upper,
            scored=False,
            peak_score=None,
            peak_band=None,
            peak_frame_number=None,
            peak_timestamp_sec=None,
            typical_score=None,
            typical_band=None,
            scored_duration_sec=0.0,
            time_at_action_sec=0.0,
            percent_at_action=0.0,
            action_threshold=threshold,
            episode_count=0,
            leading_region=None,
            leading_label=None,
            leading_points=None,
            leading_task_label=None,
            leading_task_points=None,
            headline=f"{upper} produced no valid score in this selection",
            detail=(
                "No frame in scope carried a complete set of inputs for this method. "
                "Open the evidence details to review missing inputs."
            ),
        )

    episodes = list(episodes if episodes is not None else risk_episodes(series, method))
    loads = list(loads if loads is not None else body_part_load(series, method))
    peak = max(series, key=lambda point: (point.score, -point.timestamp_sec))
    band = risk_band(peak.score, method)
    total = scored_duration_sec(series)
    at_action, percent = time_at_or_above(series, threshold)

    weighted = sum(point.score * point.dwell_sec for point in series)
    typical = (weighted / total) if total > 0 else float(sum(p.score for p in series)) / len(series)
    typical_band = risk_band(round(typical), method)

    postural = [item for item in loads if item.kind == "posture" and item.mean_points > 0]
    task_factors = [item for item in loads if item.kind == "task" and item.mean_points > 0]
    leading = postural[0] if postural else None
    leading_task = task_factors[0] if task_factors else None
    if band is None:
        headline = f"{upper} peaked at {peak.score}."
    elif band.describes_risk:
        headline = f"{upper} peaked at {peak.score} ({band.label.lower()} risk). {band.action}."
    else:
        # RULA's band names are already instructions, so appending "risk" to them
        # produced "change now risk". The action sentence carries it alone.
        headline = f"{upper} peaked at {peak.score}. {band.action}."
    peak_drivers = drivers_at(peak, method, limit=1, kind="posture")
    parts = [f"The peak occurred at {clock(peak.timestamp_sec)}."]
    if peak_drivers and peak_drivers[0].points_added > 0:
        parts.append(f"The {peak_drivers[0].label.lower()} contributed most at that moment.")
    peak_episode = next(
        (episode for episode in episodes
         if episode.start_frame_number <= peak.frame_number <= episode.end_frame_number),
        None,
    )
    if peak_episode is not None:
        parts.append(
            f"The period containing the peak lasted {peak_episode.duration_sec:.1f} s "
            f"({clock(peak_episode.start_sec)} to {clock(peak_episode.end_sec)})."
        )
    parts.append(
        f"{percent:.0f}% of the scored time ({at_action:.0f} s of {total:.0f} s) "
        f"was at {upper} {threshold} or above."
    )
    if leading is not None:
        parts.append(f"Across the recording, the largest postural contribution came from the {leading.label.lower()}.")
    return Verdict(
        method=upper,
        scored=True,
        peak_score=peak.score,
        peak_band=band,
        peak_frame_number=peak.frame_number,
        peak_timestamp_sec=peak.timestamp_sec,
        typical_score=typical,
        typical_band=typical_band,
        scored_duration_sec=total,
        time_at_action_sec=at_action,
        percent_at_action=percent,
        action_threshold=threshold,
        episode_count=len(episodes),
        leading_region=leading.region if leading else None,
        leading_label=leading.label if leading else None,
        leading_points=leading.mean_points if leading else None,
        leading_task_label=leading_task.label if leading_task else None,
        leading_task_points=leading_task.mean_points if leading_task else None,
        headline=headline,
        detail=" ".join(parts),
    )


__all__ = [
    "DEFAULT_MAXIMUM_GAP_SEC",
    "DEFAULT_MINIMUM_EPISODE_SEC",
    "POSTURE_COMPONENTS",
    "REBA_COMPONENTS",
    "REGION_LABELS",
    "REGION_MEASUREMENTS",
    "RULA_COMPONENTS",
    "TASK_LABELS",
    "BandShare",
    "BodyPartLoad",
    "ComponentSpec",
    "Driver",
    "RiskEpisode",
    "ScorePoint",
    "Verdict",
    "action_threshold",
    "band_shares",
    "body_part_load",
    "clock",
    "component_label",
    "components_for",
    "drivers_at",
    "points_from",
    "recompute_final",
    "region_key",
    "risk_episodes",
    "score_series",
    "scored_duration_sec",
    "severity",
    "time_at_or_above",
    "verdict",
    "verify_reconstruction",
]
