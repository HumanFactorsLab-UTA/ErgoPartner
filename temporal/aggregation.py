"""Timestamp-aware, deterministic video-level score aggregation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ScoreObservation:
    timestamp_sec: float
    score: int
    frame_number: int
    person_identifier: str


@dataclass(frozen=True)
class SustainedPeriod:
    start_sec: float
    end_sec: float
    duration_sec: float
    maximum_score: int
    person_identifier: str


@dataclass(frozen=True)
class ScoreSeriesSummary:
    observation_count: int
    maximum_score: int | None
    maximum_timestamp_sec: float | None
    maximum_frame_number: int | None
    high_percentile: float | None
    percentile: float
    duration_weighted_mean: float | None
    sustained_high_risk_periods: list[SustainedPeriod]
    aggregation_methods: list[str]

    def to_dict(self):
        return asdict(self)


def _validate_observations(observations: list[ScoreObservation]) -> None:
    for observation in observations:
        if observation.timestamp_sec < 0 or not np.isfinite(observation.timestamp_sec):
            raise ValueError("timestamps must be finite and non-negative")
        if isinstance(observation.score, bool) or observation.score < 1:
            raise ValueError("scores must be positive integers")
    for person in {item.person_identifier for item in observations}:
        times = [item.timestamp_sec for item in observations if item.person_identifier == person]
        if any(current <= previous for previous, current in zip(times, times[1:], strict=False)):
            raise ValueError("timestamps must increase strictly within each person track")


def summarize_score_series(
    observations: Iterable[ScoreObservation],
    *,
    percentile: float = 90.0,
    sustained_threshold: int,
    sustained_min_duration_sec: float = 1.0,
    maximum_gap_sec: float = 0.25,
) -> ScoreSeriesSummary:
    """Return max, percentile, time-weighted, and sustained-period summaries."""

    values = list(observations)
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    if sustained_min_duration_sec < 0 or maximum_gap_sec <= 0:
        raise ValueError("duration must be non-negative and maximum_gap_sec positive")
    values.sort(key=lambda item: (item.person_identifier, item.timestamp_sec, item.frame_number))
    _validate_observations(values)
    if not values:
        return ScoreSeriesSummary(
            observation_count=0,
            maximum_score=None,
            maximum_timestamp_sec=None,
            maximum_frame_number=None,
            high_percentile=None,
            duration_weighted_mean=None,
            percentile=percentile,
            sustained_high_risk_periods=[],
            aggregation_methods=[
                "maximum",
                f"percentile_{percentile:g}",
                "duration_weighted_mean",
                "sustained_high_risk",
            ],
        )

    # First occurrence is deterministic when equal maxima occur.
    maximum = max(values, key=lambda item: item.score)
    scores = np.asarray([item.score for item in values], dtype=float)
    weighted_total = 0.0
    duration_total = 0.0
    periods: list[SustainedPeriod] = []
    for person in sorted({item.person_identifier for item in values}):
        series = [item for item in values if item.person_identifier == person]
        # Each observed score contributes time up to the next sample of this person, within the gap limit.
        for first, second in zip(series, series[1:], strict=False):
            duration = second.timestamp_sec - first.timestamp_sec
            if duration <= maximum_gap_sec:
                weighted_total += first.score * duration
                duration_total += duration

        start: ScoreObservation | None = None
        last: ScoreObservation | None = None
        period_max = 0
        for observation in series:
            continues = (
                start is not None
                and last is not None
                and observation.timestamp_sec - last.timestamp_sec <= maximum_gap_sec
            )
            if observation.score >= sustained_threshold:
                if start is None or not continues:
                    if start is not None and last is not None:
                        duration = last.timestamp_sec - start.timestamp_sec
                        if duration >= sustained_min_duration_sec:
                            periods.append(
                                SustainedPeriod(
                                    start.timestamp_sec,
                                    last.timestamp_sec,
                                    duration,
                                    period_max,
                                    person,
                                )
                            )
                    start = observation
                    period_max = observation.score
                period_max = max(period_max, observation.score)
                last = observation
            else:
                if start is not None and last is not None:
                    duration = last.timestamp_sec - start.timestamp_sec
                    if duration >= sustained_min_duration_sec:
                        periods.append(
                            SustainedPeriod(
                                start.timestamp_sec,
                                last.timestamp_sec,
                                duration,
                                period_max,
                                person,
                            )
                        )
                start = None
                last = None
                period_max = 0
        if start is not None and last is not None:
            duration = last.timestamp_sec - start.timestamp_sec
            if duration >= sustained_min_duration_sec:
                periods.append(
                    SustainedPeriod(
                        start.timestamp_sec,
                        last.timestamp_sec,
                        duration,
                        period_max,
                        person,
                    )
                )

    return ScoreSeriesSummary(
        observation_count=len(values),
        maximum_score=maximum.score,
        maximum_timestamp_sec=maximum.timestamp_sec,
        maximum_frame_number=maximum.frame_number,
        high_percentile=float(np.percentile(scores, percentile)),
        percentile=percentile,
        duration_weighted_mean=(
            weighted_total / duration_total if duration_total else float(np.mean(scores))
        ),
        sustained_high_risk_periods=periods,
        aggregation_methods=[
            "maximum",
            f"percentile_{percentile:g}",
            "duration_weighted_mean",
            "sustained_high_risk",
        ],
    )
