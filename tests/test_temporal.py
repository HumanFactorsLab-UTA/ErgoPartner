import pytest

from temporal.aggregation import ScoreObservation, summarize_score_series


def observation(time, score, frame=0, person="person-0001"):
    return ScoreObservation(time, score, frame, person)


def test_temporal_summary_preserves_short_peak():
    values = [observation(0.0, 2, 0), observation(0.1, 7, 1), observation(0.2, 2, 2)]
    summary = summarize_score_series(values, sustained_threshold=5, sustained_min_duration_sec=0.5)
    assert summary.maximum_score == 7
    assert summary.maximum_timestamp_sec == 0.1
    assert summary.sustained_high_risk_periods == []


def test_sustained_period_uses_timestamps_not_frame_count():
    values = [observation(0.0, 8, 0), observation(0.5, 8, 1), observation(1.0, 9, 2)]
    summary = summarize_score_series(
        values, sustained_threshold=8, sustained_min_duration_sec=1.0, maximum_gap_sec=0.6
    )
    assert len(summary.sustained_high_risk_periods) == 1
    assert summary.sustained_high_risk_periods[0].duration_sec == pytest.approx(1.0)


def test_empty_summary_is_not_zero_score():
    summary = summarize_score_series([], sustained_threshold=5)
    assert summary.maximum_score is None
    assert summary.high_percentile is None
    assert summary.observation_count == 0
