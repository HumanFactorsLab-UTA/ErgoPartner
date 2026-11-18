"""Regression check: the decode-to-score path still produces the pinned output.

These tests are skipped when the study footage is unavailable, so the suite
still runs on a machine that only has the repository. Regenerate the snapshot
deliberately with ``python tools/capture_baseline.py --write`` after an
intentional change, and record why in the commit message.
"""

from __future__ import annotations

import pytest

from tests.regression import baseline

pytestmark = pytest.mark.regression


@pytest.fixture(scope="module")
def snapshot() -> dict:
    stored = baseline.load_baseline()
    if stored is None:
        pytest.skip("No baseline snapshot recorded; run tools/capture_baseline.py --write")
    return stored


@pytest.mark.parametrize("scenario", baseline.SCENARIOS, ids=lambda item: item.name)
def test_scenario_matches_baseline(scenario: baseline.Scenario, snapshot: dict) -> None:
    if not baseline.footage_available(scenario):
        pytest.skip(baseline.missing_footage_reason())
    expected = snapshot.get("scenarios", {}).get(scenario.name)
    if expected is None:
        pytest.skip(f"Scenario {scenario.name} is not present in the recorded baseline")

    actual = baseline.run_scenario(scenario)
    differences = list(baseline.compare_scenario(expected, actual))
    assert not differences, "Pipeline output changed against the recorded baseline:\n" + "\n".join(
        f"  - {line}" for line in differences
    )


def test_baseline_covers_the_documented_frame_range(snapshot: dict) -> None:
    assert snapshot["start_frame"] == baseline.START_FRAME
    assert snapshot["end_frame"] == baseline.END_FRAME


def _snapshot(angle: float = 10.0, score: int | None = 3, frame: int = 1) -> dict:
    return {
        "counts": {"processed_frames": 1, "valid_rula_observations": 1},
        "frames": [
            {
                "frame_number": frame,
                "timestamp_sec": 0.0,
                "angles": {"trunk_flexion": angle},
                "rula": {"final_score": score},
                "reba": {"final_score": None},
            }
        ],
    }


def test_compare_reports_nothing_for_identical_snapshots() -> None:
    assert list(baseline.compare_scenario(_snapshot(), _snapshot())) == []


def test_compare_ignores_floating_point_noise() -> None:
    nudged = _snapshot(angle=10.0 + baseline.ANGLE_TOLERANCE_DEG / 10)
    assert list(baseline.compare_scenario(_snapshot(), nudged)) == []


def test_compare_detects_angle_drift() -> None:
    differences = list(baseline.compare_scenario(_snapshot(), _snapshot(angle=12.5)))
    assert any("trunk_flexion" in line and "2.5" in line for line in differences)


def test_compare_detects_score_change() -> None:
    differences = list(baseline.compare_scenario(_snapshot(), _snapshot(score=5)))
    assert any("scores changed" in line for line in differences)
    assert any("3 -> 5" in line for line in differences)


def test_compare_detects_lost_and_gained_frames() -> None:
    differences = list(baseline.compare_scenario(_snapshot(frame=1), _snapshot(frame=2)))
    assert any("baseline but not now" in line for line in differences)
    assert any("now but not in baseline" in line for line in differences)


def test_compare_detects_count_change() -> None:
    changed = _snapshot()
    changed["counts"]["valid_rula_observations"] = 0
    differences = list(baseline.compare_scenario(_snapshot(), changed))
    assert any("valid_rula_observations: baseline 1 -> now 0" in line for line in differences)


def test_scored_scenario_actually_reaches_a_score(snapshot: dict) -> None:
    """A baseline of nothing but None would not detect a scoring regression."""

    scored = snapshot.get("scenarios", {}).get("single_camera_scored")
    if scored is None:
        pytest.skip("Scored scenario is not present in the recorded baseline")
    assert scored["counts"]["valid_rula_observations"] > 0, (
        "The scored baseline scenario must contain real RULA scores, otherwise it "
        "cannot detect a change in scoring behaviour."
    )
