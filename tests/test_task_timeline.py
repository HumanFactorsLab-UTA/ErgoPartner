"""When the risk happened, and which body part actually caused it.

These tests pin the two judgements the Results page now rests on: that a score
can be taken apart with the published tables to say what each body part cost,
and that time is weighted by how long each observation stood for rather than by
how many samples happened to land in it.
"""

from __future__ import annotations

import pytest

from dashboard.task_timeline import (
    REBA_COMPONENTS,
    RULA_COMPONENTS,
    band_shares,
    body_part_load,
    clock,
    components_for,
    drivers_at,
    points_from,
    recompute_final,
    risk_episodes,
    score_series,
    scored_duration_sec,
    severity,
    time_at_or_above,
    verdict,
    verify_reconstruction,
)
from dashboard.viz_theme import risk_action, risk_band, risk_label

NEUTRAL_REBA = {
    "trunk": 1,
    "neck": 1,
    "legs": 1,
    "upper_arm": 1,
    "lower_arm": 1,
    "wrist": 1,
    "load_force": 0,
    "coupling": 0,
    "activity": 0,
}
NEUTRAL_RULA = {
    "trunk": 1,
    "neck": 1,
    "legs": 1,
    "upper_arm": 1,
    "lower_arm": 1,
    "wrist": 1,
    "wrist_twist": 1,
    "muscle_use": 0,
    "force_load": 0,
}


def record(
    frame: int,
    timestamp: float,
    *,
    method: str = "reba",
    components: dict[str, int] | None = None,
    other_components: dict[str, int] | None = None,
    angles: dict[str, float] | None = None,
    side: str = "left",
    person: str = "person-0001",
):
    """A lightweight frame record shaped like the ones the store hands over.

    ``other_components`` fills in the anatomical side the method did not report,
    which only matters where both sides are drawn.
    """

    neutral = NEUTRAL_REBA if method == "reba" else NEUTRAL_RULA
    scores = dict(neutral) | (components or {})
    result = {
        "selected_side": side,
        "selected": {
            "final_score": recompute_final(scores, method),
            "intermediate_scores": scores,
            "side": side,
        },
        side: {
            "final_score": recompute_final(scores, method),
            "intermediate_scores": scores,
            "side": side,
        },
    }
    if other_components is not None:
        opposite = "right" if side == "left" else "left"
        other = dict(neutral) | other_components
        result[opposite] = {
            "final_score": recompute_final(other, method),
            "intermediate_scores": other,
            "side": opposite,
        }
    return {
        "frame_number": frame,
        "timestamp_sec": timestamp,
        "person_identifier": person,
        "angles": angles or {},
        method: result,
    }


# ----------------------------------------------------------------------
# The reconstruction the whole attribution rests on
# ----------------------------------------------------------------------


def test_reassembling_a_score_from_its_components_uses_the_published_tables():
    """If this drifts from the calculators, every attribution below is fiction."""

    from ergonomics.reba import REBACalculator
    from ergonomics.rula import RULACalculator

    reba = REBACalculator()
    components = dict(NEUTRAL_REBA) | {
        "trunk": 4,
        "neck": 2,
        "legs": 2,
        "upper_arm": 5,
        "lower_arm": 2,
        "wrist": 2,
        "coupling": 1,
        "activity": 2,
    }
    table_a = reba.lookup_table_a(4, 2, 2)
    table_b = reba.lookup_table_b(5, 2, 2)
    expected = reba.lookup_table_c(table_a + 0, table_b + 1) + 2
    assert recompute_final(components, "reba") == expected

    rula = RULACalculator()
    upper = dict(NEUTRAL_RULA) | {
        "upper_arm": 3,
        "lower_arm": 2,
        "wrist": 3,
        "wrist_twist": 2,
        "neck": 4,
        "trunk": 3,
        "legs": 2,
        "muscle_use": 1,
        "force_load": 1,
    }
    a = rula.lookup_table_a(3, 2, 3, 2)
    b = rula.lookup_table_b(4, 3, 2)
    assert recompute_final(upper, "rula") == rula.lookup_table_c(a + 2, b + 2)


def test_reconstruction_is_verified_against_stored_results_not_assumed():
    records = [
        record(1, 0.0, components={"trunk": 4, "upper_arm": 5}),
        record(2, 0.1, components={"trunk": 2, "neck": 2}),
    ]
    matched, checked = verify_reconstruction(records, "reba")

    assert checked == 2
    assert matched == checked


def test_an_incomplete_component_set_returns_no_score_rather_than_a_guess():
    assert recompute_final({"trunk": 3}, "reba") is None
    assert recompute_final({}, "rula") is None


# ----------------------------------------------------------------------
# Severity: neutral must be zero
# ----------------------------------------------------------------------


def test_a_neutral_component_is_zero_severity_not_a_fraction_of_the_maximum():
    """Dividing by the raw maximum made every neutral joint look implicated.

    A neck scored 1 of 6 is the method saying "this is fine"; reporting it as
    0.17 put a mild red on every joint of every trial.
    """

    trunk = next(spec for spec in REBA_COMPONENTS if spec.component == "trunk")

    assert severity(1, trunk) == 0.0
    assert severity(5, trunk) == 1.0
    assert severity(3, trunk) == pytest.approx(0.5)


# ----------------------------------------------------------------------
# What a component actually costs
# ----------------------------------------------------------------------


def test_points_from_reports_what_neutralising_a_component_would_recover():
    components = dict(NEUTRAL_REBA) | {"trunk": 5, "activity": 2}
    trunk = next(spec for spec in REBA_COMPONENTS if spec.component == "trunk")

    actual = recompute_final(components, "reba")
    relieved = recompute_final(components | {"trunk": 1}, "reba")

    assert points_from(components, "reba", trunk) == actual - relieved
    assert points_from(components, "reba", trunk) > 0


def test_a_component_the_lookup_table_absorbs_costs_nothing():
    """REBA's lower arm is 1 or 2 and often changes no total at all.

    Ranking by severity alone put it at the top of a real trial because it sat
    pinned at its maximum, which is exactly what a ranking must not do.
    """

    components = dict(NEUTRAL_REBA) | {"lower_arm": 2}
    lower = next(spec for spec in REBA_COMPONENTS if spec.component == "lower_arm")

    assert severity(2, lower) == 1.0
    assert points_from(components, "reba", lower) == 0


def test_a_pinned_binary_component_cannot_outrank_a_trunk_that_moves_the_score():
    series = score_series(
        [
            record(index, index * 0.1, components={"lower_arm": 2, "trunk": 4 if index % 2 else 5})
            for index in range(1, 21)
        ],
        "reba",
    )
    ranked = [item.region for item in body_part_load(series, "reba") if item.kind == "posture"]

    assert ranked[0] == "trunk"
    assert ranked.index("trunk") < ranked.index("left_elbow")


def test_drivers_are_ranked_by_points_and_carry_the_measured_angle():
    point = score_series(
        [
            record(
                1,
                0.0,
                components={"trunk": 5, "upper_arm": 6, "lower_arm": 2},
                angles={"trunk_flexion": 88.0, "left_upper_arm_flexion": 120.0},
            )
        ],
        "reba",
    )[0]
    drivers = drivers_at(point, "reba", limit=3)

    assert drivers[0].points_added >= drivers[-1].points_added
    trunk = next(item for item in drivers if item.region == "trunk")
    assert trunk.measurement == "trunk_flexion"
    assert trunk.value_deg == pytest.approx(88.0)
    assert trunk.score_text == "5 of 5"


def test_task_conditions_are_kept_separate_from_body_parts():
    """A load and a bent back both add points but call for different fixes."""

    series = score_series(
        [record(index, index * 0.1, components={"trunk": 4, "activity": 3}) for index in range(1, 11)],
        "reba",
    )
    loads = body_part_load(series, "reba")
    kinds = {item.region: item.kind for item in loads}

    assert kinds["trunk"] == "posture"
    assert kinds["activity"] == "task"
    assert all(spec.kind == "posture" for spec in components_for("reba")[:6])


# ----------------------------------------------------------------------
# Time, weighted by how long each sample stood for
# ----------------------------------------------------------------------


def test_each_sample_stands_for_the_interval_to_the_next_one():
    series = score_series(
        [record(1, 0.0), record(2, 0.1), record(3, 0.2)], "reba", maximum_gap_sec=0.5
    )

    assert [round(point.dwell_sec, 3) for point in series[:2]] == [0.1, 0.1]
    # The final sample takes the median dwell rather than zero, so a trial's last
    # observation is not silently worth no time at all.
    assert series[-1].dwell_sec == pytest.approx(0.1)
    assert scored_duration_sec(series) == pytest.approx(0.3)


def test_a_tracking_gap_is_not_credited_as_time_in_a_posture():
    series = score_series(
        [record(1, 0.0), record(2, 0.1), record(3, 9.0)], "reba", maximum_gap_sec=0.5
    )

    assert series[1].dwell_sec == 0.0
    assert scored_duration_sec(series) == pytest.approx(0.2)


def test_time_at_an_action_level_is_duration_weighted_not_a_sample_count():
    """Irregular sampling must not let a dense burst inflate a share."""

    series = score_series(
        [
            record(1, 0.0, components={"trunk": 5, "upper_arm": 6, "activity": 3}),
            record(2, 0.05, components={"trunk": 5, "upper_arm": 6, "activity": 3}),
            record(3, 0.10, components={"trunk": 5, "upper_arm": 6, "activity": 3}),
            record(4, 0.60, components={"trunk": 1}),
            record(5, 1.10, components={"trunk": 1}),
        ],
        "reba",
        maximum_gap_sec=0.75,
    )
    seconds, percent = time_at_or_above(series, 8)
    counts = sum(1 for point in series if point.score >= 8)

    assert counts == 3
    # Three of five samples, but they cover only a fraction of a second out of
    # more than one and a half.
    assert percent < 100 * counts / len(series)
    assert seconds == pytest.approx(0.6, abs=0.05)


def test_band_shares_cover_all_scored_time_exactly_once():
    series = score_series(
        [record(index, index * 0.1, components={"trunk": index % 5 + 1}) for index in range(1, 21)],
        "reba",
    )
    shares = band_shares(series, "reba")

    assert sum(share.duration_sec for share in shares) == pytest.approx(scored_duration_sec(series))
    assert sum(share.observation_count for share in shares) == len(series)
    assert sum(share.percent_of_scored_time for share in shares) == pytest.approx(100.0)


# ----------------------------------------------------------------------
# Episodes
# ----------------------------------------------------------------------


def _high_then_low(count_high: int = 20, count_low: int = 20):
    rows = [
        record(index, index * 0.1, components={"trunk": 5, "upper_arm": 6, "activity": 3})
        for index in range(count_high)
    ]
    rows += [
        record(count_high + index, (count_high + index) * 0.1, components={"trunk": 1})
        for index in range(count_low)
    ]
    return rows


def test_an_episode_lasts_the_sum_of_its_samples_not_first_to_last():
    """Taking last minus first makes a single-sample spike last zero seconds.

    A one-frame excursion to the top of the scale then drops out of the list
    entirely, which is the opposite of what a reviewer is looking for.
    """

    series = score_series(_high_then_low(count_high=10), "reba")
    episodes = risk_episodes(series, "reba", minimum_duration_sec=0.0)

    assert len(episodes) == 1
    span = episodes[0].end_sec - episodes[0].start_sec
    assert episodes[0].duration_sec == pytest.approx(1.0)
    assert episodes[0].duration_sec == pytest.approx(span)
    assert episodes[0].observation_count == 10


def test_episodes_split_across_a_tracking_gap():
    rows = [
        record(index, index * 0.1, components={"trunk": 5, "upper_arm": 6, "activity": 3})
        for index in range(8)
    ]
    rows += [
        record(20 + index, 9.0 + index * 0.1, components={"trunk": 5, "upper_arm": 6, "activity": 3})
        for index in range(8)
    ]
    series = score_series(rows, "reba", maximum_gap_sec=0.5)

    assert len(risk_episodes(series, "reba", minimum_duration_sec=0.0, maximum_gap_sec=0.5)) == 2


def test_episodes_are_returned_worst_first_and_numbered():
    # An earlier, milder episode and a later, worse one. Scores are 8 and 10.
    rows = [
        record(index, index * 0.1, components={"trunk": 4, "upper_arm": 5, "activity": 3})
        for index in range(10)
    ]
    rows += [record(20 + i, 5.0 + i * 0.1, components={"trunk": 1}) for i in range(5)]
    rows += [
        record(
            40 + index, 10.0 + index * 0.1, components={"trunk": 5, "upper_arm": 6, "activity": 3}
        )
        for index in range(10)
    ]
    series = score_series(rows, "reba", maximum_gap_sec=0.5)
    episodes = risk_episodes(series, "reba", minimum_duration_sec=0.0, maximum_gap_sec=0.5)

    assert len(episodes) == 2
    assert [item.index for item in episodes] == [1, 2]
    assert [item.peak_score for item in episodes] == [10, 8]
    # The worst one starts later; ranking by severity must beat ranking by time.
    assert episodes[0].start_sec > episodes[-1].start_sec


def test_an_episode_names_the_frame_a_reviewer_should_open():
    series = score_series(_high_then_low(), "reba")
    episode = risk_episodes(series, "reba", minimum_duration_sec=0.0)[0]

    assert episode.start_frame_number <= episode.peak_frame_number <= episode.end_frame_number
    assert episode.drivers
    assert episode.time_range_text.count("–") == 1


def test_a_short_spike_is_not_reported_as_a_sustained_episode():
    rows = [record(0, 0.0, components={"trunk": 5, "upper_arm": 6, "activity": 3})]
    rows += [record(index, index * 0.1, components={"trunk": 1}) for index in range(1, 20)]
    series = score_series(rows, "reba")

    assert risk_episodes(series, "reba", minimum_duration_sec=1.0) == []


# ----------------------------------------------------------------------
# The verdict
# ----------------------------------------------------------------------


def test_the_verdict_reports_both_the_peak_and_how_much_of_the_task_was_spent_there():
    """A peak alone cannot separate one awkward reach from the way the job runs."""

    series = score_series(_high_then_low(), "reba")
    result = verdict(series, "reba")

    assert result.scored
    assert result.peak_score >= 8
    assert 0 < result.percent_at_action < 100
    assert result.typical_score < result.peak_score
    assert str(result.peak_score) in result.headline
    assert "of the scored time" in result.detail
    assert result.leading_label


def test_the_headline_does_not_call_rulas_action_names_a_risk_level():
    """RULA's bands are instructions; REBA's are risk levels.

    Treating them alike produced "peaked at 7 — change now risk".
    """

    reba = verdict(score_series(_high_then_low(), "reba"), "reba")
    rula = verdict(
        score_series(
            [
                record(index, index * 0.1, method="rula", components={"trunk": 6, "neck": 6})
                for index in range(20)
            ],
            "rula",
        ),
        "rula",
    )

    assert "high risk" in reba.headline.lower()
    assert reba.peak_band is not None and reba.peak_band.describes_risk
    assert "risk" not in rula.headline.lower()
    assert rula.peak_band is not None and not rula.peak_band.describes_risk
    assert rula.peak_band.action in rula.headline


def test_an_unscored_selection_says_so_instead_of_reporting_a_zero():
    result = verdict([], "rula")

    assert not result.scored
    assert result.peak_score is None
    assert result.percent_at_action == 0.0
    assert "no valid score" in result.headline.lower()


# ----------------------------------------------------------------------
# Presentation helpers
# ----------------------------------------------------------------------


def test_time_is_shown_as_a_clock_because_that_is_how_video_is_scrubbed():
    assert clock(0) == "0:00"
    assert clock(44.4) == "0:44"
    assert clock(99.5) == "1:40"
    assert clock(None) == "—"
    assert clock(-1) == "—"


def test_reba_bands_use_rebas_own_wording_not_rulas():
    """A REBA 8 previously read as RULA's "investigate and change soon"."""

    assert risk_label(8, "reba") == "High"
    assert risk_action(8, "reba") == "Action is necessary soon"
    assert risk_label(13, "reba") == "Very high"
    assert risk_action(13, "reba") == "Action is necessary NOW"
    assert risk_label(1, "reba") == "Negligible"

    assert risk_label(2, "rula") == "Acceptable"
    assert risk_label(7, "rula") == "Change now"
    assert risk_action(7, "rula") == "Investigate and implement change immediately"


def test_a_score_past_the_top_of_a_scale_still_lands_in_the_worst_band():
    """An out-of-range value is a fault to report, not a reason to lose the peak."""

    assert risk_band(99, "reba").label == "Very high"
    assert risk_band(99, "rula").label == "Change now"
    assert risk_band(None, "reba") is None


def test_component_ranges_match_the_published_clamps():
    assert {spec.component: spec.maximum for spec in RULA_COMPONENTS if spec.kind == "posture"} == {
        "trunk": 6,
        "neck": 6,
        "legs": 2,
        "upper_arm": 6,
        "lower_arm": 3,
        "wrist": 4,
    }
    assert {spec.component: spec.maximum for spec in REBA_COMPONENTS if spec.kind == "posture"} == {
        "trunk": 5,
        "neck": 3,
        "legs": 4,
        "upper_arm": 6,
        "lower_arm": 2,
        "wrist": 3,
    }
