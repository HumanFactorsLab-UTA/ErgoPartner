"""The expert-rating paths must compose scores exactly as the calculators do.

Three artefacts turn a rater's posture judgements into a REBA/RULA score: this
package's :mod:`validation.scoring`, the JavaScript in the browser rating page, and
the formulas in the Excel workbook.  If any of them drifts from
``ergonomics.reba`` / ``ergonomics.rula``, the study measures the drift instead of
the pipeline, and it would do so silently.

These tests pin the two that can be checked without a browser or Excel: the Python
composer, and the tables and base scores the other two are generated from.  The
browser and Excel layers are generated from the same source at build time, so
covering the generator covers them.
"""

from __future__ import annotations

import itertools

import pytest

from ergonomics.reba import REBACalculator
from ergonomics.rula import RULACalculator
from validation import scoring

REBA = REBACalculator()
RULA = RULACalculator()


def _every_band_combination(fields: dict[str, tuple[str, ...]], flags: tuple[str, ...]):
    """Cartesian product of the band choices and modifier flags for one segment."""

    names = list(fields) + list(flags)
    domains = [fields[name] for name in fields] + [(False, True)] * len(flags)
    for values in itertools.product(*domains):
        yield dict(zip(names, values, strict=True))


# ---------------------------------------------------------------------------
# Component derivation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("obs", _every_band_combination(
    {"trunk": tuple(scoring.TRUNK_BANDS)}, ("trunk_twisted", "trunk_side_bent")))
def test_trunk_matches_calculators(obs):
    degrees = scoring.TRUNK_BANDS[obs["trunk"]]
    assert scoring.reba_components(obs)["trunk"] == REBA.score_trunk(
        degrees, obs["trunk_twisted"] or obs["trunk_side_bent"]
    )
    # RULA 1993 defines no trunk-extension bin; the pipeline calls that input invalid.
    if degrees < 0:
        assert scoring.rula_components(obs)["trunk"] is None
    else:
        assert scoring.rula_components(obs)["trunk"] == RULA.score_trunk(
            degrees, obs["trunk_twisted"], obs["trunk_side_bent"]
        )


@pytest.mark.parametrize("obs", _every_band_combination(
    {"neck": tuple(scoring.NECK_BANDS)}, ("neck_twisted", "neck_side_bent")))
def test_neck_matches_calculators(obs):
    degrees = scoring.NECK_BANDS[obs["neck"]]
    assert scoring.reba_components(obs)["neck"] == REBA.score_neck(
        degrees, obs["neck_twisted"] or obs["neck_side_bent"]
    )
    assert scoring.rula_components(obs)["neck"] == RULA.score_neck(
        degrees, obs["neck_twisted"], obs["neck_side_bent"]
    )


@pytest.mark.parametrize("obs", _every_band_combination(
    {"legs": scoring.LEGS_BANDS, "knee": tuple(scoring.KNEE_BANDS)},
    ("seated", "legs_supported_balanced")))
def test_legs_matches_calculators(obs):
    expected = REBA.score_legs(
        obs["legs"], 0.0 if obs["seated"] else scoring.KNEE_BANDS[obs["knee"]], obs["seated"]
    )
    assert scoring.reba_components(obs)["legs"] == expected
    assert scoring.rula_components(obs)["legs"] == (1 if obs["legs_supported_balanced"] else 2)


@pytest.mark.parametrize("obs", _every_band_combination(
    {"upper_arm": tuple(scoring.UPPER_ARM_BANDS)},
    ("arm_abducted", "shoulder_raised", "arm_supported")))
def test_upper_arm_matches_calculators(obs):
    degrees = scoring.UPPER_ARM_BANDS[obs["upper_arm"]]
    expected = REBA.score_upper_arm(
        degrees, obs["arm_abducted"], obs["shoulder_raised"], obs["arm_supported"]
    )
    assert scoring.reba_components(obs)["upper_arm"] == expected
    # Both instruments use the same upper-arm rule; a divergence would be a real change.
    assert scoring.rula_components(obs)["upper_arm"] == RULA.score_upper_arm(
        degrees, obs["arm_abducted"], obs["shoulder_raised"], obs["arm_supported"]
    )


@pytest.mark.parametrize("obs", _every_band_combination(
    {"lower_arm": tuple(scoring.LOWER_ARM_BANDS)}, ("lower_arm_across",)))
def test_lower_arm_matches_calculators(obs):
    degrees = scoring.LOWER_ARM_BANDS[obs["lower_arm"]]
    assert scoring.reba_components(obs)["lower_arm"] == REBA.score_lower_arm(degrees)
    assert scoring.rula_components(obs)["lower_arm"] == RULA.score_lower_arm(
        degrees, obs["lower_arm_across"]
    )


@pytest.mark.parametrize("obs", _every_band_combination(
    {"wrist": tuple(scoring.WRIST_BANDS)}, ("wrist_deviated", "wrist_twist_end")))
def test_wrist_matches_calculators(obs):
    degrees = scoring.WRIST_BANDS[obs["wrist"]]
    assert scoring.reba_components(obs)["wrist"] == REBA.score_wrist(
        degrees, obs["wrist_deviated"] or obs["wrist_twist_end"]
    )
    rula = scoring.rula_components(obs)
    assert rula["wrist"] == RULA.score_wrist(degrees, obs["wrist_deviated"])
    assert rula["wrist_twist"] == (2 if obs["wrist_twist_end"] else 1)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def _complete_observation(**overrides):
    obs = {
        "trunk": "flex_20_60", "trunk_twisted": False, "trunk_side_bent": False,
        "neck": "flex_10_20", "neck_twisted": False, "neck_side_bent": False,
        "legs": "bilateral", "seated": False, "legs_supported_balanced": True, "knee": "k30_60",
        "upper_arm": "f45_90", "arm_abducted": False, "shoulder_raised": False,
        "arm_supported": False,
        "lower_arm": "f60_100", "lower_arm_across": False,
        "wrist": "le15", "wrist_deviated": False, "wrist_twist_end": False,
    }
    obs.update(overrides)
    return obs


NO_GIVEN = dict.fromkeys(scoring.GIVEN_FIELDS, 0)


def test_reba_composition_follows_the_calculator_tables():
    obs = _complete_observation()
    result = scoring.compose_reba(obs, NO_GIVEN)
    parts = scoring.reba_components(obs)
    table_a = REBA.lookup_table_a(parts["trunk"], parts["neck"], parts["legs"])
    table_b = REBA.lookup_table_b(parts["upper_arm"], parts["lower_arm"], parts["wrist"])
    assert result["table_a"] == table_a
    assert result["table_b"] == table_b
    assert result["final"] == REBA.lookup_table_c(table_a, table_b)
    assert result["risk"] == REBA.RISK_LEVELS[result["final"]][0]


def test_rula_composition_follows_the_calculator_tables():
    obs = _complete_observation()
    result = scoring.compose_rula(obs, NO_GIVEN)
    parts = scoring.rula_components(obs)
    table_a = RULA.lookup_table_a(
        parts["upper_arm"], parts["lower_arm"], parts["wrist"], parts["wrist_twist"]
    )
    table_b = RULA.lookup_table_b(parts["neck"], parts["trunk"], parts["legs"])
    assert result["final"] == RULA.lookup_table_c(table_a, table_b)


@pytest.mark.parametrize("field,value", [
    ("reba_load_force", 2), ("reba_coupling", 3), ("reba_activity", 1),
])
def test_given_subscores_move_the_reba_total(field, value):
    """A rater cannot judge load, coupling or activity, so they are supplied per frame."""

    obs = _complete_observation()
    base = scoring.compose_reba(obs, NO_GIVEN)
    raised = scoring.compose_reba(obs, {**NO_GIVEN, field: value})
    assert raised["final"] >= base["final"]


def test_rula_total_is_none_for_trunk_extension_but_reba_still_scores():
    obs = _complete_observation(trunk="ext_gt20")
    assert scoring.compose_rula(obs, NO_GIVEN)["final"] is None
    assert scoring.compose_reba(obs, NO_GIVEN)["final"] is not None


def test_incomplete_observation_yields_no_score():
    obs = _complete_observation()
    del obs["wrist"]
    assert scoring.compose_reba(obs, NO_GIVEN)["final"] is None
    assert not scoring.is_complete(obs)


def test_knee_is_only_required_when_standing():
    seated = _complete_observation(seated=True)
    del seated["knee"]
    assert scoring.is_complete(seated)
    assert scoring.compose_reba(seated, NO_GIVEN)["final"] is not None

    standing = _complete_observation(seated=False)
    del standing["knee"]
    assert not scoring.is_complete(standing)


def test_yes_no_strings_are_accepted_as_flags():
    """The workbook stores modifiers as the words a rater picked, not as booleans."""

    typed = _complete_observation(trunk_twisted="Yes", neck_side_bent="No")
    boolean = _complete_observation(trunk_twisted=True, neck_side_bent=False)
    assert scoring.compose_reba(typed, NO_GIVEN) == scoring.compose_reba(boolean, NO_GIVEN)


def test_reba_final_stays_inside_the_defined_range():
    """Every band/modifier/given combination must land on a defined risk level."""

    for trunk in scoring.TRUNK_BANDS:
        for upper in scoring.UPPER_ARM_BANDS:
            obs = _complete_observation(
                trunk=trunk, upper_arm=upper, trunk_twisted=True,
                arm_abducted=True, shoulder_raised=True, legs="unilateral", knee="gt60",
            )
            given = {"reba_load_force": 3, "reba_coupling": 3, "reba_activity": 3,
                     "rula_force_load": 3, "rula_muscle_use": 1}
            final = scoring.compose_reba(obs, given)["final"]
            assert final in REBA.RISK_LEVELS
