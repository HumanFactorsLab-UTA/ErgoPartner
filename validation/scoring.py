"""Compose REBA and RULA from a rater's postural observations.

This is the single source of truth for turning what a rater observed into scores, and
it is used for every return path -- the web app's JSON and the Excel workbook alike --
so the two cannot disagree about what a given set of observations means.

Nothing here transcribes a threshold or a table.  Each posture band is mapped to a
representative angle inside it and handed to the project's own
:class:`~ergonomics.reba.REBACalculator` / :class:`~ergonomics.rula.RULACalculator`
scoring functions, and composition goes through their lookup tables.  A change to the
calculators therefore changes this module's output too, which is the point: expert and
pipeline scores must be composed by the same code for their difference to mean
anything.

The subscores a rater cannot judge from a still frame -- load, coupling, activity,
muscle use -- are supplied per frame as ``given`` rather than observed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ergonomics.reba import REBACalculator
from ergonomics.rula import RULACalculator

_REBA = REBACalculator()
_RULA = RULACalculator()

#: Posture bands offered to the rater, and a representative angle inside each.  The
#: bands are the union of REBA's and RULA's cut points, so one answer serves both.
TRUNK_BANDS: dict[str, float] = {
    "upright": 0.0,
    "flex_0_20": 10.0,
    "flex_20_60": 40.0,
    "flex_gt60": 75.0,
    "ext_0_20": -10.0,
    "ext_gt20": -30.0,
}
NECK_BANDS: dict[str, float] = {
    "flex_0_10": 5.0,
    "flex_10_20": 15.0,
    "flex_gt20": 30.0,
    "ext": -10.0,
}
KNEE_BANDS: dict[str, float] = {"lt30": 10.0, "k30_60": 45.0, "gt60": 80.0}
UPPER_ARM_BANDS: dict[str, float] = {
    "n20_20": 0.0,
    "mid": 30.0,
    "f45_90": 70.0,
    "f_gt90": 110.0,
}
LOWER_ARM_BANDS: dict[str, float] = {"f60_100": 80.0, "outside": 30.0}
WRIST_BANDS: dict[str, float] = {"neutral": 0.0, "le15": 10.0, "gt15": 25.0}

LEGS_BANDS = ("bilateral", "unilateral")

#: Every field a completed observation carries.  Radio-style fields have no default;
#: the modifiers default to False so an untouched checkbox is still a recorded answer.
OBSERVATION_CHOICES: dict[str, tuple[str, ...]] = {
    "trunk": tuple(TRUNK_BANDS),
    "neck": tuple(NECK_BANDS),
    "legs": LEGS_BANDS,
    "knee": tuple(KNEE_BANDS),
    "upper_arm": tuple(UPPER_ARM_BANDS),
    "lower_arm": tuple(LOWER_ARM_BANDS),
    "wrist": tuple(WRIST_BANDS),
}
MODIFIER_FIELDS: tuple[str, ...] = (
    "trunk_twisted",
    "trunk_side_bent",
    "neck_twisted",
    "neck_side_bent",
    "seated",
    "legs_supported_balanced",
    "arm_abducted",
    "shoulder_raised",
    "arm_supported",
    "lower_arm_across",
    "wrist_deviated",
    "wrist_twist_end",
)

REBA_COMPONENTS = ("trunk", "neck", "legs", "upper_arm", "lower_arm", "wrist")
RULA_COMPONENTS = ("upper_arm", "lower_arm", "wrist", "wrist_twist", "neck", "trunk", "legs")

GIVEN_FIELDS = (
    "reba_load_force",
    "reba_coupling",
    "reba_activity",
    "rula_force_load",
    "rula_muscle_use",
)


def _flag(obs: Mapping[str, Any], name: str) -> bool:
    value = obs.get(name)
    if isinstance(value, str):
        return value.strip().lower() in {"yes", "true", "y", "1"}
    return bool(value)


def _band(obs: Mapping[str, Any], name: str, table: Mapping[str, float]) -> float | None:
    value = obs.get(name)
    if value is None:
        return None
    return table.get(str(value).strip())


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


def reba_components(obs: Mapping[str, Any]) -> dict[str, int | None]:
    """REBA posture subscores, or ``None`` for any segment not yet answered."""

    trunk = _band(obs, "trunk", TRUNK_BANDS)
    neck = _band(obs, "neck", NECK_BANDS)
    upper = _band(obs, "upper_arm", UPPER_ARM_BANDS)
    lower = _band(obs, "lower_arm", LOWER_ARM_BANDS)
    wrist = _band(obs, "wrist", WRIST_BANDS)

    legs_choice = obs.get("legs")
    seated = _flag(obs, "seated")
    knee = _band(obs, "knee", KNEE_BANDS)
    legs_score: int | None = None
    if legs_choice in LEGS_BANDS:
        # Knee angle is only consulted when standing, so a seated rater may leave it blank.
        if seated:
            legs_score = _REBA.score_legs(str(legs_choice), 0.0, True)
        elif knee is not None:
            legs_score = _REBA.score_legs(str(legs_choice), knee, False)

    return {
        "trunk": None if trunk is None
        else _REBA.score_trunk(trunk, _flag(obs, "trunk_twisted") or _flag(obs, "trunk_side_bent")),
        "neck": None if neck is None
        else _REBA.score_neck(neck, _flag(obs, "neck_twisted") or _flag(obs, "neck_side_bent")),
        "legs": legs_score,
        "upper_arm": None if upper is None
        else _REBA.score_upper_arm(
            upper, _flag(obs, "arm_abducted"), _flag(obs, "shoulder_raised"), _flag(obs, "arm_supported")
        ),
        "lower_arm": None if lower is None else _REBA.score_lower_arm(lower),
        "wrist": None if wrist is None
        else _REBA.score_wrist(wrist, _flag(obs, "wrist_deviated") or _flag(obs, "wrist_twist_end")),
    }


def rula_components(obs: Mapping[str, Any]) -> dict[str, int | None]:
    """RULA posture subscores.

    ``trunk`` is ``None`` for any trunk extension: RULA 1993 defines no extension bin,
    and the pipeline treats that input as invalid rather than guessing a score.  The
    rest of the assessment is unaffected.
    """

    trunk_band = obs.get("trunk")
    trunk_deg = _band(obs, "trunk", TRUNK_BANDS)
    neck = _band(obs, "neck", NECK_BANDS)
    upper = _band(obs, "upper_arm", UPPER_ARM_BANDS)
    lower = _band(obs, "lower_arm", LOWER_ARM_BANDS)
    wrist = _band(obs, "wrist", WRIST_BANDS)

    trunk_score: int | None = None
    if trunk_deg is not None and trunk_deg >= 0:
        trunk_score = _RULA.score_trunk(
            trunk_deg, _flag(obs, "trunk_twisted"), _flag(obs, "trunk_side_bent")
        )

    return {
        "upper_arm": None if upper is None
        else _RULA.score_upper_arm(
            upper, _flag(obs, "arm_abducted"), _flag(obs, "shoulder_raised"), _flag(obs, "arm_supported")
        ),
        "lower_arm": None if lower is None
        else _RULA.score_lower_arm(lower, _flag(obs, "lower_arm_across")),
        "wrist": None if wrist is None else _RULA.score_wrist(wrist, _flag(obs, "wrist_deviated")),
        "wrist_twist": 2 if _flag(obs, "wrist_twist_end") else 1,
        "neck": None if neck is None
        else _RULA.score_neck(neck, _flag(obs, "neck_twisted"), _flag(obs, "neck_side_bent")),
        "trunk": trunk_score,
        "legs": None if obs.get("legs") not in LEGS_BANDS
        else (1 if _flag(obs, "legs_supported_balanced") else 2),
        "_trunk_band": trunk_band,  # retained for callers that need to explain a None
    }


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def _given(given: Mapping[str, Any], name: str) -> int:
    value = given.get(name, 0)
    try:
        return 0 if value is None else int(value)
    except (TypeError, ValueError):
        return 0


def compose_reba(obs: Mapping[str, Any], given: Mapping[str, Any]) -> dict[str, Any]:
    """Full REBA from observations plus the given load/coupling/activity subscores."""

    components = reba_components(obs)
    if any(value is None for value in components.values()):
        return {"components": components, "final": None, "risk": None}
    scored = {name: int(value) for name, value in components.items() if value is not None}

    table_a = _REBA.lookup_table_a(scored["trunk"], scored["neck"], scored["legs"])
    score_a = table_a + _given(given, "reba_load_force")
    table_b = _REBA.lookup_table_b(scored["upper_arm"], scored["lower_arm"], scored["wrist"])
    score_b = table_b + _given(given, "reba_coupling")
    table_c = _REBA.lookup_table_c(min(max(score_a, 1), 12), min(max(score_b, 1), 12))
    final = min(15, max(1, table_c + _given(given, "reba_activity")))
    return {
        "components": components,
        "table_a": table_a,
        "score_a": score_a,
        "table_b": table_b,
        "score_b": score_b,
        "table_c": table_c,
        "final": final,
        "risk": _REBA.RISK_LEVELS[final][0],
    }


def compose_rula(obs: Mapping[str, Any], given: Mapping[str, Any]) -> dict[str, Any]:
    """Full RULA; ``final`` is ``None`` when a component is unanswered or undefined."""

    components = {k: v for k, v in rula_components(obs).items() if not k.startswith("_")}
    if any(value is None for value in components.values()):
        return {"components": components, "final": None, "risk": None}
    scored = {name: int(value) for name, value in components.items() if value is not None}

    muscle = _given(given, "rula_muscle_use")
    force = _given(given, "rula_force_load")
    table_a = _RULA.lookup_table_a(
        scored["upper_arm"], scored["lower_arm"], scored["wrist"], scored["wrist_twist"]
    )
    score_a = table_a + muscle + force
    table_b = _RULA.lookup_table_b(scored["neck"], scored["trunk"], scored["legs"])
    score_b = table_b + muscle + force
    final = _RULA.lookup_table_c(score_a, score_b)
    return {
        "components": components,
        "table_a": table_a,
        "score_a": score_a,
        "table_b": table_b,
        "score_b": score_b,
        "final": final,
        "risk": _RULA.RISK_LEVELS[final][0],
    }


def is_complete(obs: Mapping[str, Any]) -> bool:
    """True when every segment needed to compose a REBA score has been answered."""

    for field in ("trunk", "neck", "legs", "upper_arm", "lower_arm", "wrist"):
        if obs.get(field) not in OBSERVATION_CHOICES[field]:
            return False
    if not _flag(obs, "seated") and obs.get("knee") not in OBSERVATION_CHOICES["knee"]:
        return False
    return True
