"""RNLE multiplier, validity, unit, and worked-case tests."""

import math

import pytest

from ergonomics.niosh import NIOSHCalculator, NIOSHInput, NIOSHTaskApplicability


def ideal_input(**changes) -> NIOSHInput:
    values = {
        "horizontal_cm": 25.0,
        "vertical_cm": 75.0,
        "travel_distance_cm": 25.0,
        "asymmetry_deg": 0.0,
        "frequency_lifts_per_min": 0.2,
        "duration": "short",
        "coupling": "good",
        "load_weight_kg": 23.0,
        "applicability": NIOSHTaskApplicability.reviewed_applicable(),
    }
    values.update(changes)
    return NIOSHInput(**values)


def test_ideal_rnle_case():
    result = NIOSHCalculator().calculate(ideal_input())
    assert result.validity_status == "valid"
    assert result.recommended_weight_limit_kg == pytest.approx(23.0)
    assert result.lifting_index == pytest.approx(1.0)
    assert result.multipliers == {
        "LC": 23.0,
        "HM": 1.0,
        "VM": 1.0,
        "DM": 1.0,
        "AM": 1.0,
        "FM": 1.0,
        "CM": 1.0,
    }


@pytest.mark.parametrize(
    ("load_kg", "expected"),
    [(22.9999, "At or below RWL"), (23.0, "At or below RWL"), (23.0001, "Exceeds RWL")],
)
def test_lifting_index_one_decision_boundary(load_kg, expected):
    result = NIOSHCalculator().calculate(ideal_input(load_weight_kg=load_kg))
    assert result.risk_level == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("horizontal_cm", 63.0001),
        ("vertical_cm", 175.0001),
        ("travel_distance_cm", 175.0001),
        ("asymmetry_deg", 135.0001),
        ("frequency_lifts_per_min", 15.0001),
    ],
)
def test_out_of_range_inputs_are_not_clamped(field, value):
    result = NIOSHCalculator().calculate(ideal_input(**{field: value}))
    assert result.recommended_weight_limit_kg is None
    assert result.validity_status == "unsupported_rnle_task"
    assert field in result.invalid_inputs


def test_h_and_d_below_minimum_use_method_defined_floor():
    result = NIOSHCalculator().calculate(ideal_input(horizontal_cm=20.0, travel_distance_cm=10.0))
    assert result.validity_status == "valid_with_limitations"
    assert result.multipliers["HM"] == 1.0
    assert result.multipliers["DM"] == 1.0
    assert any("25 cm" in warning for warning in result.warnings)


def test_missing_video_invisible_inputs_do_not_receive_defaults():
    result = NIOSHCalculator().calculate(
        NIOSHInput(
            horizontal_cm=30,
            vertical_cm=75,
            travel_distance_cm=50,
            asymmetry_deg=0,
            frequency_lifts_per_min=1,
        )
    )
    assert result.recommended_weight_limit_kg is None
    assert result.lifting_index is None
    assert set(result.missing_inputs) >= {"duration", "coupling", "load_weight_kg"}
    assert result.validity_status == "missing_required_task_input"


def test_unsupported_one_handed_lift_is_flagged():
    applicability = NIOSHTaskApplicability.reviewed_applicable()
    applicability = NIOSHTaskApplicability(**{**applicability.__dict__, "one_handed": True})
    result = NIOSHCalculator().calculate(ideal_input(applicability=applicability))
    assert result.validity_status == "unsupported_rnle_task"
    assert "one_handed" in result.unsupported_conditions


def test_multi_task_job_is_explicitly_unsupported():
    applicability = NIOSHTaskApplicability.reviewed_applicable()
    applicability = NIOSHTaskApplicability(**{**applicability.__dict__, "multiple_task_job": True})
    result = NIOSHCalculator().calculate(ideal_input(applicability=applicability))
    assert result.validity_status == "unsupported_rnle_task"
    assert "multiple_task_job" in result.unsupported_conditions


def test_frequency_multiplier_interpolates_as_manual_allows():
    result = NIOSHCalculator().calculate(ideal_input(frequency_lifts_per_min=1.5))
    assert result.multipliers["FM"] == pytest.approx((0.94 + 0.91) / 2)


def test_unreviewed_or_non_boolean_applicability_is_not_assumed():
    unreviewed = NIOSHCalculator().calculate(ideal_input(applicability=NIOSHTaskApplicability()))
    assert unreviewed.validity_status == "missing_required_task_input"
    assert "applicability.one_handed" in unreviewed.missing_inputs

    invalid = NIOSHCalculator().calculate(
        ideal_input(
            applicability=NIOSHTaskApplicability(
                **{
                    **NIOSHTaskApplicability.reviewed_applicable().__dict__,
                    "one_handed": "no",
                }
            )
        )
    )
    assert invalid.validity_status == "unsupported_rnle_task"
    assert "applicability.one_handed" in invalid.invalid_inputs


def test_multiplier_formulas_at_nonideal_values():
    result = NIOSHCalculator().calculate(
        ideal_input(
            horizontal_cm=50,
            vertical_cm=100,
            travel_distance_cm=50,
            asymmetry_deg=90,
            coupling="fair",
        )
    )
    assert result.multipliers["HM"] == pytest.approx(0.5)
    assert result.multipliers["VM"] == pytest.approx(0.925)
    assert result.multipliers["DM"] == pytest.approx(0.91)
    assert result.multipliers["AM"] == pytest.approx(0.712)
    assert result.multipliers["CM"] == pytest.approx(1.0)
    product = math.prod(result.multipliers.values())
    assert result.recommended_weight_limit_kg == pytest.approx(product)


def test_exact_upper_validity_limits_are_accepted():
    result = NIOSHCalculator().calculate(
        ideal_input(
            horizontal_cm=63,
            vertical_cm=175,
            travel_distance_cm=175,
            asymmetry_deg=135,
            frequency_lifts_per_min=15,
        )
    )
    assert result.validity_status == "valid"
    assert result.recommended_weight_limit_kg is not None


def test_imperial_conversion_is_exact_and_auditable():
    inp = NIOSHInput.from_imperial(
        horizontal_in=10,
        vertical_in=30,
        travel_distance_in=10,
        asymmetry_deg=0,
        frequency_lifts_per_min=0.2,
        duration="short",
        coupling="good",
        load_weight_lb=51,
        applicability=NIOSHTaskApplicability(),
    )
    assert inp.horizontal_cm == pytest.approx(25.4)
    assert inp.vertical_cm == pytest.approx(76.2)
    assert inp.load_weight_kg == pytest.approx(23.13321087)


def test_destination_is_required_for_significant_control():
    result = NIOSHCalculator().calculate_lift(ideal_input(), significant_destination_control=True)
    assert result.recommended_weight_limit_kg is None
    assert result.validity_status == "missing_required_task_input"


def test_complete_origin_destination_lift_uses_lower_rwl():
    origin = ideal_input(location="origin")
    destination = ideal_input(
        location="destination",
        horizontal_cm=50,
        vertical_cm=125,
        travel_distance_cm=100,
    )
    result = NIOSHCalculator().calculate_lift(
        origin,
        destination,
        significant_destination_control=True,
    )
    assert result.origin.recommended_weight_limit_kg == pytest.approx(23.0)
    assert result.destination.recommended_weight_limit_kg < 23.0
    assert result.controlling_location == "destination"
    assert result.recommended_weight_limit_kg == result.destination.recommended_weight_limit_kg
    assert result.lifting_index == result.destination.lifting_index


def test_non_positive_load_is_invalid():
    result = NIOSHCalculator().calculate(ideal_input(load_weight_kg=0.0))
    assert result.recommended_weight_limit_kg is None
    assert "load_weight_kg" in result.invalid_inputs
    assert not math.isinf(result.lifting_index or 0.0)
