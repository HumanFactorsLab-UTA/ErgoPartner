"""Exhaustive golden checks for every literal method-table cell.

The digests freeze the primary-source transcriptions independently of individual
worked examples. The lookup sweeps then verify indexing/order for every cell.
"""

import hashlib
import json

from ergonomics.niosh import (
    CM_TABLE,
    FM_TABLE,
    NIOSHCalculator,
    NIOSHInput,
    NIOSHTaskApplicability,
)
from ergonomics.reba import REBACalculator
from ergonomics.rula import RULACalculator

EXPECTED_DIGESTS = {
    "rula_a": "102a59eb4a1ac9992799d4fdcd20d70b2a89268eb111ff5477f95bbab8dd0226",
    "rula_b": "96624e5fba60b33a02076c6798a682940c11009f8f586df719989c2d9d75a0b8",
    "rula_c": "94a180b02f20ba19cfaee779644b28c76b7e1afef96ee15f36c374c9087681a1",
    "reba_a": "c09a3778083138d7d726da93f46005c9bae316a85366d2ffb08265a33e38a328",
    "reba_b": "a4aa37f34b70de7976806fabc7a0f452c677a4720087a8737d76a61ae52f6832",
    "reba_c": "ea99583e0f40a2b422b9e956436770d7f844fccb9fbd61697954d7efab782f29",
    "rnle_fm": "00c9aa8e455200a0f9976e89737fdb88453dbdc79203e594c954c393b7b6f183",
    "rnle_cm": "0359b3cc26638cde1acec9d3a9a333b2f92008f4e8a658647f9d66d5e16de9c6",
}


def _digest(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_primary_source_table_transcriptions_have_golden_digests():
    actual = {
        "rula_a": _digest(RULACalculator.TABLE_A),
        "rula_b": _digest(RULACalculator.TABLE_B),
        "rula_c": _digest(RULACalculator.TABLE_C),
        "reba_a": _digest(REBACalculator.TABLE_A),
        "reba_b": _digest(REBACalculator.TABLE_B),
        "reba_c": _digest(REBACalculator.TABLE_C),
        "rnle_fm": _digest(FM_TABLE),
        "rnle_cm": _digest(CM_TABLE),
    }
    assert actual == EXPECTED_DIGESTS


def test_every_rula_table_lookup_cell():
    calc = RULACalculator()
    for upper_index, lower_rows in enumerate(calc.TABLE_A, start=1):
        for lower_index, wrist_rows in enumerate(lower_rows, start=1):
            for wrist_index, twist_cells in enumerate(wrist_rows, start=1):
                for twist_index, expected in enumerate(twist_cells, start=1):
                    assert (
                        calc.lookup_table_a(upper_index, lower_index, wrist_index, twist_index)
                        == expected
                    )
    for neck_index, trunk_rows in enumerate(calc.TABLE_B, start=1):
        for trunk_index, leg_cells in enumerate(trunk_rows, start=1):
            for leg_index, expected in enumerate(leg_cells, start=1):
                assert calc.lookup_table_b(neck_index, trunk_index, leg_index) == expected
    for score_a, score_b_cells in enumerate(calc.TABLE_C, start=1):
        for score_b, expected in enumerate(score_b_cells, start=1):
            assert calc.lookup_table_c(score_a, score_b) == expected
    assert calc.lookup_table_c(99, 99) == calc.TABLE_C[-1][-1]


def test_every_reba_table_lookup_cell():
    calc = REBACalculator()
    for trunk_index, neck_rows in enumerate(calc.TABLE_A, start=1):
        for neck_index, leg_cells in enumerate(neck_rows, start=1):
            for leg_index, expected in enumerate(leg_cells, start=1):
                assert calc.lookup_table_a(trunk_index, neck_index, leg_index) == expected
    for upper_index, lower_rows in enumerate(calc.TABLE_B, start=1):
        for lower_index, wrist_cells in enumerate(lower_rows, start=1):
            for wrist_index, expected in enumerate(wrist_cells, start=1):
                assert calc.lookup_table_b(upper_index, lower_index, wrist_index) == expected
    for score_a, score_b_cells in enumerate(calc.TABLE_C, start=1):
        for score_b, expected in enumerate(score_b_cells, start=1):
            assert calc.lookup_table_c(score_a, score_b) == expected


def _rnle_input(*, frequency: float, duration: str, vertical: float, coupling: str):
    return NIOSHInput(
        horizontal_cm=25,
        vertical_cm=vertical,
        travel_distance_cm=25,
        asymmetry_deg=0,
        frequency_lifts_per_min=frequency,
        duration=duration,
        coupling=coupling,
        load_weight_kg=1,
        applicability=NIOSHTaskApplicability.reviewed_applicable(),
    )


def test_every_rnle_frequency_table_cell_through_calculator():
    calc = NIOSHCalculator()
    for frequency, duration_values in FM_TABLE.items():
        for duration in ("short", "moderate", "long"):
            for vertical_index, vertical in enumerate((50.0, 75.0)):
                result = calc.calculate(
                    _rnle_input(
                        frequency=frequency,
                        duration=duration,
                        vertical=vertical,
                        coupling="good",
                    )
                )
                assert result.multipliers["FM"] == duration_values[duration][vertical_index]


def test_every_rnle_coupling_table_cell_through_calculator():
    calc = NIOSHCalculator()
    for coupling, expected_cells in CM_TABLE.items():
        for vertical_index, vertical in enumerate((50.0, 75.0)):
            result = calc.calculate(
                _rnle_input(
                    frequency=0.2,
                    duration="short",
                    vertical=vertical,
                    coupling=coupling,
                )
            )
            assert result.multipliers["CM"] == expected_cells[vertical_index]
