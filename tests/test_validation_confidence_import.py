"""Typed confidence labels use the scale printed in the returned workbook."""
import pytest

from validation.build_workbook import _read_confidence


@pytest.mark.parametrize("value, expected", [
    (None, ""), ("", ""), (3, "3"), ("3 — Moderate", "3"),
    ("moderate", "3"), ("  VERY CONFIDENT ", "5"),
    ("Confident", "4"), ("Low", "2"), ("Very low", "1"),
    ("unsure", "unsure"), ("0", "0"),
])
def test_confidence_labels_map_without_guessing_unknown_text(value, expected):
    assert _read_confidence(value) == expected
