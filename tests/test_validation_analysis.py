"""Agreement statistics, and the claim that the two rating paths are interchangeable.

The headline promise of the package is that a rater's answers mean the same thing
whether they came from the browser page or the Excel workbook.  That is only true if
the analysis recomposes both from the raw observations, so it is tested directly:
the same judgements, delivered both ways, must reach the report identically.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from validation import analyze_ratings as analysis
from validation import scoring

BANDS = ("Low", "Medium", "High")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_kappa_is_one_when_the_raters_never_differ():
    left = pd.Series(["Low", "Medium", "High"] * 5)
    assert analysis.cohen_kappa(left, left.copy(), BANDS) == pytest.approx(1.0)


def test_kappa_is_about_zero_when_agreement_is_only_what_chance_gives():
    rng = np.random.default_rng(0)
    left = pd.Series(rng.choice(BANDS, 4000))
    right = pd.Series(rng.choice(BANDS, 4000))
    assert abs(analysis.cohen_kappa(left, right, BANDS)) < 0.06


def test_kappa_is_negative_when_the_raters_disagree_systematically():
    left = pd.Series(["Low"] * 10 + ["High"] * 10)
    right = pd.Series(["High"] * 10 + ["Low"] * 10)
    assert analysis.cohen_kappa(left, right, BANDS) < 0


def test_weighting_changes_kappa_because_the_design_is_not_the_population():
    """A balanced sample over-represents rare bands; weights undo that."""

    expert = pd.Series(["Low"] * 10 + ["Medium"] * 10)
    pipeline = pd.Series(["Low"] * 8 + ["Medium"] * 2 + ["Medium"] * 10)
    flat = pd.Series([1.0] * 20)
    tilted = pd.Series([0.1] * 10 + [5.0] * 10)   # Medium far commoner in the population
    assert analysis.cohen_kappa(expert, pipeline, BANDS, flat) != pytest.approx(
        analysis.cohen_kappa(expert, pipeline, BANDS, tilted)
    )


def test_kappa_ignores_rows_where_either_side_is_missing():
    left = pd.Series(["Low", "Medium", None, "High"])
    right = pd.Series(["Low", "Medium", "High", None])
    assert analysis.cohen_kappa(left, right, BANDS) == pytest.approx(1.0)


def test_quadratic_weighted_kappa_rewards_near_misses_over_far_ones():
    truth = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
    near = truth + 1
    far = pd.Series([8, 7, 6, 5, 4, 3, 2, 1])
    assert analysis.quadratic_weighted_kappa(truth, truth.copy()) == pytest.approx(1.0)
    assert analysis.quadratic_weighted_kappa(truth, near) > analysis.quadratic_weighted_kappa(
        truth, far
    )


def test_fleiss_kappa_reproduces_the_published_worked_example():
    """Fleiss (1971), 10 subjects x 5 raters x 3 categories."""

    counts = np.array([
        [5, 0, 0], [0, 5, 0], [0, 0, 5], [4, 1, 0], [3, 2, 0],
        [0, 4, 1], [1, 0, 4], [0, 3, 2], [5, 0, 0], [0, 0, 5],
    ])
    ratings = np.array(
        [[c for c, n in enumerate(row) for _ in range(n)] for row in counts], dtype=float
    )
    raters, items = 5, len(counts)
    proportions = counts.sum(axis=0) / (items * raters)
    agreement = ((counts**2).sum(axis=1) - raters) / (raters * (raters - 1))
    expected = (agreement.mean() - (proportions**2).sum()) / (1 - (proportions**2).sum())
    assert analysis.fleiss_kappa(ratings) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("statistic", [analysis.fleiss_kappa, analysis.krippendorff_alpha])
def test_panel_statistics_are_one_when_every_reviewer_agrees(statistic):
    perfect = np.tile(np.array([[0.0], [1.0], [2.0]]), (10, 5))
    assert statistic(perfect) == pytest.approx(1.0)


@pytest.mark.parametrize("statistic", [analysis.fleiss_kappa, analysis.krippendorff_alpha])
def test_panel_statistics_are_about_zero_when_reviewers_are_guessing(statistic):
    rng = np.random.default_rng(0)
    assert abs(statistic(rng.integers(0, 3, size=(2000, 5)).astype(float))) < 0.05


def test_the_ordinal_metric_treats_an_off_by_one_as_milder_than_an_inversion():
    """Calling a High frame Medium is a smaller failure than calling it Low."""

    truth = np.repeat([[0.0], [1.0], [2.0]], 20, axis=0)
    near = np.hstack([truth, np.clip(truth + 1, 0, 2)])
    inverted = np.hstack([truth, 2 - truth])
    assert analysis.krippendorff_alpha(near, "ordinal") > analysis.krippendorff_alpha(
        inverted, "ordinal"
    )


def test_krippendorff_keeps_frames_a_reviewer_could_not_score():
    """Unscorable frames leave holes; alpha uses the remaining ratings, Fleiss drops the row."""

    holes = np.tile(np.array([[0.0], [1.0], [2.0]]), (10, 5))
    holes[0, 0] = np.nan
    holes[5, 3] = np.nan
    assert analysis.krippendorff_alpha(holes) == pytest.approx(1.0)
    assert np.isfinite(analysis.fleiss_kappa(holes))


def test_icc_is_one_for_identical_raters_and_low_for_unrelated_ones():
    column = np.arange(1.0, 21.0)
    identical = np.column_stack([column, column, column])
    assert analysis.icc_two_way_agreement(identical) == pytest.approx(1.0)
    rng = np.random.default_rng(3)
    unrelated = rng.normal(size=(20, 3))
    assert analysis.icc_two_way_agreement(unrelated) < 0.5


def test_band_of_follows_the_same_boundaries_as_the_sampler():
    assert [analysis.band_of(v) for v in (1, 3, 4, 7, 8, 15)] == [
        "Low", "Low", "Medium", "Medium", "High", "High"
    ]
    assert analysis.band_of(None) is None


def test_markdown_tables_escape_pipes_so_a_header_cannot_shear_the_table():
    frame = pd.DataFrame([{"mean |diff|": 1.5, "note": "a|b"}])
    rendered = analysis.md_table(frame)
    header, rule, row = rendered.splitlines()

    def separators(line: str) -> int:
        return line.replace(r"\|", "").count("|")

    assert separators(header) == separators(rule) == separators(row) == 3
    assert r"\|" in header and r"\|" in row


def test_the_worst_component_callout_looks_across_both_instruments():
    """The table is grouped by instrument, so the first row is not the worst row."""

    rows = [
        {"instrument": "REBA", "component": "legs", "n": 60, "exact": 0.93,
         "qwk": 0.90, "bias": -0.01, "mean abs diff": 0.06},
        {"instrument": "RULA", "component": "wrist", "n": 60, "exact": 0.38,
         "qwk": 0.63, "bias": 0.53, "mean abs diff": 0.62},
    ]
    table = pd.DataFrame(rows).sort_values(
        ["instrument", "mean abs diff"], ascending=[True, False]
    )
    worst = table.loc[table["mean abs diff"].idxmax()]
    assert worst["instrument"] == "RULA"
    assert worst["component"] == "wrist"
    assert table.iloc[0]["component"] == "legs", "grouped sort still puts REBA first"


def test_markdown_tables_show_missing_values_rather_than_nan():
    frame = pd.DataFrame([{"kappa": float("nan"), "n": 3}])
    assert "—" in analysis.md_table(frame)
    assert "nan" not in analysis.md_table(frame).lower()


def test_bootstrap_resamples_participants_not_frames():
    """Frames from one participant are not independent, so the cluster is the person."""

    frame = pd.DataFrame(
        {"participant": ["001"] * 10 + ["002"] * 10, "value": [1.0] * 10 + [5.0] * 10}
    )
    low, high = analysis.bootstrap_ci(
        frame, lambda f: float(f["value"].mean()), iterations=300, cluster="participant"
    )
    # With two clusters the mean can only be 1, 3 or 5; row resampling would concentrate near 3.
    assert low == pytest.approx(1.0)
    assert high == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Loading and recomposition
# ---------------------------------------------------------------------------


def _observation(**overrides):
    obs = {
        "trunk": "flex_20_60", "trunk_twisted": False, "trunk_side_bent": False,
        "neck": "flex_10_20", "neck_twisted": False, "neck_side_bent": False,
        "legs": "bilateral", "seated": False, "legs_supported_balanced": True, "knee": "k30_60",
        "upper_arm": "f45_90", "arm_abducted": False, "shoulder_raised": False,
        "arm_supported": False, "lower_arm": "f60_100", "lower_arm_across": False,
        "wrist": "le15", "wrist_deviated": False, "wrist_twist_end": False,
    }
    obs.update(overrides)
    return obs


def _key_frame(tmp_path):
    key = pd.DataFrame(
        [
            {
                "frame_id": "frame_001", "band": "Medium", "stratum": "sustained",
                "participant": 1, "task": "task1_niosh_lifting", "sampling_weight": 1.0,
                "band_run_sec": 1.2, "pipeline_reba_final": 6, "pipeline_rula_final": 5,
                "given_reba_load_force": 1, "given_reba_coupling": 0, "given_reba_activity": 1,
                "given_rula_force_load": 0, "given_rula_muscle_use": 1,
                **{f"pipeline_reba_{n}": 2 for n in analysis.REBA_COMPONENTS},
                **{f"pipeline_rula_{n}": 2 for n in analysis.RULA_COMPONENTS},
            }
        ]
    )
    path = tmp_path / "ground_truth_key.csv"
    key.to_csv(path, index=False)
    return path


def _app_export(tmp_path, name="ratings_R01.json", observations=None, cannot=False):
    payload = {
        "study_id": "test", "rater_id": "R01",
        "ratings": [
            {
                "rater_id": "R01", "presentation_index": 1, "item_key": "frame_001",
                "frame_id": "frame_001", "is_repeat": False, "cannot_score": cannot,
                "cannot_reason": "", "confidence": "4", "notes": "",
                "observations": observations if observations is not None else _observation(),
            }
        ],
    }
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_app_export_is_recomposed_from_its_observations(tmp_path):
    key_path = _key_frame(tmp_path)
    ratings = analysis.load_ratings([_app_export(tmp_path)])
    merged = analysis.merge_truth(ratings, key_path)

    expected = scoring.compose_reba(
        _observation(),
        {"reba_load_force": 1, "reba_coupling": 0, "reba_activity": 1,
         "rula_force_load": 0, "rula_muscle_use": 1},
    )
    assert merged.loc[0, "expert_reba_final"] == expected["final"]
    assert merged.loc[0, "expert_band"] == analysis.band_of(expected["final"])
    assert merged.loc[0, "pipeline_band"] == "Medium"
    assert bool(merged.loc[0, "complete"])


def test_a_scores_precomputed_by_the_page_are_not_trusted_over_the_observations(tmp_path):
    """The page ships its own totals; the analysis must recompute them anyway."""

    key_path = _key_frame(tmp_path)
    payload = json.loads(_app_export(tmp_path).read_text(encoding="utf-8"))
    payload["ratings"][0]["expert_reba_final"] = 99      # nonsense, must be ignored
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    merged = analysis.merge_truth(analysis.load_ratings([path]), key_path)
    assert merged.loc[0, "expert_reba_final"] != 99


def test_an_unscorable_frame_contributes_no_score(tmp_path):
    key_path = _key_frame(tmp_path)
    ratings = analysis.load_ratings([_app_export(tmp_path, cannot=True, observations={})])
    merged = analysis.merge_truth(ratings, key_path)
    assert pd.isna(merged.loc[0, "expert_reba_final"])
    assert not bool(merged.loc[0, "complete"])


def test_repeats_are_recognised_from_the_item_key(tmp_path):
    key_path = _key_frame(tmp_path)
    payload = json.loads(_app_export(tmp_path).read_text(encoding="utf-8"))
    payload["ratings"][0]["item_key"] = "frame_001#repeat"
    payload["ratings"][0]["frame_id"] = "frame_001"
    path = tmp_path / "repeat.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    merged = analysis.merge_truth(analysis.load_ratings([path]), key_path)
    assert bool(merged.loc[0, "is_repeat"])
    assert merged.loc[0, "frame_id"] == "frame_001"


def test_an_unreadable_return_is_skipped_without_losing_the_others(tmp_path):
    key_path = _key_frame(tmp_path)
    good = _app_export(tmp_path)
    bad = tmp_path / "broken.json"
    bad.write_text("{not json", encoding="utf-8")
    ratings = analysis.load_ratings([bad, good])
    assert len(ratings) == 1
    assert analysis.merge_truth(ratings, key_path).loc[0, "rater_id"] == "R01"


def test_lock_files_left_by_excel_are_not_mistaken_for_returns(tmp_path):
    _app_export(tmp_path)
    (tmp_path / "~$scoring_workbook_R01.xlsx").write_text("lock", encoding="utf-8")
    paths = analysis.resolve_rating_paths([str(tmp_path)])
    assert all(not path.name.startswith("~$") for path in paths)


def test_the_workbook_and_the_page_reach_the_report_identically(tmp_path):
    """The same judgements, delivered both ways, must produce the same numbers."""

    openpyxl = pytest.importorskip("openpyxl")
    pytest.importorskip("PIL")
    from PIL import Image

    from validation import build_workbook
    from validation.build_expert_package import build_items

    key_path = _key_frame(tmp_path)
    key = pd.read_csv(key_path)
    key["image_filename"] = "frame_001.jpg"
    key["assumed_load_kg"] = 5.0
    key["assumed_coupling"] = "good"
    key["panel_bounds"] = "[[0, 100], [106, 206], [212, 312]]"
    key["composite_width"] = 312

    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (312, 72), (80, 80, 80)).save(images / "frame_001.jpg", "JPEG")
    items = build_items(key, repeats=0, seed=1)
    workbook_path = tmp_path / "scoring_workbook_R02.xlsx"
    build_workbook.build(items, images, workbook_path, "R02")

    # Enter the same answers a page rater would have given.
    obs = _observation()
    book = openpyxl.load_workbook(workbook_path)
    sheet = book["F001"]
    for offset, (field, _label, kind, _help) in enumerate(build_workbook.QUESTIONS):
        row = build_workbook.R_OBS + offset
        if kind == "yesno":
            sheet.cell(row=row, column=2, value="Yes" if obs[field] else "No")
        else:
            label = next(
                lab for lab, k in build_workbook.OPTION_LABELS[kind] if k == obs[field]
            )
            sheet.cell(row=row, column=2, value=label)
    sheet.cell(row=build_workbook.R_META, column=2, value="4 — Confident")
    book.save(workbook_path)
    book.close()

    merged = analysis.merge_truth(
        analysis.load_ratings([_app_export(tmp_path), workbook_path]), key_path
    )
    assert set(merged["source"]) == {"app", "workbook"}
    assert merged["expert_reba_final"].nunique() == 1
    assert merged["expert_rula_final"].nunique() == 1
    for name in analysis.REBA_COMPONENTS:
        assert merged[f"expert_reba_{name}"].nunique() == 1, name
    for name in analysis.RULA_COMPONENTS:
        assert merged[f"expert_rula_{name}"].nunique() == 1, name
