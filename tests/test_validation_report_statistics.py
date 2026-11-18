"""Regression checks for returned-reviewer coverage and reported agreement."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from validation import analyze_ratings as analysis


def _row(frame="frame_001", rater="R01", score=3, **changes):
    row = {
        "frame_id": frame, "rater_id": rater, "item_key": frame,
        "is_repeat": False, "cannot_score": False, "complete": True,
        "expert_reba_final": score, "expert_rula_final": score,
        "expert_band": analysis.band_of(score),
        "pipeline_reba_final": score, "pipeline_rula_final": score,
        "pipeline_band": analysis.band_of(score),
        "band": analysis.band_of(score), "stratum": "sustained",
        "participant": "001", "task": "lifting", "sampling_weight": 1.0,
        **{f"expert_reba_{name}": 2 for name in analysis.REBA_COMPONENTS},
        **{f"expert_rula_{name}": 2 for name in analysis.RULA_COMPONENTS},
        **{f"pipeline_reba_{name}": 2 for name in analysis.REBA_COMPONENTS},
        **{f"pipeline_rula_{name}": 2 for name in analysis.RULA_COMPONENTS},
    }
    row.update(changes)
    return row


def _capture_tables(monkeypatch):
    tables = []
    original = analysis.md_table

    def capture(frame, index=False):
        tables.append(frame.copy())
        return original(frame, index=index)

    monkeypatch.setattr(analysis, "md_table", capture)
    return tables


@pytest.mark.parametrize("left,right", [
    ([1, 2, 8], [1, 8, 8]),
    ([2, 2.5, 7], [2, 3, 6]),
])
def test_qwk_uses_numeric_distances_for_sparse_and_half_scores(left, right):
    x, y = np.asarray(left), np.asarray(right)
    # Independent calculation: observed squared error / expected squared error
    # under independent empirical marginals, without constructing categories.
    expected = 1 - np.mean((x - y) ** 2) / np.mean((x[:, None] - y[None, :]) ** 2)
    assert analysis.quadratic_weighted_kappa(pd.Series(left), pd.Series(right)) == pytest.approx(expected)


def test_exact_agreement_weights_available_pairs_only():
    left = pd.Series(["Low", "Medium", None, "High"])
    right = pd.Series(["Low", "High", "Low", None])
    weights = pd.Series([9.0, 1.0, 1000.0, 1000.0])
    assert analysis.exact_agreement(left, right) == pytest.approx(0.5)
    assert analysis.exact_agreement(left, right, weights) == pytest.approx(0.9)
    assert np.isnan(analysis.exact_agreement(left, right, weights * 0))


def test_consensus_excludes_incomplete_unscorable_and_repeat_ratings():
    data = pd.DataFrame([
        _row(rater="R01", score=3),
        _row(rater="R02", score=4, expert_rula_final=None),
        _row(rater="R03", score=None, complete=False, expert_rula_wrist_twist=1),
        _row(rater="R04", score=15, cannot_score=True, expert_rula_wrist_twist=1),
        _row(rater="R01", score=15, is_repeat=True, expert_rula_wrist_twist=1),
    ])
    consensus = analysis._consensus(data).iloc[0]
    assert consensus["expert_reba_final"] == 3.5
    assert consensus["expert_band"] == "Medium"
    assert consensus["expert_rula_wrist_twist"] == 2
    assert consensus["n_raters"] == 2
    assert consensus["n_reba_raters"] == 2
    assert consensus["n_rula_raters"] == 1
    assert analysis.band_of(7.5) == "High"


def test_incomplete_observations_do_not_create_default_components(tmp_path):
    key = tmp_path / "key.csv"
    pd.DataFrame([{
        "frame_id": "frame_001", "band": "Low", "pipeline_reba_final": 2,
        "pipeline_rula_final": 2,
    }]).to_csv(key, index=False)
    ratings = pd.DataFrame([{
        "frame_id": "frame_001", "rater_id": "R03", "cannot_score": False,
        "observations": {"wrist_twist_end": False},
    }])
    merged = analysis.merge_truth(ratings, key)
    expert_columns = [column for column in merged if column.startswith("expert_")]
    assert merged.loc[0, expert_columns].isna().all()
    assert not bool(merged.loc[0, "complete"])


def test_empty_and_unscorable_reviewers_are_excluded_from_panel(monkeypatch):
    data = pd.DataFrame([
        _row(frame=f"frame_{frame:03}", rater=rater, score=score)
        for frame, score in enumerate((1, 5, 8), start=1)
        for rater in ("R01", "R02")
    ] + [
        _row(rater="R03", score=None, complete=False),
        _row(rater="R04", score=15, cannot_score=True),
    ])
    tables = _capture_tables(monkeypatch)
    report = "\n".join(analysis.section_between_rater(data))
    assert "4 reviewers returned ratings; 2 with usable REBA data across 3 frames" in report
    assert "no usable scores: R03, R04" in report
    panel = tables[0]
    assert panel["value"].tolist() == [1.0, 1.0, 1.0]
    assert "3 complete panel frames" in panel.iloc[0]["note"]
    assert tables[1]["pair"].tolist() == ["R01 vs R02"]


@pytest.mark.parametrize("usable_count", [0, 1])
def test_insufficient_usable_reviewers_is_reported_without_crashing(usable_count):
    data = pd.DataFrame([
        _row(rater="R01", score=3 if usable_count else None, complete=bool(usable_count)),
        _row(rater="R03", score=None, complete=False),
    ])
    report = "\n".join(analysis.section_between_rater(data))
    assert f"{usable_count} with usable REBA data" in report
    assert "cannot be estimated" in report


def test_an_entirely_incomplete_return_has_no_pipeline_or_component_scores():
    data = pd.DataFrame([_row(rater="R03", score=None, complete=False)])
    assert analysis._consensus(data).empty
    assert "No scorable frames." in analysis.section_pipeline_agreement(data)
    assert "No component scores available to compare." in analysis.section_components(data)


def test_repeat_pair_denominator_excludes_missing_and_unscorable_scores(monkeypatch):
    data = pd.DataFrame([
        _row(frame=f"frame_{frame:03}", score=score)
        for frame, score in enumerate((1, 5, 8, 9), start=1)
    ] + [
        _row(frame="frame_001", score=1, is_repeat=True),
        _row(frame="frame_002", score=5, is_repeat=True),
        _row(frame="frame_003", score=None, is_repeat=True),
        _row(frame="frame_004", score=9, is_repeat=True, cannot_score=True),
    ])
    tables = _capture_tables(monkeypatch)
    analysis.section_within_rater(data)
    assert tables[0].iloc[0]["n"] == 2
    assert tables[0].iloc[0]["band kappa"] == 1


def test_pipeline_report_weights_exact_agreement_and_uses_available_totals(monkeypatch):
    data = pd.DataFrame([
        _row(frame="frame_001", score=1, sampling_weight=9.0),
        _row(frame="frame_002", score=5, pipeline_reba_final=8,
             pipeline_rula_final=6, sampling_weight=1.0),
        _row(frame="frame_003", score=8, pipeline_reba_final=None,
             expert_rula_final=None, pipeline_rula_final=7, sampling_weight=1000.0),
    ])
    tables = _capture_tables(monkeypatch)
    report = "\n".join(analysis.section_pipeline_agreement(data))
    band = tables[0].set_index("subset")
    assert band.loc["All frames (balanced design)", "exact agreement"] == 0.5
    assert band.loc["All frames (prevalence-weighted)", "exact agreement"] == 0.9
    assert band.loc["All frames (prevalence-weighted)", "n"] == 2
    reba, rula = tables[1].iloc[0], tables[2].iloc[0]
    assert reba["n"] == 2
    assert reba["within ±1"] == 0.5
    assert rula["n"] == 2
    assert rula["within ±1"] == 1.0
    assert "### RULA total" in report
    assert "not measure uncertainty from selecting different reviewers" in report


def test_stratum_report_does_not_claim_a_causal_instability_cost(monkeypatch):
    data = pd.DataFrame([
        _row(frame=f"frame_{index}", score=score, stratum=stratum)
        for index, (score, stratum) in enumerate([
            (1, "sustained"), (5, "sustained"), (1, "transient"), (5, "transient"),
        ])
    ])
    monkeypatch.setattr(analysis, "bootstrap_ci", lambda *args, **kwargs: (-0.1, 0.1))
    report = "\n".join(analysis.section_pipeline_agreement(data))
    assert "observed association" in report
    assert "cost of frame-to-frame" not in report
    assert "20-point difference" not in report


def _loaded_record():
    return {"rater_id": "R02", "item_key": "frame_001", "frame_id": "frame_001", "source": "workbook"}


def test_filename_identity_override_is_explicit_and_records_provenance(monkeypatch, caplog):
    monkeypatch.setattr(analysis, "_load_xlsx", lambda path: [_loaded_record()])
    path = Path("scoring_workbook_R01.xlsx")
    assert analysis.load_ratings([path]).iloc[0]["rater_id"] == "R02"
    overridden = analysis.load_ratings([path], rater_ids_from_filenames=True).iloc[0]
    assert overridden["rater_id"] == "R01"
    assert overridden["original_rater_id"] == "R02"
    assert overridden["source_file"] == path.name
    assert "stored rater(s) R02 -> R01" in caplog.text


@pytest.mark.parametrize("filename", ["reviewer_one.xlsx", "scoring_workbook_R01.json"])
def test_explicit_filename_override_rejects_unparseable_inputs(filename):
    with pytest.raises(SystemExit, match="cannot parse"):
        analysis.load_ratings([Path(filename)], rater_ids_from_filenames=True)


def test_duplicate_reviewer_item_keys_are_rejected(monkeypatch):
    monkeypatch.setattr(analysis, "_load_xlsx", lambda path: [_loaded_record()])
    with pytest.raises(SystemExit, match=r"Duplicate \(rater_id, item_key\)"):
        analysis.load_ratings([Path("scoring_workbook_R01.xlsx"), Path("scoring_workbook_R02.xlsx")])


def test_cli_accepts_explicit_filename_identity_option():
    args = analysis.parse_args(["--ratings", "returned/", "--rater-ids-from-filenames"])
    assert args.rater_ids_from_filenames
