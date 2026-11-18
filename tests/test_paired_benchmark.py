import json

import numpy as np
import pandas as pd
import pytest

from validation import paired_benchmark as pb
from validation.analyze_ratings import quadratic_weighted_kappa


@pytest.fixture
def inputs(tmp_path):
    identities = pd.DataFrame(
        [
            {
                "frame_id": f"f{i}",
                "participant": person,
                "study": "s1",
                "task": "t",
                "frame_number": 10 * i,
            }
            for i, person in enumerate(["01", "01", "02", "03"], 1)
        ]
    )
    original = identities.assign(reba_final=[4, 4, 6, 8], rula_final=[3, 4, 6, 7])
    candidate = identities.assign(reba_final=[3, 3, "", 7], rula_final=[3, 4, 5, 6])
    originals = original.rename(columns={f"{m}_final": f"pipeline_{m}_final" for m in pb.LEVELS})
    raters = pd.concat(
        [
            originals.assign(
                rater_id="R1", expert_reba_final=[3, 3, 5, 7], expert_rula_final=[2, 3, 5, 7]
            ),
            originals.assign(
                rater_id="R2", expert_reba_final=[4, 3, 5, 7], expert_rula_final=[3, "", 5, 7]
            ),
        ]
    ).assign(complete=True, is_repeat=False, cannot_score=False)
    repeat = raters.iloc[:1].assign(is_repeat=True, expert_reba_final=15)
    ref, before, after = [
        tmp_path / f"{name}.csv" for name in ("reference", "baseline", "candidate")
    ]
    pd.concat([raters, repeat]).to_csv(ref, index=False)
    original.to_csv(before, index=False)
    candidate.iloc[::-1].to_csv(after, index=False)
    return ref, before, after


def test_frozen_medians_coverage_paired_errors_and_official_bands(inputs):
    report = pb.benchmark(*inputs, draws=100)
    assert report["ratings"] == {
        "non_repeat_rows": 8,
        "reviewers": 2,
        "available_reba_reviewer_totals": 8,
        "available_rula_reviewer_totals": 7,
    }
    result = report["methods"]["reba"]
    assert result["reference_frames"] == 4
    assert result["all_available"]["historical"] == result["all_available"]["baseline"]
    assert result["lost_candidate_frame_ids"] == ["f3"]
    assert result["paired_frames"] == 3 and result["paired_participants"] == 2
    scores = result["paired"]["candidate"]
    x, y = pd.Series([3.5, 3, 7]), pd.Series([3, 3, 7])
    assert scores["quadratic_weighted_kappa"] == pytest.approx(quadratic_weighted_kappa(x, y))
    assert scores["mae"] == pytest.approx(0.5 / 3)
    assert scores["exact"] == scores["band_agreement"] == pytest.approx(2 / 3)
    matrix = np.array(scores["band_confusion"]["expert_rows_prediction_columns"])
    assert matrix.shape == (5, 5) and matrix[2, 1] == 1 and matrix.sum() == 3
    assert result["candidate_minus_baseline"]["mae"]["estimate"] == pytest.approx(-2 / 3)
    # Each cluster, drawn by itself, gives the two interval extremes.
    assert result["candidate_minus_baseline"]["mae"]["ci95"] == [-1, -0.5]
    assert result["candidate_minus_baseline"]["quadratic_weighted_kappa"]["estimable_draws"] < 100
    json.dumps(report, allow_nan=False)


def test_no_paired_frames_or_estimable_kappa(inputs):
    ref, before, after = inputs
    data = pd.read_csv(after).assign(reba_final="")
    data.to_csv(after, index=False)
    data = pd.read_csv(ref).assign(expert_rula_final=4)
    data.to_csv(ref, index=False)
    for path in (before, after):
        pd.read_csv(path).assign(rula_final=4).to_csv(path, index=False)
    report = pb.benchmark(*inputs, draws=5)["methods"]
    assert report["reba"]["paired_frames"] == 0
    assert report["reba"]["candidate_minus_baseline"]["mae"]["ci95"] is None
    assert report["rula"]["paired"]["candidate"]["quadratic_weighted_kappa"] is None


@pytest.mark.parametrize(
    "problem", ["frame_number", "duplicate", "omitted", "extra", "range", "fraction", "typo"]
)
def test_rejects_misjoins_and_invalid_scores(inputs, problem):
    df = pd.read_csv(inputs[2], dtype=str, keep_default_na=False)
    if problem == "frame_number":
        df.loc[0, "frame_number"] = "665"
    elif problem in {"duplicate", "extra"}:
        df = pd.concat(
            [df, df.iloc[:1].assign(frame_id="other" if problem == "extra" else df.frame_id[0])]
        )
    elif problem == "omitted":
        df = df.iloc[1:]
    else:
        df.loc[0, "rula_final"] = {"range": "8", "fraction": "2.5", "typo": "unknown"}[problem]
    df.to_csv(inputs[2], index=False)
    with pytest.raises(ValueError):
        pb.benchmark(*inputs, draws=5)


def test_rejects_conflicting_historical_totals(inputs):
    data = pd.read_csv(inputs[0])
    data.loc[0, "pipeline_reba_final"] = 12
    data.to_csv(inputs[0], index=False)
    with pytest.raises(ValueError, match="Conflicting"):
        pb.benchmark(*inputs, draws=5)


def test_design_weighted_sensitivity_and_task_breakdown_use_paired_cases(inputs):
    ref, *_ = inputs
    for path in inputs:
        df = pd.read_csv(path)
        df["task"] = df.frame_id.map({"f1": "a", "f2": "a", "f3": "b", "f4": "b"})
        if path == ref:
            df["sampling_weight"] = df.frame_id.map({"f1": 4, "f2": 1, "f3": 100, "f4": 1})
        df.to_csv(path, index=False)
    result = pb.benchmark(*inputs, draws=5)["methods"]["reba"]
    sen = result["design_weighted_paired"]
    assert sen["sum_weights"] == 6 and sen["paired_frames"] == 3
    assert sen["candidate"]["mae"] == pytest.approx(2 / 6)
    assert sen["candidate"]["exact"] == pytest.approx(2 / 6)
    assert sen["candidate"]["within_one"] == 1
    assert sen["candidate_minus_baseline"]["mae"] == pytest.approx(-2 / 6)
    assert result["paired_by_task"]["b"]["reference_frames"] == 2
    assert result["paired_by_task"]["b"]["paired_frames"] == 1
    assert result["paired_by_task"]["a"]["candidate"]["mae"] == 0.25


@pytest.mark.parametrize("bad_weight", ["-1", "0", "inf", "", "?", "2"])
def test_sampling_weights_must_be_positive_finite_and_consistent(inputs, bad_weight):
    data = pd.read_csv(inputs[0], dtype=str).assign(sampling_weight="1")
    data.loc[0, "sampling_weight"] = bad_weight
    data.to_csv(inputs[0], index=False)
    with pytest.raises(ValueError):
        pb.benchmark(*inputs, draws=5)
