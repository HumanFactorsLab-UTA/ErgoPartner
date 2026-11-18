"""Compare two runs on identical source frames against frozen expert totals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ergonomics.reba import REBACalculator
from ergonomics.rula import RULACalculator
from paths import PROJECT_ROOT

DEFAULT_REFERENCE = (
    PROJECT_ROOT
    / "validation_sample/delivery/3_when_ratings_return"
    / "analysis_2026-09-17_updated/combined_ratings.csv"
)
FROZEN_SHA256 = "cc3b839c36f8b2ece2dec621ab9ebb2f0a423203da7076227a99c27294ae385c"
IDENTITY = ["participant", "study", "task", "frame_number"]
LEVELS = {
    "reba": {score: risk[0] for score, risk in REBACalculator.RISK_LEVELS.items()},
    "rula": {score: risk[0] for score, risk in RULACalculator.RISK_LEVELS.items()},
}
METRICS = ["mae", "bias", "exact", "within_one", "band_agreement", "quadratic_weighted_kappa"]


def _read(path: Path, required: list[str]) -> pd.DataFrame:
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(["frame_id", *IDENTITY, *required]) - set(data.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns {sorted(missing)}")
    if (
        data.empty
        or data[["frame_id", *IDENTITY]].apply(lambda c: c.str.strip().eq("")).any().any()
    ):
        raise ValueError(f"{path.name} has missing frame identities")
    if not data["frame_number"].str.fullmatch(r"[0-9]+").all():
        raise ValueError(f"{path.name} contains an invalid source frame number")
    data["frame_number"] = data.frame_number.astype(int)
    # The study key pads numeric participant IDs. The frozen reviewer export does not.
    data["participant"] = data.participant.map(lambda s: str(int(s)) if s.isdecimal() else s)
    return data


def _scores(column: pd.Series, instrument: str) -> pd.Series:
    scores = pd.to_numeric(column.mask(column.eq("")), errors="raise")
    invalid = scores.notna() & ~scores.isin(LEVELS[instrument])
    if invalid.any():
        raise ValueError(
            f"{column.name} contains a score outside the worksheet scale: {column[invalid].iloc[0]}"
        )
    return scores


def _reference(path: Path) -> tuple[pd.DataFrame, dict]:
    required = ["rater_id", "is_repeat", "complete", "cannot_score"]
    required.extend(f"{source}_{m}_final" for source in ("expert", "pipeline") for m in LEVELS)
    data = _read(path, required)
    for flag in ("is_repeat", "complete", "cannot_score"):
        values = data[flag].str.lower()
        if not values.isin(["true", "false"]).all():
            raise ValueError(f"Invalid {flag} in frozen ratings")
        data[flag] = values.eq("true")
    first = data.loc[~data.is_repeat].copy()
    if (
        first.empty
        or first.rater_id.str.strip().eq("").any()
        or first.duplicated(["frame_id", "rater_id"]).any()
    ):
        raise ValueError("Expected one identified first rating per frame and reviewer")
    eligible = first.complete & ~first.cannot_score
    details = {"non_repeat_rows": len(first), "reviewers": int(first.rater_id.nunique())}
    grouped = first.groupby("frame_id", sort=True)
    locked = [*IDENTITY, *(f"pipeline_{m}_final" for m in LEVELS)]
    if "sampling_weight" in first.columns:
        weights = pd.to_numeric(first.sampling_weight, errors="raise")
        if not np.isfinite(weights).all() or not (weights > 0).all():
            raise ValueError(
                "Every frame needs a finite positive sampling_weight when weights are supplied"
            )
        first["sampling_weight"] = weights
        locked.append("sampling_weight")
    for m in LEVELS:
        for source in ("pipeline", "expert"):
            name = f"{source}_{m}_final"
            first[name] = _scores(first[name], m)
        name = f"expert_{m}_final"
        first.loc[~eligible, name] = np.nan
        details[f"available_{m}_reviewer_totals"] = int(first[name].count())
    if grouped[locked].nunique(dropna=False).gt(1).any().any():
        raise ValueError(
            "Conflicting source identities, sampling weights or historical predictions across reviewers"
        )
    frames = grouped[locked].first()
    if frames.duplicated(IDENTITY).any():
        raise ValueError("Multiple reference frame IDs point to the same source frame")
    for m in LEVELS:
        frames[f"expert_{m}_final"] = grouped[f"expert_{m}_final"].median()
    return frames, details


def _predictions(path: Path, reference: pd.DataFrame) -> pd.DataFrame:
    scores = [f"{m}_final" for m in LEVELS]
    data = _read(path, scores).set_index("frame_id", verify_integrity=True)
    if set(data.index) != set(reference.index):
        raise ValueError(f"{path.name} must contain exactly one row for every reference frame ID")
    data = data.loc[reference.index]
    mismatch = (data[IDENTITY] != reference[IDENTITY]).any(axis=1)
    if mismatch.any():
        raise ValueError(f"{path.name} has a source identity mismatch at {mismatch.idxmax()}")
    for m in LEVELS:
        data[f"{m}_final"] = _scores(data[f"{m}_final"], m)
    return data[scores]


def _band_indices(values: np.ndarray, instrument: str) -> np.ndarray:
    levels = LEVELS[instrument]
    names = list(dict.fromkeys(levels.values()))
    by_score = np.array([-1, *(names.index(levels[i]) for i in range(1, max(levels) + 1))])
    return by_score[np.rint(values).astype(int)]


def _metrics(expert: np.ndarray, prediction: np.ndarray, band_matches: np.ndarray) -> np.ndarray:
    diff = prediction - expert
    # Expected squared distance under independent empirical marginals. The common
    # scale factor in quadratic Cohen weights cancels, including for half points.
    expected = expert.var() + prediction.var() + diff.mean() ** 2
    qwk = 1 - np.mean(diff**2) / expected if expected > 0 else np.nan
    return np.array(
        [
            np.abs(diff).mean(),
            diff.mean(),
            (diff == 0).mean(),
            (np.abs(diff) <= 1).mean(),
            band_matches.mean(),
            qwk,
        ]
    )


def _summary(data: pd.DataFrame, version: str, instrument: str) -> dict:
    subset = data[["expert", version]].dropna()
    levels = list(dict.fromkeys(LEVELS[instrument].values()))
    confusion = np.zeros((len(levels), len(levels)), dtype=int)
    values = np.full(len(METRICS), np.nan)
    if len(subset):
        expert, prediction = subset.to_numpy(dtype=float).T
        rows, cols = (_band_indices(v, instrument) for v in (expert, prediction))
        np.add.at(confusion, (rows, cols), 1)
        values = _metrics(expert, prediction, rows == cols)
    return {
        "n": len(subset),
        **{
            name: float(v) if np.isfinite(v) else None
            for name, v in zip(METRICS, values, strict=True)
        },
        "band_confusion": {"levels": levels, "expert_rows_prediction_columns": confusion.tolist()},
    }


def _intervals(paired: pd.DataFrame, instrument: str, draws: int, seed: int) -> dict:
    groups = list(paired.groupby("participant").indices.values())
    deltas = np.full((draws, len(METRICS)), np.nan)
    if len(groups) >= 2:
        rng = np.random.default_rng(seed)
        expert, before, after = paired[["expert", "baseline", "candidate"]].to_numpy(dtype=float).T
        expert_bands = _band_indices(expert, instrument)
        before_match = expert_bands == _band_indices(before, instrument)
        after_match = expert_bands == _band_indices(after, instrument)
        for i in range(draws):
            idx = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
            deltas[i] = _metrics(expert[idx], after[idx], after_match[idx]) - _metrics(
                expert[idx], before[idx], before_match[idx]
            )
    output = {}
    for name, column in zip(METRICS, deltas.T, strict=True):
        finite = column[np.isfinite(column)]
        output[name] = {
            "estimable_draws": len(finite),
            "ci95": np.percentile(finite, [2.5, 97.5]).tolist() if len(finite) else None,
        }
    return output


def _weighted_sensitivity(paired: pd.DataFrame) -> dict:
    weights = paired.sampling_weight.to_numpy(dtype=float)
    totals = {}
    for version in ("baseline", "candidate"):
        error = (paired[version] - paired.expert).abs().to_numpy(dtype=float)
        metrics = {"mae": error, "exact": error == 0, "within_one": error <= 1}
        totals[version] = {
            name: float(np.average(values, weights=weights)) if weights.size else None
            for name, values in metrics.items()
        }
    delta = {}
    for name, after in totals["candidate"].items():
        before = totals["baseline"][name]
        delta[name] = after - before if before is not None and after is not None else None
    return {
        "interpretation": "Descriptive sensitivity using original frame sampling weights on paired cases",
        "paired_frames": len(paired),
        "sum_weights": float(weights.sum()),
        **totals,
        "candidate_minus_baseline": delta,
    }


def benchmark(
    reference: Path, baseline: Path, candidate: Path, draws: int = 2000, seed: int = 7
) -> dict:
    if draws <= 0 or seed < 0:
        raise ValueError("Bootstrap draws must be positive and seed must be nonnegative")
    inputs = {}
    for name, path in (("reference", reference), ("baseline", baseline), ("candidate", candidate)):
        inputs[name] = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if (
        reference.resolve() == DEFAULT_REFERENCE.resolve()
        and inputs["reference"]["sha256"] != FROZEN_SHA256
    ):
        raise ValueError(
            "The default frozen reference checksum differs from the verified reviewer export"
        )
    ref, details = _reference(reference)
    runs = {
        name: _predictions(path, ref)
        for name, path in (("baseline", baseline), ("candidate", candidate))
    }
    results = {}
    for instrument in LEVELS:
        col = f"{instrument}_final"
        data = pd.DataFrame(
            {
                "expert": ref[f"expert_{col}"],
                "historical": ref[f"pipeline_{col}"],
                "participant": ref.participant,
                "task": ref.task,
                **{name: run[col] for name, run in runs.items()},
            }
        )
        if "sampling_weight" in ref:
            data["sampling_weight"] = ref.sampling_weight
        paired = data.dropna(subset=["expert", *runs])
        summaries = {name: _summary(paired, name, instrument) for name in runs}
        deltas = _intervals(paired, instrument, draws, seed)
        for name in METRICS:
            left, right = (summaries[run][name] for run in runs)
            deltas[name]["estimate"] = (
                right - left if left is not None and right is not None else None
            )
        lost = data.expert.notna() & data.baseline.notna() & data.candidate.isna()
        gained = data.expert.notna() & data.baseline.isna() & data.candidate.notna()
        by_task = {}
        for task, task_rows in data.groupby("task"):
            task_pairs = task_rows.dropna(subset=["expert", *runs])
            by_task[task] = {
                "reference_frames": int(task_rows.expert.count()),
                "paired_frames": len(task_pairs),
                **{name: _summary(task_pairs, name, instrument) for name in runs},
            }
        results[instrument] = {
            "reference_frames": int(data.expert.count()),
            "paired_frames": len(paired),
            "paired_participants": int(paired.participant.nunique()),
            "all_available": {
                name: _summary(data, name, instrument) for name in ("historical", *runs)
            },
            "lost_candidate_frame_ids": list(data.index[lost]),
            "gained_candidate_frame_ids": list(data.index[gained]),
            "paired": summaries,
            "candidate_minus_baseline": deltas,
            "paired_by_task": by_task,
            "design_weighted_paired": _weighted_sensitivity(paired)
            if "sampling_weight" in ref
            else None,
        }
    return {
        "designation": "Development benchmark on the recorded reviewer panel; unweighted sampled frames",
        "reference_policy": "Per-frame median of available, complete, first reviewer totals; no imputation",
        "band_policy": "Official score levels after rounding a median to nearest integer, ties to even, for bands only",
        "bootstrap": {
            "unit": "participant",
            "draws": draws,
            "seed": seed,
            "interval": "paired percentile",
        },
        "inputs": inputs,
        "source_frames": len(ref),
        "participants": int(ref.participant.nunique()),
        "ratings": details,
        "methods": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new JSON file")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    try:
        report = benchmark(args.reference, args.baseline, args.candidate, args.draws, args.seed)
        payload = json.dumps(report, indent=2, allow_nan=False)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload + "\n")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Benchmark failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
