"""Join returned expert ratings to ground truth and estimate agreement.

Reports, in the order a paper would want them:

1. **Within-rater consistency** from the repeated frames, which describes the
   repeatability of the reviewers' assessments.
2. **Between-rater agreement**, the reference standard's own reliability.
3. **Pipeline vs expert agreement**, reported three ways --
   *balanced* (the design as drawn), *prevalence-weighted* (re-weighted to the
   frames the study actually contains), and *per stratum*.  The balanced number
   gives rare bands more influence because the design deliberately over-samples
   them; kappa is sensitive to prevalence, so a balanced-design kappa does not
   describe the population.
4. **Per-component agreement**, which says *which body segment* the disagreement
   lives in rather than only that the totals differed.

The sustained/transient contrast describes an association between stratum and
agreement. It does not by itself isolate a causal effect of pose instability.

Usage
-----
    python -m validation.analyze_ratings --ratings returned/*.json
    python -m validation.analyze_ratings --ratings returned/ --output report.md
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import re
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from validation import scoring

LOGGER = logging.getLogger("validation.analyze_ratings")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLE_DIR = PROJECT_ROOT / "validation_sample"

BANDS = ("Low", "Medium", "High")
REBA_COMPONENTS = ("trunk", "neck", "legs", "upper_arm", "lower_arm", "wrist")
RULA_COMPONENTS = ("upper_arm", "lower_arm", "wrist", "wrist_twist", "neck", "trunk", "legs")


def band_of(score: Any) -> str | None:
    """Study REBA band after rounding to the nearest integer, with ties to even.

    This retains the study's existing policy for half-integer median scores:
    3.5 becomes 4 (Medium), and 7.5 becomes 8 (High).
    """

    if score is None or (isinstance(float(score) if score is not None else None, float) and pd.isna(score)):
        return None
    value = int(round(float(score)))
    if value <= 3:
        return "Low"
    if value <= 7:
        return "Medium"
    return "High"


# ---------------------------------------------------------------------------
# Agreement statistics
# ---------------------------------------------------------------------------


def cohen_kappa(a: pd.Series, b: pd.Series, categories: tuple[str, ...], weights: pd.Series | None = None) -> float:
    """Cohen's kappa, optionally with per-observation sampling weights."""

    mask = a.isin(categories) & b.isin(categories)
    if weights is not None:
        mask &= weights.notna() & np.isfinite(weights) & (weights > 0)
    a, b = a[mask], b[mask]
    w = pd.Series(1.0, index=a.index) if weights is None else weights[mask].astype(float)
    if a.empty or w.sum() == 0:
        return float("nan")

    index = {name: position for position, name in enumerate(categories)}
    size = len(categories)
    observed = np.zeros((size, size))
    for left, right, weight in zip(a, b, w, strict=True):
        if left in index and right in index:
            observed[index[left], index[right]] += weight
    total = observed.sum()
    if total == 0:
        return float("nan")
    observed /= total
    po = float(np.trace(observed))
    pe = float((observed.sum(axis=0) * observed.sum(axis=1)).sum())
    return float("nan") if np.isclose(pe, 1.0) else (po - pe) / (1 - pe)


def exact_agreement(a: pd.Series, b: pd.Series, weights: pd.Series | None = None) -> float:
    """Proportion of matching available pairs, optionally sampling-weighted."""

    mask = a.notna() & b.notna()
    if weights is not None:
        mask &= weights.notna() & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return float("nan")
    matches = (a[mask] == b[mask]).astype(float)
    if weights is None:
        return float(matches.mean())
    return float(np.average(matches, weights=weights[mask]))


def quadratic_weighted_kappa(a: pd.Series, b: pd.Series) -> float:
    """Weighted kappa using squared numeric score distances, including medians.

    Sparse categories retain their original spacing, and half-integer expert
    medians are not compressed into equally spaced category ranks.
    """

    mask = a.notna() & b.notna()
    x, y = a[mask].astype(float), b[mask].astype(float)
    if x.empty:
        return float("nan")
    values = sorted(set(x) | set(y))
    if len(values) < 2:
        return float("nan")
    index = {value: position for position, value in enumerate(values)}
    size = len(values)
    observed = np.zeros((size, size))
    for left, right in zip(x, y, strict=True):
        observed[index[left], index[right]] += 1
    observed /= observed.sum()
    rows, cols = observed.sum(axis=1), observed.sum(axis=0)
    expected = np.outer(rows, cols)
    scores = np.asarray(values, dtype=float)
    penalty = ((scores[:, None] - scores[None, :]) / (scores[-1] - scores[0])) ** 2
    denominator = float((penalty * expected).sum())
    return float("nan") if np.isclose(denominator, 0) else 1 - float((penalty * observed).sum()) / denominator


def fleiss_kappa(matrix: np.ndarray) -> float:
    """Fleiss' kappa over a targets x raters matrix of category indices.

    The standard multi-rater statistic, and what a reader expects when five reviewers
    scored the same items: one number for the panel rather than ten pairwise ones.
    Rows with any missing rating are dropped, which is what Fleiss' assumes -- use
    :func:`krippendorff_alpha` when unscorable frames make that wasteful.
    """

    data = matrix[~np.isnan(matrix).any(axis=1)]
    if data.shape[0] < 2 or data.shape[1] < 2:
        return float("nan")
    items, raters = data.shape
    categories = sorted(set(data.astype(int).ravel().tolist()))
    counts = np.zeros((items, len(categories)))
    for column, category in enumerate(categories):
        counts[:, column] = (data == category).sum(axis=1)

    proportions = counts.sum(axis=0) / (items * raters)
    agreement = ((counts**2).sum(axis=1) - raters) / (raters * (raters - 1))
    observed = float(agreement.mean())
    expected = float((proportions**2).sum())
    return float("nan") if np.isclose(expected, 1.0) else (observed - expected) / (1 - expected)


def krippendorff_alpha(matrix: np.ndarray, level: str = "ordinal") -> float:
    """Krippendorff's alpha over a targets x raters matrix, tolerating missing ratings.

    Preferred over Fleiss' kappa here because a reviewer may mark a frame unscorable,
    which leaves a hole rather than a category.  ``level='ordinal'`` respects that
    Low < Medium < High, so calling a High frame Medium counts as a smaller failure
    than calling it Low.
    """

    values = sorted({float(v) for v in matrix.ravel() if not np.isnan(v)})
    if len(values) < 2:
        return float("nan")
    index = {value: position for position, value in enumerate(values)}
    size = len(values)

    # Coincidence matrix: how often each pair of values co-occurs on one target.
    coincidence = np.zeros((size, size))
    for row in matrix:
        present = [float(v) for v in row if not np.isnan(v)]
        if len(present) < 2:
            continue
        weight = len(present) - 1
        for left in present:
            for right in present:
                if left is right:
                    continue
                coincidence[index[left], index[right]] += 1.0 / weight
    total = coincidence.sum()
    if total <= 0:
        return float("nan")

    marginals = coincidence.sum(axis=1)
    if level == "nominal":
        distance = 1.0 - np.eye(size)
    else:
        # Ordinal metric: the squared mass of the interval spanned by the two values.
        cumulative = np.cumsum(marginals)
        distance = np.zeros((size, size))
        for c in range(size):
            for k in range(size):
                low, high = min(c, k), max(c, k)
                span = cumulative[high] - cumulative[low] + marginals[low]
                distance[c, k] = (span - (marginals[c] + marginals[k]) / 2.0) ** 2
        scale = distance.max() or 1.0
        distance = distance / scale

    observed = float((coincidence * distance).sum())
    expected = float(
        (np.outer(marginals, marginals) * distance).sum() - (marginals * np.diag(distance)).sum()
    ) / (total - 1)
    return float("nan") if np.isclose(expected, 0) else 1.0 - observed / expected


def icc_two_way_agreement(matrix: np.ndarray) -> float:
    """ICC(2,1), absolute agreement, on a targets x raters matrix."""

    data = matrix[~np.isnan(matrix).any(axis=1)]
    n, k = data.shape
    if n < 2 or k < 2:
        return float("nan")
    grand = data.mean()
    ms_rows = k * ((data.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_cols = n * ((data.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    residual = data - data.mean(axis=1, keepdims=True) - data.mean(axis=0, keepdims=True) + grand
    ms_error = (residual**2).sum() / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    return float("nan") if np.isclose(denominator, 0) else (ms_rows - ms_error) / denominator


def bootstrap_ci(
    frame: pd.DataFrame, statistic, iterations: int = 2000, seed: int = 7, cluster: str | None = "participant"
) -> tuple[float, float]:
    """Percentile CI, resampling clusters rather than rows where a cluster is given.

    Frames from one participant are not independent observations, so resampling rows
    would understate the interval.
    """

    rng = np.random.default_rng(seed)
    if cluster and cluster in frame.columns:
        groups = [group for _, group in frame.groupby(cluster)]
    else:
        groups = [frame.iloc[[i]] for i in range(len(frame))]
    if len(groups) < 2:
        return (float("nan"), float("nan"))
    estimates = []
    for _ in range(iterations):
        picked = rng.integers(0, len(groups), len(groups))
        sample = pd.concat([groups[i] for i in picked], ignore_index=True)
        try:
            value = statistic(sample)
        except Exception:
            value = float("nan")
        if value is not None and np.isfinite(value):
            estimates.append(value)
    if len(estimates) < 50:
        return (float("nan"), float("nan"))
    return (float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5)))


def md_table(frame: pd.DataFrame, index: bool = False) -> str:
    """Render a Markdown table without pulling in ``tabulate``.

    ``DataFrame.to_markdown`` needs an optional dependency the project does not carry,
    and a validation report should not fail to print for want of a formatter.
    """

    table = frame.reset_index() if index else frame
    # A literal pipe would be read as a column break, silently shearing the table.
    headers = [str(column).replace("|", "\\|") for column in table.columns]

    def cell(value: Any) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "—"
        if isinstance(value, float):
            return f"{value:.3f}".rstrip("0").rstrip(".")
        return str(value).replace("|", "\\|")

    rows = [[cell(value) for value in record] for record in table.itertuples(index=False)]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    out = ["| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)) + " |"]
    out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        out.append("| " + " | ".join(v.ljust(w) for v, w in zip(row, widths, strict=True)) + " |")
    return "\n".join(out)


def interpret(kappa: float) -> str:
    """Landis & Koch bands, for orientation only."""

    if not np.isfinite(kappa):
        return "not estimable"
    for threshold, label in ((0.0, "poor"), (0.20, "slight"), (0.40, "fair"),
                             (0.60, "moderate"), (0.80, "substantial")):
        if kappa <= threshold:
            return label
    return "almost perfect"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rater = payload.get("rater_id") or path.stem
    rows = []
    for entry in payload.get("ratings", []):
        rows.append(
            {
                "rater_id": entry.get("rater_id") or rater,
                "item_key": entry.get("item_key") or entry.get("frame_id"),
                "frame_id": entry.get("frame_id"),
                "presentation_index": entry.get("presentation_index"),
                "cannot_score": bool(entry.get("cannot_score")),
                "cannot_reason": entry.get("cannot_reason", ""),
                "confidence": entry.get("confidence"),
                "notes": entry.get("notes", ""),
                "observations": entry.get("observations") or {},
                "source": "app",
            }
        )
    return rows


def _load_xlsx(path: Path) -> list[dict[str, Any]]:
    from validation.build_workbook import read_workbook

    rows = []
    for record in read_workbook(path):
        record["source"] = "workbook"
        record.setdefault("cannot_reason", "")
        rows.append(record)
    return rows


def load_ratings(paths: list[Path], *, rater_ids_from_filenames: bool = False) -> pd.DataFrame:
    """Read every returned file, from either rating path, into one table.

    Only the raw observations are taken from a rater's file.  Scores are recomposed in
    :func:`merge_truth`, so a workbook and a browser export of the same judgements
    produce the same numbers, and neither depends on Excel having evaluated a formula.
    Filename-derived reviewer identities require an explicit opt-in; the stored
    identity is retained as provenance when that override is used.
    """

    rows: list[dict[str, Any]] = []
    for path in paths:
        filename_rater = None
        if rater_ids_from_filenames:
            match = re.fullmatch(r"scoring_workbook_(R\d{2})", path.stem, flags=re.IGNORECASE)
            if path.suffix.lower() != ".xlsx" or match is None:
                raise SystemExit(
                    "--rater-ids-from-filenames requires every input to be named "
                    f"scoring_workbook_Rxx.xlsx; cannot parse {path.name!r}"
                )
            filename_rater = match.group(1).upper()
        try:
            if path.suffix.lower() in {".xlsx", ".xlsm"}:
                records = _load_xlsx(path)
            else:
                records = _load_json(path)
        except Exception as exc:  # noqa: BLE001 - a bad return file must not stop the rest
            LOGGER.warning("Skipping %s: %s", path.name, exc)
            continue
        if filename_rater is not None:
            stored_raters = sorted({str(record["rater_id"]) for record in records})
            if stored_raters != [filename_rater]:
                LOGGER.warning(
                    "Explicit filename identity override for %s: stored rater(s) %s -> %s",
                    path.name, ", ".join(stored_raters) or "none", filename_rater,
                )
            for record in records:
                record["original_rater_id"] = record["rater_id"]
                record["rater_id"] = filename_rater
        for record in records:
            record["source_file"] = path.name
        rows.extend(records)
    if not rows:
        raise SystemExit("no ratings could be read; check --ratings")

    frame = pd.DataFrame(rows)
    duplicates = frame.duplicated(["rater_id", "item_key"], keep=False)
    if duplicates.any():
        conflicts = frame.loc[duplicates, ["rater_id", "item_key", "source_file"]]
        raise SystemExit(
            "Duplicate (rater_id, item_key) ratings would count the same reviewer twice. "
            "Resolve the reviewer identity or duplicate return files before analysis:\n"
            + conflicts.to_string(index=False)
        )
    frame["is_repeat"] = frame["item_key"].astype(str).str.contains("#repeat")
    frame["frame_id"] = frame["frame_id"].astype(str).str.replace("#repeat", "", regex=False)
    by_source = frame["source"].value_counts().to_dict()
    LOGGER.info(
        "Loaded %d rating rows from %d rater(s) (%s): %s",
        len(frame),
        frame["rater_id"].nunique(),
        ", ".join(f"{v} via {k}" for k, v in by_source.items()),
        ", ".join(sorted(frame["rater_id"].astype(str).unique())),
    )
    return frame


def merge_truth(ratings: pd.DataFrame, key_path: Path) -> pd.DataFrame:
    """Join ground truth, then recompose every expert score from its observations.

    The given subscores come from the key rather than the rater's file: they are a
    property of the frame, not of the rating, and taking them from one place keeps the
    two rating paths comparable.
    """

    key = pd.read_csv(key_path)
    merged = ratings.merge(key, on="frame_id", how="left", validate="many_to_one")
    unmatched = int(merged["band"].isna().sum())
    if unmatched:
        LOGGER.warning("%d rating row(s) did not match the ground truth key", unmatched)

    def given_for(row: pd.Series) -> dict[str, int]:
        out = {}
        for field in scoring.GIVEN_FIELDS:
            value = row.get(f"given_{field}")
            out[field] = 0 if value is None or pd.isna(value) else int(value)
        return out

    finals_reba, finals_rula, complete = [], [], []
    components: dict[str, list[Any]] = {
        **{f"expert_reba_{name}": [] for name in REBA_COMPONENTS},
        **{f"expert_rula_{name}": [] for name in RULA_COMPONENTS},
    }
    for _, row in merged.iterrows():
        obs = row["observations"] if isinstance(row["observations"], dict) else {}
        given = given_for(row)
        reba: dict[str, Any]
        rula: dict[str, Any]
        if row["cannot_score"] or not scoring.is_complete(obs):
            reba = {"components": dict.fromkeys(REBA_COMPONENTS), "final": None}
            rula = {"components": dict.fromkeys(RULA_COMPONENTS), "final": None}
            complete.append(False)
        else:
            reba = scoring.compose_reba(obs, given)
            rula = scoring.compose_rula(obs, given)
            complete.append(True)
        finals_reba.append(reba["final"])
        finals_rula.append(rula["final"])
        reba_parts: dict[str, Any] = reba["components"]
        rula_parts: dict[str, Any] = rula["components"]
        for name in REBA_COMPONENTS:
            components[f"expert_reba_{name}"].append(reba_parts.get(name))
        for name in RULA_COMPONENTS:
            components[f"expert_rula_{name}"].append(rula_parts.get(name))

    merged["expert_reba_final"] = finals_reba
    merged["expert_rula_final"] = finals_rula
    merged["complete"] = complete
    for column, values in components.items():
        merged[column] = values
    merged["expert_band"] = merged["expert_reba_final"].map(band_of)
    merged["pipeline_band"] = merged["pipeline_reba_final"].map(band_of)
    return merged


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def section_within_rater(data: pd.DataFrame) -> list[str]:
    lines = ["## 1. Within-rater consistency (repeated frames)", ""]
    repeats = data[data["is_repeat"]]
    if repeats.empty:
        return lines + ["No repeated frames were present in the returned ratings.", ""]

    rows = []
    usable = data[~data["cannot_score"] & data["complete"]]
    for rater, group in usable.groupby("rater_id"):
        first = group[~group["is_repeat"]].set_index("frame_id")
        second = group[group["is_repeat"]].set_index("frame_id")
        shared = first.index.intersection(second.index)
        paired = (
            first.loc[shared, "expert_reba_final"].notna()
            & second.loc[shared, "expert_reba_final"].notna()
        )
        shared = shared[paired.to_numpy()]
        if len(shared) < 2:
            continue
        rows.append(
            {
                "rater": rater,
                "n": len(shared),
                "band kappa": cohen_kappa(
                    first.loc[shared, "expert_band"], second.loc[shared, "expert_band"], BANDS
                ),
                "REBA qwk": quadratic_weighted_kappa(
                    first.loc[shared, "expert_reba_final"], second.loc[shared, "expert_reba_final"]
                ),
                "REBA mean abs diff": float(
                    (first.loc[shared, "expert_reba_final"] - second.loc[shared, "expert_reba_final"])
                    .abs().mean()
                ),
            }
        )
    if not rows:
        return lines + ["Too few matched repeat pairs to estimate consistency.", ""]
    table = pd.DataFrame(rows).round(3)
    lines += [md_table(table), ""]
    lines += [
        "_Only repeat pairs with two usable REBA scores are included. Repeatability "
        "describes reference uncertainty; it is not a mathematical ceiling on agreement "
        "between the pipeline and a pooled expert consensus._",
        "",
    ]
    return lines


def section_between_rater(data: pd.DataFrame) -> list[str]:
    lines = ["## 2. Between-rater agreement (reference standard reliability)", ""]
    first = data[~data["is_repeat"]]
    raters = sorted(first["rater_id"].unique())
    usable = first[~first["cannot_score"] & first["complete"]]
    wide_band = usable.pivot_table(
        index="frame_id", columns="rater_id", values="expert_band", aggfunc="first"
    ).reindex(index=first["frame_id"].unique(), columns=raters)
    wide_reba = usable.pivot_table(
        index="frame_id", columns="rater_id", values="expert_reba_final", aggfunc="first"
    ).reindex(index=wide_band.index, columns=raters)
    active_raters = [rater for rater in raters if wide_reba[rater].notna().any()]
    empty_raters = [rater for rater in raters if rater not in active_raters]
    n_frames = int(wide_reba.notna().any(axis=1).sum())
    lines += [
        f"**{len(raters)} reviewers returned ratings; {len(active_raters)} with usable "
        f"REBA data across {n_frames} frames.**",
        "",
    ]
    if empty_raters:
        lines += [
            "Excluded from agreement statistics because they have no usable scores: "
            + ", ".join(str(rater) for rater in empty_raters) + ".",
            "",
        ]
    if len(active_raters) < 2:
        return lines + ["Fewer than two reviewers have usable scores; between-rater agreement cannot be estimated.", ""]

    # Empty reviewer columns must not turn every otherwise complete panel row missing.
    band_index = {name: position for position, name in enumerate(BANDS)}
    band_matrix = wide_band[active_raters].apply(lambda column: column.map(band_index)).to_numpy(dtype=float)
    reba_matrix = wide_reba[active_raters].to_numpy(dtype=float)
    complete_band = int((~np.isnan(band_matrix).any(axis=1)).sum())
    paired_band = int(((~np.isnan(band_matrix)).sum(axis=1) >= 2).sum())
    complete_reba = int((~np.isnan(reba_matrix).any(axis=1)).sum())

    measured: list[tuple[str, float, str]] = [
        ("Fleiss' kappa (band)", fleiss_kappa(band_matrix), f"{complete_band} complete panel frames"),
        (
            "Krippendorff's alpha (band, ordinal)",
            krippendorff_alpha(band_matrix, "ordinal"),
            f"{paired_band} frames with at least two scores",
        ),
        ("ICC(2,1) on the REBA total", icc_two_way_agreement(reba_matrix), f"absolute agreement; {complete_reba} complete panel frames"),
    ]
    panel = [
        {
            "statistic": name,
            "value": round(value, 3) if np.isfinite(value) else None,
            "note": note,
            "reading": interpret(value),
        }
        for name, value, note in measured
    ]
    lines += [md_table(pd.DataFrame(panel)), ""]
    lines += [
        "_These estimates describe agreement among the contributing reviewers. They "
        "provide context for reference uncertainty, but do not impose a strict upper "
        "bound on pipeline agreement with their median consensus. Qualitative bands "
        "are descriptive aids, not instrument-specific acceptance thresholds._",
        "",
        "### Pairwise, for reference",
        "",
    ]

    rows = []
    for left, right in combinations(active_raters, 2):
        rows.append(
            {
                "pair": f"{left} vs {right}",
                "n": int((wide_band[left].notna() & wide_band[right].notna()).sum()),
                "band kappa": cohen_kappa(wide_band[left], wide_band[right], BANDS),
                "REBA qwk": quadratic_weighted_kappa(wide_reba[left], wide_reba[right]),
            }
        )
    table = pd.DataFrame(rows).round(3)
    lines += [md_table(table), ""]
    spread = table["band kappa"].dropna()
    if len(spread) > 1:
        lines += [
            f"Pairwise band kappa ranges {spread.min():.2f} to {spread.max():.2f}. A reviewer "
            f"who sits well below the rest is worth looking at before pooling them.",
            "",
        ]
    return lines


def _consensus(data: pd.DataFrame) -> pd.DataFrame:
    """Framewise medians of complete, scorable, non-repeat expert assessments.

    Totals and components are pooled separately over their available values. The
    band is derived from the median REBA total using :func:`band_of`; it is not
    a majority vote. RULA totals undefined for trunk extension remain missing.
    """

    first = data[~data["is_repeat"] & ~data["cannot_score"] & data["complete"]]
    grouped = first.groupby("frame_id")
    consensus = grouped.agg(
        expert_reba_final=("expert_reba_final", "median"),
        expert_rula_final=("expert_rula_final", "median"),
        n_raters=("rater_id", "nunique"),
        n_reba_raters=("expert_reba_final", "count"),
        n_rula_raters=("expert_rula_final", "count"),
    )
    for name in REBA_COMPONENTS:
        consensus[f"expert_reba_{name}"] = grouped[f"expert_reba_{name}"].median()
    for name in RULA_COMPONENTS:
        consensus[f"expert_rula_{name}"] = grouped[f"expert_rula_{name}"].median()
    consensus["expert_band"] = consensus["expert_reba_final"].map(band_of)

    truth_columns = [
        "band", "stratum", "participant", "task", "sampling_weight", "band_run_sec",
        "pipeline_reba_final", "pipeline_rula_final",
        *[f"pipeline_reba_{name}" for name in REBA_COMPONENTS],
        *[f"pipeline_rula_{name}" for name in RULA_COMPONENTS],
    ]
    truth = first.groupby("frame_id")[[c for c in truth_columns if c in first.columns]].first()
    merged = consensus.join(truth)
    merged["pipeline_band"] = merged["pipeline_reba_final"].map(band_of)
    return merged.reset_index()


def section_pipeline_agreement(data: pd.DataFrame) -> list[str]:
    lines = ["## 3. Pipeline vs expert agreement", ""]
    consensus = _consensus(data)
    if consensus.empty:
        return lines + ["No scorable frames.", ""]
    lines += [
        "Expert totals and components are separate medians across available, complete "
        "assessments. The expert band comes from the median REBA total, rounded to the "
        "nearest integer with ties to even (3.5 → 4, 7.5 → 8). Each statistic uses only "
        "frames with both required scores; RULA may have fewer contributing reviewers "
        "where a score is undefined for trunk extension.",
        "",
    ]

    def band_kappa(frame: pd.DataFrame) -> float:
        return cohen_kappa(frame["expert_band"], frame["pipeline_band"], BANDS)

    def band_kappa_weighted(frame: pd.DataFrame) -> float:
        return cohen_kappa(frame["expert_band"], frame["pipeline_band"], BANDS, frame["sampling_weight"])

    rows = []
    for label, subset, statistic, weighted in (
        ("All frames (balanced design)", consensus, band_kappa, False),
        ("All frames (prevalence-weighted)", consensus, band_kappa_weighted, True),
        ("Sustained stratum", consensus[consensus["stratum"] == "sustained"], band_kappa, False),
        ("Transient stratum", consensus[consensus["stratum"] == "transient"], band_kappa, False),
    ):
        pair_mask = subset["expert_band"].notna() & subset["pipeline_band"].notna()
        if weighted:
            pair_mask &= (
                subset["sampling_weight"].notna()
                & np.isfinite(subset["sampling_weight"])
                & (subset["sampling_weight"] > 0)
            )
        subset = subset[pair_mask]
        if subset.empty:
            continue
        value = statistic(subset)
        low, high = bootstrap_ci(subset, statistic)
        agreement = exact_agreement(
            subset["expert_band"], subset["pipeline_band"],
            subset["sampling_weight"] if weighted else None,
        )
        rows.append(
            {
                "subset": label,
                "n": len(subset),
                "exact agreement": round(agreement, 3),
                "kappa": round(value, 3) if np.isfinite(value) else None,
                "95% CI": f"{low:.2f} – {high:.2f}" if np.isfinite(low) else "—",
                "reading": interpret(value),
            }
        )
    lines += [md_table(pd.DataFrame(rows)), ""]
    lines += [
        "The prevalence-weighted row applies the sampling weights to both kappa and "
        "exact agreement. Confidence intervals resample participants as clusters; "
        "they condition on the returned reviewer panel and do not measure uncertainty "
        "from selecting different reviewers. An em dash means not estimable.",
        "",
    ]

    for instrument in ("reba", "rula"):
        lines += [f"### {instrument.upper()} total", ""]
        rows = []
        for label, subset in (
            ("All frames", consensus),
            ("Sustained", consensus[consensus["stratum"] == "sustained"]),
            ("Transient", consensus[consensus["stratum"] == "transient"]),
        ):
            expert_column = f"expert_{instrument}_final"
            pipeline_column = f"pipeline_{instrument}_final"
            subset = subset[subset[expert_column].notna() & subset[pipeline_column].notna()]
            if subset.empty:
                continue
            difference = subset[pipeline_column] - subset[expert_column]
            rows.append(
                {
                    "subset": label,
                    "n": len(subset),
                    "qwk": round(quadratic_weighted_kappa(subset[expert_column], subset[pipeline_column]), 3),
                    "bias (pipeline − expert)": round(float(difference.mean()), 2),
                    "mean abs diff": round(float(difference.abs().mean()), 2),
                    "within ±1": round(float((difference.abs() <= 1).mean()), 3),
                }
            )
        lines += [md_table(pd.DataFrame(rows)) if rows else "No matched score pairs.", ""]

    sustained = consensus[consensus["stratum"] == "sustained"]
    transient = consensus[consensus["stratum"] == "transient"]
    if not sustained.empty and not transient.empty:
        gap = band_kappa(sustained) - band_kappa(transient)

        def gap_statistic(frame: pd.DataFrame) -> float:
            left = frame[frame["stratum"] == "sustained"]
            right = frame[frame["stratum"] == "transient"]
            if left.empty or right.empty:
                return float("nan")
            return band_kappa(left) - band_kappa(right)

        low, high = bootstrap_ci(consensus, gap_statistic)
        interval = f" (95% CI {low:+.2f} to {high:+.2f})" if np.isfinite(low) else ""
        contrast = (
            f"Sustained minus transient band kappa is {gap:+.3f}{interval}."
            if np.isfinite(gap) else "The band-kappa difference is not estimable."
        )
        lines += [
            f"**Stratum contrast.** {contrast} This is an observed association, not an "
            "isolated causal effect of pose instability. Differences in posture mix, "
            "band prevalence, task, and participant can also affect the contrast.",
            "",
            "Interpret the estimate alongside its participant-cluster interval. The "
            "sample size alone does not establish statistical power, and an interval "
            "containing zero is compatible with either direction of association.",
            "",
        ]

    lines += ["### Confusion, expert (rows) vs pipeline (columns)", ""]
    confusion = pd.crosstab(consensus["expert_band"], consensus["pipeline_band"]).reindex(
        index=list(BANDS), columns=list(BANDS), fill_value=0
    )
    lines += [md_table(confusion, index=True), ""]
    return lines


def section_components(data: pd.DataFrame) -> list[str]:
    lines = ["## 4. Where the disagreement lives (per component)", ""]
    consensus = _consensus(data)
    rows = []
    for instrument, components in (("reba", REBA_COMPONENTS), ("rula", RULA_COMPONENTS)):
        for name in components:
            expert = consensus.get(f"expert_{instrument}_{name}")
            pipeline = consensus.get(f"pipeline_{instrument}_{name}")
            if expert is None or pipeline is None:
                continue
            mask = expert.notna() & pipeline.notna()
            if mask.sum() < 3:
                continue
            difference = pipeline[mask].astype(float) - expert[mask].astype(float)
            rows.append(
                {
                    "instrument": instrument.upper(),
                    "component": name.replace("_", " "),
                    "n": int(mask.sum()),
                    "exact": round(float((difference == 0).mean()), 3),
                    "qwk": round(quadratic_weighted_kappa(expert[mask], pipeline[mask]), 3),
                    "bias": round(float(difference.mean()), 2),
                    "mean abs diff": round(float(difference.abs().mean()), 2),
                }
            )
    if not rows:
        return lines + ["No component scores available to compare.", ""]
    table = pd.DataFrame(rows).sort_values(["instrument", "mean abs diff"], ascending=[True, False])
    lines += [md_table(table), ""]
    # The table is grouped by instrument for reading; the worst component is whichever
    # deviates most overall, which is not the first row of a grouped sort.
    worst = table.loc[table["mean abs diff"].idxmax()]
    direction = "over" if worst["bias"] > 0 else "under"
    lines += [
        f"_Largest deviation: **{worst['instrument']} {worst['component']}**, "
        f"{worst['mean abs diff']:.2f} points on average, with the pipeline scoring it "
        f"{direction} the reviewers by {abs(worst['bias']):.2f}. That is where to look first._",
        "",
    ]
    return lines


def section_coverage(data: pd.DataFrame) -> list[str]:
    lines = ["## 5. Data quality and coverage", ""]
    total = len(data)
    unscorable = data[data["cannot_score"]]
    lines += [
        f"- Rating rows returned: **{total}** from {data['rater_id'].nunique()} rater(s)",
        f"- Marked unscorable: **{len(unscorable)}** ({100 * len(unscorable) / max(1, total):.1f}%)",
        f"- Incomplete: **{int((~data['complete']).sum())}**",
    ]
    if not unscorable.empty and "stratum" in unscorable.columns:
        by_stratum = unscorable["stratum"].value_counts().to_dict()
        lines.append(f"- Unscorable by stratum: {by_stratum}")
    if data["confidence"].notna().any():
        confidence = pd.to_numeric(data["confidence"], errors="coerce")
        lines.append(f"- Mean confidence: **{confidence.mean():.2f}** of 5")
        if "stratum" in data.columns:
            by_stratum = data.assign(c=confidence).groupby("stratum")["c"].mean().round(2).to_dict()
            lines.append(f"- Mean confidence by stratum: {by_stratum}")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


RETURN_SUFFIXES = (".json", ".xlsx", ".xlsm")


def resolve_rating_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        candidate = Path(pattern)
        if candidate.is_dir():
            for suffix in RETURN_SUFFIXES:
                paths.extend(sorted(candidate.glob(f"*{suffix}")))
        else:
            paths.extend(Path(match) for match in sorted(glob.glob(pattern)))
    # Excel leaves a lock file beside an open workbook; it is not a return.
    paths = [path for path in paths if not path.name.startswith("~$")]
    unique = sorted({path.resolve() for path in paths if path.is_file()})
    if not unique:
        raise SystemExit(f"no rating files matched {patterns}")
    return unique


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--ratings", nargs="+", required=True,
        help="returned .json and/or .xlsx files, a glob, or a directory containing them",
    )
    parser.add_argument("--key", type=Path, default=DEFAULT_SAMPLE_DIR / "ground_truth_key.csv")
    parser.add_argument("--output", type=Path, default=None, help="write the report to a Markdown file")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--rater-ids-from-filenames", action="store_true",
        help="explicitly use Rxx from scoring_workbook_Rxx.xlsx filenames instead of stored reviewer IDs",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )
    if not args.key.is_file():
        LOGGER.error("ground truth key not found: %s", args.key)
        return 1

    ratings = load_ratings(
        resolve_rating_paths(args.ratings), rater_ids_from_filenames=args.rater_ids_from_filenames
    )
    data = merge_truth(ratings, args.key)

    lines = ["# Expert validation — agreement report", ""]
    lines += section_within_rater(data)
    lines += section_between_rater(data)
    lines += section_pipeline_agreement(data)
    lines += section_components(data)
    lines += section_coverage(data)
    report = "\n".join(lines)

    if args.output:
        args.output.write_text(report, encoding="utf-8")
        LOGGER.info("Wrote %s", args.output)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

