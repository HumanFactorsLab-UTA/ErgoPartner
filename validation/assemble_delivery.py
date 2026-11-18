"""Assemble the final delivery: what goes to the reviewers, and what stays with you.

Everything the study needs ends up under one folder, split by who is allowed to see
it.  The split is the point: the reviewer package must not contain the pipeline's
scores, and the answer key must not travel with it.

    delivery/
      START_HERE.html          what to do, in order
      1_send_to_reviewers/     the package, and one workbook per reviewer
      2_keep_private/          the answer key, the manifest, the design record
      3_when_ratings_return/   where returned files go, and how to analyse them

Usage
-----
    python -m validation.assemble_delivery --raters R01,R02,R03,R04,R05
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
from datetime import date
from html import escape
from pathlib import Path

import pandas as pd

from validation import build_expert_package, build_reference_doc
from validation.package_output import publish_directory, require_new_output

LOGGER = logging.getLogger("validation.assemble_delivery")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLE_DIR = PROJECT_ROOT / "validation_sample"
DEFAULT_DELIVERY_DIR = PROJECT_ROOT / "validation_sample" / "delivery"

DEFAULT_RATERS = ["R01", "R02", "R03", "R04", "R05"]

PRIVATE_FILES = (
    "ground_truth_key.csv",
    "sampling_manifest.json",
)


class DeliveryError(RuntimeError):
    """Raised when the delivery cannot be assembled."""


def _start_here(raters: list[str], frames: int, repeats: int, analysis_command: str) -> str:
    reviewer_rows = "\n".join(
        f"        <tr><td><code>{r}</code></td>"
        f"<td>scoring_workbook_{r}.xlsx</td><td></td></tr>"
        for r in raters
    )
    repeat_note = (
        f"<p>{repeats} of the {frames + repeats} images are deliberate repeats, included so "
        f"each reviewer's agreement with <em>themselves</em> can be measured.</p>"
        if repeats
        else "<p>Each reviewer scores the same 60 frames, in the same order.</p>"
    )
    return f"""<meta charset="utf-8">
<title>Expert validation — start here</title>
<style>
  :root {{ --ink:#15181d; --muted:#5d6470; --line:#dfe3e9; --accent:#1f4e79;
           --panel:#fff; --bg:#f6f7f9; --warn:#9c0006; --warnbg:#f8d7da; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --ink:#e9ecf1; --muted:#9aa3b1; --line:#2b313b; --accent:#6d95ee;
             --panel:#1a1e25; --bg:#12151a; --warn:#ef6f61; --warnbg:#3a1f22; }}
  }}
  * {{ box-sizing:border-box }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
         font:16px/1.65 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif }}
  .wrap {{ max-width:820px; margin:0 auto; padding:40px 24px 80px }}
  h1 {{ font-size:1.9rem; margin:0 0 .2em; line-height:1.2 }}
  h2 {{ font-size:1.2rem; margin:2em 0 .5em; color:var(--accent) }}
  .lede {{ color:var(--muted); margin:0 0 2em; font-size:1.05rem }}
  .card {{ background:var(--panel); border:1px solid var(--line); border-radius:12px;
          padding:20px 24px; margin:0 0 18px }}
  .warn {{ border-color:var(--warn); background:var(--warnbg) }}
  .warn strong {{ color:var(--warn) }}
  code, pre {{ font-family:ui-monospace,Consolas,Menlo,monospace; font-size:.9em }}
  pre {{ background:var(--bg); border:1px solid var(--line); border-radius:8px;
        padding:12px 14px; overflow-x:auto }}
  table {{ border-collapse:collapse; width:100%; margin:.6em 0; font-size:.94rem }}
  th, td {{ border:1px solid var(--line); padding:7px 10px; text-align:left }}
  th {{ background:var(--bg) }}
  ol.steps {{ padding-left:1.2em }} ol.steps > li {{ margin:.9em 0 }}
  ul.tight li {{ margin:.3em 0 }}
  .muted {{ color:var(--muted) }}
  .step-n {{ display:inline-block; background:var(--accent); color:#fff; border-radius:99px;
            width:26px; height:26px; text-align:center; line-height:26px; font-size:.85rem;
            font-weight:700; margin-right:8px }}
</style>
<div class="wrap">

<h1>Expert validation — start here</h1>
<p class="lede">Five reviewers score {frames} still frames. Their ratings become the reference
standard the automated assessment is measured against. Built {date.today().isoformat()}.</p>

<div class="card warn">
  <strong>Before anything else:</strong> send only what is in
  <code>1_send_to_reviewers/</code>. The folder <code>2_keep_private/</code> holds the
  pipeline's own scores for these exact frames. If a reviewer sees it, the study is void.
</div>

<h2><span class="step-n">1</span>Send the package</h2>
<div class="card">
  <p>Share <code>1_send_to_reviewers/expert_package.zip</code> with all five reviewers —
  one link is enough. Tell each of them which rater ID is theirs:</p>
  <table>
    <tr><th>Rater ID</th><th>Their workbook</th><th>Name</th></tr>
{reviewer_rows}
  </table>
  <p class="muted">Fill in the third column for your own records. The reviewers do not
  need it.</p>
  <p>Inside the zip a reviewer finds a <code>README.txt</code> and two ways to work. They
  pick one:</p>
  <ul class="tight">
    <li><b>index.html</b> — opens in any browser, nothing to install. Bigger image, and
    a single camera view can be enlarged in place.</li>
    <li><b>scoring_workbook_&lt;ID&gt;.xlsx</b> — one sheet per image, drop-downs, score
    calculated as they go.</li>
  </ul>
  <p>Both ask identical questions and are scored by identical code, so the two can be
  mixed across your panel without affecting anything.</p>
  {repeat_note}
  <p class="muted">About 2–3 minutes per image, so 2–3 hours in total. Neither route has
  to be finished in one sitting.</p>
</div>

<h2><span class="step-n">2</span>While you wait</h2>
<div class="card">
  <p>Open <code>2_keep_private/ground_truth_reference.xlsx</code>. Its first sheet explains
  the study; <b>Frames</b> lists every frame with its image, the pipeline's REBA and RULA
  scores, and each component subscore; <b>Design</b> shows how the {frames} frames are
  composed across risk band, task and participant.</p>
  <p>This is the document to read when you want to check a particular frame by eye, or to
  describe the sample in a methods section.</p>
</div>

<h2><span class="step-n">3</span>When the ratings come back</h2>
<div class="card">
  <p>Drop every returned file into <code>3_when_ratings_return/returned/</code> — the
  <code>.json</code> exports from the rating page and the <code>.xlsx</code> workbooks can
  sit side by side. Then run:</p>
  <pre>{escape(analysis_command)}</pre>
  <p>The report gives, in order: agreement between the reviewers themselves; the pipeline
  against the reviewer consensus, reported balanced, re-weighted to the population, and
  split by stratum; and a per-component breakdown showing which body segment any
  disagreement lives in.</p>
</div>

<h2>What the reviewers are and are not judging</h2>
<div class="card">
  <p>They record what they observe for each body segment. REBA and RULA are then composed
  from those observations using the same lookup tables the pipeline uses. Hand-composing a
  score through three lookup tables is a known source of error in reliability studies, and
  a table slip would otherwise be recorded as genuine disagreement.</p>
  <p>Load, coupling and activity are <em>given</em> to them per frame, set to whatever the
  pipeline used. A still frame cannot convey handled load, and activity depends on what
  happens either side of the instant shown. Letting reviewers estimate them would measure
  disagreement about task facts instead of about posture.</p>
  <p>So every difference that survives is a difference in postural judgement. That is the
  thing worth reporting.</p>
</div>

<h2>Two things a reader will ask</h2>
<div class="card">
  <p><b>Why are the frames split into "sustained" and "transient"?</b> A sustained frame's
  risk band holds across the frames either side of it, so it describes a posture that was
  actually held. A transient one lasts an instant. On this dataset the rare risk bands are
  overwhelmingly transient — in the screwdriving task the median run of consecutive
  high-risk frames is a single frame. A sample balanced on risk alone would therefore have
  handed reviewers the pipeline's momentary pose glitches and counted the disagreement
  against its posture assessment. Reporting the two separately keeps that honest, and the
  gap between them is itself a result.</p>
  <p><b>Why is the sample balanced when the real data is not?</b> Because rare bands need
  enough frames to say anything about. The pool is roughly three-quarters medium risk; a
  balanced sample over-represents the extremes on purpose. Every frame carries a weight
  that undoes this, and the report gives both the balanced and the re-weighted figure —
  kappa is prevalence-sensitive, so the balanced number alone would not describe the
  study.</p>
</div>

<h2>Folder map</h2>
<div class="card">
<pre>delivery/
  START_HERE.html            this page
  1_send_to_reviewers/
    expert_package.zip       <b>&larr; share this</b>
    expert_package/          the same thing, unzipped
  2_keep_private/
    ground_truth_reference.xlsx   the answer key, readable
    ground_truth_key.csv          the answer key, machine-readable
    sampling_manifest.json        settings and provenance
    design_record.md              how the sample was drawn
  3_when_ratings_return/
    returned/                put the returned files here</pre>
</div>

</div>
"""


def _design_record(sample_dir: Path, key: pd.DataFrame) -> str:
    manifest_path = sample_dir / "sampling_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    )
    design = manifest.get("design", {})
    provenance = manifest.get("pool_provenance", {})
    settings = manifest.get("settings", {})
    counts = provenance.get("filter_counts", {})

    bands = ["Low", "Medium", "High"]
    by_stratum = pd.crosstab(key["stratum"], key["band"]).reindex(columns=bands, fill_value=0)
    by_task = pd.crosstab(key["task"], key["band"]).reindex(columns=bands, fill_value=0)

    def as_table(frame: pd.DataFrame, first: str) -> str:
        head = f"| {first} | " + " | ".join(frame.columns) + " |"
        rule = "|" + "|".join(["---"] * (len(frame.columns) + 1)) + "|"
        rows = [
            f"| {index} | " + " | ".join(str(int(v)) for v in values) + " |"
            for index, values in frame.iterrows()
        ]
        return "\n".join([head, rule, *rows])

    notes = design.get("notes") or []
    note_lines = "\n".join(f"- {note}" for note in notes) or "- None; every target was met."

    return f"""# How this sample was drawn

Generated {date.today().isoformat()} from `sampling_manifest.json`. This is the record to
cite in a methods section.

## Source

| | |
|---|---|
| Processing variant | `{provenance.get('fps_variant', '—')}` |
| Recordings | {provenance.get('recordings', '—')} |
| Participants | {provenance.get('participants', '—')} |
| Frames scored by the pipeline | {counts.get('reba_scored', '—')} |
| Candidate pool after banding | {counts.get('banded', '—')} |
| Random seed | {settings.get('seed', '—')} |

Only frames the pipeline scored with REBA are candidates. Admitting a frame on its RULA
score instead would rescue exactly those frames REBA rejected for insufficient pose
quality, and would label them with a different instrument.

Archived runs — directories retired as `.<task>.archive-<tag>` — are excluded. A plain
directory scan picks them up, and because the name still parses as a task they arrive as
extra tasks competing for the quota.

## Strata

A frame is **sustained** when its risk band is unchanged across
±{settings.get('stability_half_window', '—')} samples either side of it, and **transient**
otherwise. `band_run_sec` is recorded per frame so this threshold can be varied in
sensitivity analysis rather than taken on trust.

{as_table(by_stratum, 'Stratum')}

## Composition

{as_table(by_task, 'Task')}

Participants represented: {key['participant'].nunique()} of
{provenance.get('participants', '—')}, contributing between
{key['participant'].value_counts().min()} and {key['participant'].value_counts().max()}
frames each, drawn from {key.groupby(['participant', 'task']).ngroups} distinct recordings.

## Where the design gave ground

{note_lines}

## Prevalence and weighting

The pool the frames were drawn from is
{provenance.get('band_prevalence', {})} by risk band. The sample is balanced on purpose,
so it does not resemble that. Each frame carries `sampling_weight`, the inverse inclusion
probability, and the agreement report gives both the balanced and the prevalence-weighted
estimate. Cohen's kappa is prevalence-sensitive, so a balanced-design kappa does not
describe the population.

## Limitations to disclose

- Every REBA score in this dataset carries `valid_with_limitations`; `knee_flexion_deg` is
  missing on some recordings.
- Load and coupling were assumed rather than measured on most recordings
  (`standing_light_work`: 0 kg, good coupling). Reviewers are given the same values, so the
  comparison isolates posture, but the assumption belongs in the limitations.
- `pose_mean_visibility` is not an image-quality gate — its minimum here is 0.744 and its
  median 0.9999. Reviewers flag unusable frames instead, which is the real check.
- Faces are visible in the composites. The protocol blinds the assessment, not the subject.
- One participant has no third task, so the sample is drawn from 59 recordings, not 60.
"""


def assemble(
    sample_dir: Path,
    delivery_dir: Path,
    raters: list[str],
    repeats: int,
    seed: int,
    randomise_order: bool,
) -> Path:
    key_path = sample_dir / "ground_truth_key.csv"
    if not key_path.is_file():
        raise DeliveryError(
            f"{key_path} not found — run `python -m validation.sample_frames --allow-seek` first"
        )
    require_new_output(delivery_dir)
    delivery_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=delivery_dir.parent, prefix=".delivery-") as temporary:
        staged = Path(temporary) / "delivery"
        _write_delivery(
            sample_dir, staged, raters, repeats, seed, randomise_order,
            pd.read_csv(key_path), _analysis_command(delivery_dir),
        )
        publish_directory(staged, delivery_dir)
    LOGGER.info("Delivery assembled at %s", delivery_dir)
    return delivery_dir


def _analysis_command(delivery: Path) -> str:
    def quote(path: Path) -> str:
        return "'" + str(path.resolve()).replace("'", "''") + "'"

    results = delivery / "3_when_ratings_return"
    return (
        f"python -m validation.analyze_ratings --ratings {quote(results / 'returned')} "
        f"--key {quote(delivery / '2_keep_private' / 'ground_truth_key.csv')} "
        f"--output {quote(results / 'agreement_report.md')}"
    )


def _write_delivery(
    sample_dir: Path, delivery_dir: Path, raters: list[str], repeats: int,
    seed: int, randomise_order: bool, key: pd.DataFrame, analysis_command: str,
) -> None:
    send = delivery_dir / "1_send_to_reviewers"
    private = delivery_dir / "2_keep_private"
    returned = delivery_dir / "3_when_ratings_return" / "returned"
    for folder in (send, private, returned):
        folder.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Building the reviewer package for %d reviewer(s)", len(raters))
    build_expert_package.write_package_files(
        sample_dir=sample_dir,
        package_dir=send / "expert_package",
        repeats=repeats,
        seed=seed,
        make_zip=True,
        raters=raters,
        randomise_order=randomise_order,
        ledger_dir=private,
    )

    LOGGER.info("Building the reference document")
    build_reference_doc.build(sample_dir, private / "ground_truth_reference.xlsx")

    for name in PRIVATE_FILES:
        source = sample_dir / name
        if source.is_file():
            shutil.copy2(source, private / name)
    (private / "design_record.md").write_text(_design_record(sample_dir, key), encoding="utf-8")

    (returned / "PUT_RETURNED_FILES_HERE.txt").write_text(
        "Drop every returned file here — .json exports from the rating page and .xlsx\n"
        "workbooks can sit side by side. From the project root in PowerShell, run:\n\n"
        f"  {analysis_command}\n",
        encoding="utf-8",
    )

    frames = int(key["frame_id"].nunique())
    (delivery_dir / "START_HERE.html").write_text(
        _start_here(raters, frames, repeats, analysis_command), encoding="utf-8"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument(
        "--delivery-dir", type=Path, default=DEFAULT_DELIVERY_DIR,
        help="new directory to create for this delivery",
    )
    parser.add_argument("--raters", default=",".join(DEFAULT_RATERS))
    parser.add_argument(
        "--repeats", type=int, default=build_expert_package.DEFAULT_REPEATS,
        help="frames shown twice, for within-reviewer consistency (default 0)",
    )
    parser.add_argument("--seed", type=int, default=build_expert_package.RANDOM_SEED)
    parser.add_argument(
        "--randomise-order", action="store_true",
        help="give each reviewer their own image order instead of one shared sequence",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )
    raters = [name.strip() for name in args.raters.split(",") if name.strip()]
    if not raters:
        LOGGER.error("no rater IDs given")
        return 1
    try:
        delivery = assemble(
            args.sample_dir, args.delivery_dir, raters, args.repeats, args.seed,
            args.randomise_order,
        )
    except (FileExistsError, DeliveryError, build_expert_package.PackageError,
            build_reference_doc.ReferenceError) as exc:
        LOGGER.error("%s", exc)
        return 1

    print(f"\nDelivery: {delivery}")
    print(f"  open {delivery / 'START_HERE.html'}")
    print(f"  share {delivery / '1_send_to_reviewers' / 'expert_package.zip'}")
    print(f"  keep  {delivery / '2_keep_private'} away from the reviewers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
