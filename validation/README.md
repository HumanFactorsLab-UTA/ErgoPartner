# Blind expert validation

Prepare a blinded set of frames from your own processed studies, collect independent RULA and REBA ratings, and compare those ratings with the pipeline. The example below uses five reviewer identifiers.

```bash
python -m validation.sample_frames --allow-seek                              # draw the frames
python -m validation.assemble_delivery --raters R01,R02,R03,R04,R05          # build everything
python -m validation.analyze_ratings --ratings validation_sample/delivery/3_when_ratings_return/returned --key validation_sample/delivery/2_keep_private/ground_truth_key.csv --output report.md
```

Each delivery uses a new directory. To prepare another package from the same sample,
choose a new path with `--delivery-dir validation_sample/delivery_round_2`. Keep returned
ratings with the delivery that produced the workbooks. Its `START_HERE.html` contains
the analysis command for those ratings and their answer key. The standalone
`build_expert_package` command also requires a new `--package-dir` and unused ZIP names.
It reuses identical presentation maps in the sample directory. Use a separate delivery
for a different reviewer list or order so each round keeps its corresponding maps.

If confirmed reviewer identities follow `scoring_workbook_Rxx.xlsx` filenames, add
`--rater-ids-from-filenames` to override an incorrect ID stored inside a workbook.
The import records the original ID and does not edit the workbook. Duplicate
reviewer/item pairs stop the run. Blank or incomplete assessments remain in coverage
counts but do not contribute to agreement statistics. Both REBA and RULA total
agreement are reported using the available complete assessments.

Sampling time depends on the available recordings. Use `--no-images` to inspect the sampling design before decoding frames.

### Comparing two pipeline versions on the returned expert panel

```bash
python -m validation.paired_benchmark --reference reference.csv --baseline before.csv --candidate after.csv --output comparison.json
```

Each prediction CSV needs `frame_id,participant,study,task,frame_number,reba_final,rula_final`.
Supply one row for every reference frame, with a blank score when the run abstains. The
comparison checks every source identity and rejects omitted, extra, or duplicated frames.
The output must be a new file. It contains input checksums and per-instrument coverage,
exact and within-one agreement, signed bias, MAE, quadratic-weighted Cohen's kappa and
official risk-level confusion matrices. Each candidate-minus-baseline estimate uses frames
scored by both systems and has a paired participant-bootstrap percentile interval. The
report also shows each system's results over its own available cases and the IDs of gained
or lost cases. Shares and their changes are expressed as fractions.
Subgroup rows compare the paired cases within each task. When the reference supplies
`sampling_weight`, a separate descriptive sensitivity analysis weights paired MAE and
exact/within-one agreement by those original design weights. Frames and expert totals are
identical in both analyses. The weights describe the sampled strata and do not establish
accuracy on a new workforce.

Supply the reference export explicitly when running this comparison on your study. The tool uses the median of eligible reviewer totals for each frame. The study-specific reference export used in development is not distributed.

The second assembles `validation_sample/delivery/`, split by who may see what:

```
delivery/
  START_HERE.html            open this first
  1_send_to_reviewers/       expert_package.zip — share this one link with all five
  2_keep_private/            the answer key, the manifest, the design record
  3_when_ratings_return/     where returned files go
```

By default reviewers receive the same frame order. The analysis joins ratings on `frame_id`. Use `--randomise-order` to create a separate sequence and workbook for each reviewer.

Reviewers work **either** in the browser page **or** in the Excel workbook — both are in
the zip. They ask the same questions in the same words, and both are read back by
`validation/scoring.py`, which recomposes REBA and RULA from the raw observations using
this project's own calculators. Whichever a reviewer picks, the numbers that reach the
analysis are produced by identical code, and returns can be mixed freely in one run.

`--repeats 6` shows six frames a second time, so each reviewer's agreement with
*themselves* can be measured — the ceiling on every other figure in the report. It is off
by default because it makes the set longer than the 60 frames that were drawn.

---

## What the study is asking

> Do the pipeline's REBA/RULA assessments agree with expert human raters?

Everything below exists to keep that question answerable rather than merely answered.

---

## 1. The sampling design

### Why not simply balance risk and task

The obvious design — 20 Low / 20 Medium / 20 High, 20 per task, every participant
represented — does not survive contact with the data. Measured on the 10 fps results:

| | Low | Medium | High |
|---|---|---|---|
| task1_niosh_lifting | 3.4% | 98.1% | 78.7% |
| task2_reba_stocking | 20.3% | 98.1% | 19.1% |
| task3_rula_screwdriving | 77.0% | 97.4% | **0.0%** |

Those percentages are how often a frame's REBA band survives being compared against a
rolling median of its own neighbours. In `task3_rula_screwdriving` there are 39 "High"
frames out of 6,250, they occur in runs with a **median length of one frame**, and not one
of them survives smoothing. They are pose-estimation transients, not postures.

A design that insists on filling every task × band cell must therefore go hunting in
exactly the cells where only artefacts live. It would hand raters the pipeline's worst
glitches, and the resulting disagreement would be reported as an assessment failure.

### What is done instead

Frames are stratified on **band × stability**:

- **sustained** — the REBA band is unchanged across ±`--stability-window` samples
  (default ±2, i.e. it holds for 0.5 s at 10 fps);
- **transient** — it is not.

Default quota, 60 frames:

| | Low | Medium | High |
|---|---|---|---|
| sustained | 12 | 12 | 12 |
| transient | 8 | 8 | 8 |

The sustained stratum answers the actual research question. The transient stratum
**measures what pose instability costs**, separately, instead of letting it contaminate
the headline number. The gap between the two is a result worth reporting.

`band_run_sec` and `band_run_frames` are exported per frame, so the stability threshold
can be varied in sensitivity analysis rather than taken on trust.

### Hard versus soft constraints

Only the (stratum, band) counts are exact. Task balance, per-participant coverage and
spreading frames across distinct recordings are expressed as **costs in a min-cost flow**,
not as quotas. A design that demands them exactly reports itself infeasible the moment one
task genuinely cannot supply a cell — which is precisely what happens here, since seated
screwdriving has no sustained high-risk posture at all. The solver relaxes task balance one
frame at a time and states in the manifest what it had to give up.

### Prevalence

The pool is 77% Medium, 16% High, 7% Low. A balanced design deliberately over-samples the
rare bands, so a kappa computed on it does **not** describe the population — kappa is
prevalence-sensitive. Each frame therefore carries a `sampling_weight` (inverse inclusion
probability), and `analyze_ratings` reports balanced and prevalence-weighted estimates side
by side.

---

## 2. The expert package

`build_expert_package` produces one self-contained zip per rater, each holding the 60
composites, the browser rating page, and that rater's Excel workbook.

### Two ways in, one set of answers

| | Browser (`index.html`) | Excel (`scoring_workbook_R01.xlsx`) |
|---|---|---|
| Install needed | none | Excel or LibreOffice |
| Image | large, one camera view zoomable in place | embedded per sheet, full-size via link |
| Progress | autosaved in the browser | saved with the file |
| Returns | JSON + CSV | the workbook itself |

The workbook is one sheet per image in presentation order (`F001`, `F002`, …), with an
`Index` sheet for jumping around and seeing what is left, and a `Results` sheet holding a
flat one-row-per-image table. Drop-downs restrict every answer to a listed option, and the
score is composed live by in-sheet formulas whose lookup tables are copied out of
`ergonomics/reba.py` and `ergonomics/rula.py` at build time.

Those formulas are for the rater's benefit only. The analysis reads the raw drop-down
answers from each sheet and recomposes the score in Python, so a workbook returned from
LibreOffice or Google Sheets — or one whose formulas were never recalculated — is still
read correctly.

Each rater's workbook has **its own image order**, because a workbook's sheet order is
fixed when the file is written. The browser page shuffles per rater from the rater ID.
Either way, no two raters meet the images in the same sequence.

The browser page is still the better tool — bigger image, per-panel zoom, no risk of a
formula being disturbed — and the README in the package says so, while leaving the choice
to the rater.

### Raters record observations, not totals

The rater selects the posture band they observe for each body segment. The page composes
REBA and RULA from those selections. This is not a convenience:

- **It removes clerical error.** Hand-composing a score through three lookup tables is a
  known source of mistakes in reliability studies, and a table slip would otherwise be
  recorded as genuine disagreement with the pipeline.
- **It isolates the measurement.** Expert and pipeline are composed by identical tables, so
  every remaining difference is postural judgement.
- **It keeps the diagnosis.** Segment-level answers say *which* body part disagrees. A
  grand score cannot be decomposed afterwards.

Each joint is asked about **once**, using the union of REBA's and RULA's cut points, and
each instrument derives its own component from that single answer. Asking twice would let
one rater give two different accounts of the same body.

The lookup tables are copied out of `ergonomics/reba.py` and `ergonomics/rula.py` **at build
time**, so they cannot drift from the code under test.

### Subscores the rater is not asked to judge

`load_force`, `coupling`, `activity` (REBA) and `force_load`, `muscle_use` (RULA) are shown
per frame as **givens**, taken from what the pipeline actually used. A still frame cannot
convey handled load or coupling, and activity and muscle use depend on what happens either
side of the instant shown. Letting raters estimate them would measure disagreement about
task facts rather than about posture.

This matters concretely: most frames were scored under the `standing_light_work` profile
(`load_kg = 0`, `coupling = good`), but a few carry real values (1 kg / fair, 10 kg / good).
The givens are therefore per-frame, not a blanket instruction.

### Blinding

The page receives only `frame_id`, the image, and the givens. Band, stratum, participant,
task and every pipeline score stay in `ground_truth_key.csv`, which is **not** part of the
package. File order is shuffled before naming, so the sequence carries no signal, and each
rater is dealt their own permutation from their rater ID.

### Within-rater reliability

Six frames (`--repeats`) are shown a second time, at least 12 items away from the first
showing — enforced per rater, inside the page, because each rater's order is generated
there. Without this there is no way to separate a rater's disagreement with the pipeline
from their disagreement with themselves.

### Reading the frames

The composite is a wide three-panel strip, so the buttons under the image (or keys
<kbd>1</kbd>–<kbd>4</kbd>) enlarge one camera view in place; clicking the image opens
full-screen zoom and pan. Panel boundaries are recorded when the composite is written, since
the three cameras need not share an aspect ratio.

---

## 3. Analysis

`analyze_ratings` reports, in order:

1. **within-rater consistency** from the repeats — the ceiling on everything else;
2. **between-rater agreement** — the reference standard's own reliability;
3. **pipeline vs expert**, balanced, prevalence-weighted, and per stratum;
4. **per-component agreement** — where the disagreement lives;
5. **coverage** — unscorable frames and confidence.

Confidence intervals bootstrap over **participants**, not frames: several frames come from
one person and are not independent observations.

---

## Files

| | |
|---|---|
| `pool.py` | discovery, banding, stability |
| `allocate.py` | matrix enumeration + min-cost flow |
| `sample_frames.py` | CLI, compositing, exports |
| `scoring.py` | **observations → REBA/RULA; the one place scores are composed** |
| `build_expert_package.py` | assembles the packages |
| `expert_app_template.html` | the browser rating page |
| `build_workbook.py` | the Excel workbook, and reading one back |
| `analyze_ratings.py` | agreement statistics |

Outputs land in `validation_sample/`:

| | |
|---|---|
| `images/` | the 60 composites |
| `ground_truth_key.csv` | **withhold until scoring is complete** |
| `sampling_manifest.json` | settings, provenance, realised design |
| `package_item_ledger.csv` | presentation item → frame, including repeats |
| `workbook_sheet_map.csv` | which workbook sheet holds which frame, per rater |
| `expert_package/` | the assembled package |
| `expert_package_<rater>.zip` | what each rater receives |

---

## How the two rating paths were checked against each other

```bash
pytest tests/test_validation_scoring.py tests/test_validation_design.py \
       tests/test_validation_package.py tests/test_validation_analysis.py
```

179 tests hold the invariants below in place. The ones that matter most: every posture
band is scored the way the calculators score it; the workbook's flattened lookup tables
reproduce the calculators *under the row arithmetic its formulas use* (an off-by-one
there would still recalculate cleanly and give plausible scores); nothing reaching a
rater names a band, stratum, participant, task or pipeline score; and the same
judgements delivered through the page and through the workbook arrive at the analysis
identically.

The claim that the browser page, the workbook, and the pipeline all compose scores the
same way was also checked once against the live artefacts:

| Check | Result |
|---|---|
| Lookup tables embedded in `index.html` vs the calculators | exhaustive, 0 mismatches |
| Every REBA/RULA table lookup path, both instruments | 0 mismatches |
| Browser band→component derivations vs the calculators | 112/112 combinations |
| `scoring.py` vs the browser derivations | 112/112 combinations |
| Excel formulas vs `scoring.py`, driven through real Excel | 1254 cells, 0 mismatches, 0 error cells |
| Workbook filled in Excel → saved → read back | observations, scores and flags all round-trip |
| Nothing in a rater's zip discloses band, stratum, participant, task or a pipeline score | clean |

`openpyxl` is needed only to build or read a workbook; the sampler and the analysis run
without it.

## Notes and limitations

- **Only the 10 fps variant is complete.** `results/native/` holds one participant.
  `--fps` defaults to the variant with the widest coverage and logs its choice.
- **Archived runs are excluded.** Superseded runs are retired as `.<task>.archive-<tag>`;
  a plain directory scan picks them up and they arrive as extra tasks competing for quota.
- **59 recordings, not 60.** Participant 001 has no `task3_rula_screwdriving`.
- **`pose_mean_visibility` is not an image-quality gate.** Its minimum here is 0.744 and its
  median 0.9999 — MediaPipe's self-reported visibility is near-saturated, so a threshold on
  it filters nothing. Raters flag unusable frames instead, which is the real check.
- **Frames are admitted only if the pipeline scored them with REBA.** Admitting a frame on
  its RULA score instead rescues exactly those frames REBA rejected for
  `insufficient_pose_quality`, and labels them with a different instrument.
- **Every REBA score in this dataset is `valid_with_limitations`**, and `knee_flexion_deg` is
  missing on some recordings. That belongs in the paper's limitations.
- **Faces are visible.** The protocol blinds the assessment, not the subject.
- **A per-rater zip is ~40 MB**, mostly the 60 composites. Share it by link rather than
  email. The workbook embeds its own downscaled copies, so it opens without the folder,
  but the "open full-size image" links only resolve when `images/` is beside it.
- **Excel raters get a fixed order per file.** It differs between raters, but a rater who
  restarts gets the same sequence; the browser page has the same property.
