"""Build the study lead's reference document: what the pipeline scored, per frame.

``ground_truth_key.csv`` already holds every number this study will be judged on, but
a 40-column CSV is not something anyone reads.  This turns it into a workbook that can
be opened and understood: what each frame is, what the pipeline said about it and why,
and how the 60 frames are composed across risk band, stratum, task and participant.

**This file is the answer key.  It must not reach a reviewer.**  It is written outside
the reviewer package for that reason, and nothing here is ever copied into one.

Usage
-----
    python -m validation.build_reference_doc
"""

from __future__ import annotations

import argparse
import io
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

LOGGER = logging.getLogger("validation.build_reference_doc")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLE_DIR = PROJECT_ROOT / "validation_sample"

FONT = "Arial"
ACCENT = "1F4E79"
MUTED = "5F6672"
HEAD_FILL = PatternFill("solid", fgColor="1F4E79")
BAND_FILL = {
    "Low": PatternFill("solid", fgColor="D6EAD6"),
    "Medium": PatternFill("solid", fgColor="FFF2CC"),
    "High": PatternFill("solid", fgColor="F8D7DA"),
}
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

THUMB_WIDTH_PX = 460
REBA_COMPONENTS = ("trunk", "neck", "legs", "upper_arm", "lower_arm", "wrist")
RULA_COMPONENTS = ("upper_arm", "lower_arm", "wrist", "wrist_twist", "neck", "trunk", "legs")


class ReferenceError(RuntimeError):
    """Raised when the reference document cannot be built."""


def _title(cell: Any, size: int = 12) -> None:
    cell.font = Font(name=FONT, size=size, bold=True, color="FFFFFF")
    cell.fill = HEAD_FILL
    cell.alignment = Alignment(horizontal="left", vertical="center", indent=1)


def _header_row(sheet: Any, row: int, labels: list[str], widths: list[int]) -> None:
    for index, (label, width) in enumerate(zip(labels, widths, strict=True), start=1):
        cell = sheet.cell(row=row, column=index, value=label)
        cell.font = Font(name=FONT, size=9, bold=True, color="FFFFFF")
        cell.fill = HEAD_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BOX
        sheet.column_dimensions[get_column_letter(index)].width = width
    sheet.row_dimensions[row].height = 30


# ---------------------------------------------------------------------------
# Sheets
# ---------------------------------------------------------------------------


def _how_to_use(book: Workbook, key: pd.DataFrame, manifest: dict[str, Any]) -> None:
    sheet = book.create_sheet("How to use this", 0)
    sheet.sheet_view.showGridLines = False
    sheet.column_dimensions["A"].width = 3
    sheet.column_dimensions["B"].width = 108

    design = manifest.get("design", {})
    provenance = manifest.get("pool_provenance", {})
    settings = manifest.get("settings", {})

    blocks: list[tuple[str, str]] = [
        ("h", "Blind expert validation — reference copy"),
        ("warn", "THIS FILE IS THE ANSWER KEY. Do not send it, or anything derived from it, "
                 "to a reviewer until every rating has been returned."),
        ("h2", "What this study is doing"),
        ("p", "Five reviewers independently score 60 still frames drawn from the processed "
              "study footage. Each frame shows one instant from three synchronised cameras. "
              "The reviewers do not see the automated system's scores, the participant, the "
              "task, or anything else about the frame. Their ratings are the reference "
              "standard the pipeline is measured against."),
        ("p", "Reviewers record what they observe for each body segment; REBA and RULA are "
              "then composed from those observations using the same lookup tables the "
              "pipeline uses. That removes hand-arithmetic error and means any difference "
              "between a reviewer and the pipeline is a difference in postural judgement, "
              "not in bookkeeping."),
        ("h2", "What is in this workbook"),
        ("p", "'Frames' — one row per frame: the image, what the pipeline scored, and every "
              "component subscore behind that score. This is what you compare the returned "
              "ratings against."),
        ("p", "'Design' — how the 60 frames are composed across risk band, stratum, task and "
              "participant, and the settings that produced them."),
        ("p", "'Column guide' — what each column in the Frames sheet means."),
        ("h2", "How to compare when the ratings come back"),
        ("p", "Put the returned files (.json from the rating page, .xlsx from the workbook, "
              "mixed freely) in one folder and run:"),
        ("code", "python -m validation.analyze_ratings --ratings returned/ --output report.md"),
        ("p", "That produces the agreement report: between-reviewer reliability first, then "
              "the pipeline against the reviewer consensus — balanced, prevalence-weighted, "
              "and split by stratum — then a per-component breakdown showing which body "
              "segment any disagreement lives in."),
        ("p", "You do not need this workbook to run that; the analysis reads "
              "ground_truth_key.csv directly. This is for reading and for checking "
              "individual frames by eye."),
        ("h2", "Why the frames are split into two strata"),
        ("p", "'sustained' means the frame's risk band holds across the frames either side of "
              "it, so the label describes a posture the participant actually held. "
              "'transient' means it does not — the band lasts an instant. Measured on this "
              "dataset the rare bands are overwhelmingly transient, so a sample balanced on "
              "risk alone would have handed reviewers the pipeline's momentary pose glitches "
              "and counted the resulting disagreement against its posture assessment. "
              "Splitting them lets the report state each separately, and the gap between "
              "them is how much pose instability costs."),
        ("h2", "Provenance"),
    ]

    row = 2
    for kind, text in blocks:
        cell = sheet.cell(row=row, column=2, value=text)
        if kind == "h":
            cell.font = Font(name=FONT, size=16, bold=True, color=ACCENT)
            row += 2
        elif kind == "h2":
            cell.font = Font(name=FONT, size=12, bold=True, color=ACCENT)
            row += 2
        elif kind == "warn":
            cell.font = Font(name=FONT, size=11, bold=True, color="9C0006")
            cell.fill = PatternFill("solid", fgColor="F8D7DA")
            cell.border = BOX
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            sheet.row_dimensions[row].height = 32
            row += 2
        elif kind == "code":
            cell.font = Font(name="Consolas", size=10)
            cell.fill = PatternFill("solid", fgColor="F2F2F2")
            cell.border = BOX
            row += 2
        else:
            cell.font = Font(name=FONT, size=11)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            sheet.row_dimensions[row].height = 14 * (1 + len(text) // 100)
            row += 2

    facts = [
        ("Frames drawn", f"{len(key)}"),
        ("Participants represented", f"{key['participant'].nunique()}"),
        ("Recordings drawn from", f"{key.groupby(['participant', 'task']).ngroups}"),
        ("Processing variant", str(provenance.get("fps_variant", "—"))),
        ("Candidate pool", f"{provenance.get('filter_counts', {}).get('banded', '—')} frames"),
        ("Random seed", str(settings.get("seed", "—"))),
        ("Stability half-window", f"±{settings.get('stability_half_window', '—')} samples"),
        ("Task balance achieved", str(design.get("by_task", "—"))),
    ]
    for label, value in facts:
        sheet.cell(row=row, column=2, value=f"{label}:  {value}").font = Font(
            name=FONT, size=10, color=MUTED
        )
        row += 1


def _frames_sheet(book: Workbook, key: pd.DataFrame, images_dir: Path) -> list[io.BytesIO]:
    from PIL import Image as PILImage

    sheet = book.create_sheet("Frames")
    sheet.sheet_view.showGridLines = False
    sheet.merge_cells("A1:S1")
    _title(sheet.cell(row=1, column=1, value="The 60 frames, and what the pipeline scored"), 13)
    sheet.row_dimensions[1].height = 24

    labels = [
        "Frame", "Image", "Participant", "Task", "Band", "Stratum", "Samples to band change",
        "Band run (s)", "REBA", "RULA", "Trunk", "Neck", "Legs", "Upper arm", "Lower arm",
        "Wrist", "Load", "Coupling", "Weight", "Source frame",
    ]
    widths = [11, 62, 11, 22, 9, 10, 13, 11, 7, 7, 7, 7, 7, 10, 10, 7, 8, 10, 9, 12]
    _header_row(sheet, 2, labels, widths)

    keep: list[io.BytesIO] = []
    ordered = key.sort_values("frame_id").reset_index(drop=True)
    for offset, row in ordered.iterrows():
        line = 3 + offset
        sheet.row_dimensions[line].height = 68

        source = images_dir / str(row["image_filename"])
        if source.is_file():
            with PILImage.open(source) as picture:
                ratio = THUMB_WIDTH_PX / picture.width
                size = (THUMB_WIDTH_PX, max(1, int(round(picture.height * ratio))))
                thumb = picture.convert("RGB").resize(size, PILImage.Resampling.LANCZOS)
                buffer = io.BytesIO()
                thumb.save(buffer, format="JPEG", quality=70, optimize=True)
                buffer.seek(0)
            keep.append(buffer)
            picture_ref = XLImage(buffer)
            picture_ref.width, picture_ref.height = size
            sheet.add_image(picture_ref, f"B{line}")

        to_change = row.get("frames_to_band_change")
        values = [
            row["frame_id"], "", row["participant"], row["task"], row["band"], row["stratum"],
            "—" if pd.isna(to_change) else int(to_change),
            round(float(row["band_run_sec"]), 1),
            row.get("pipeline_reba_final"), row.get("pipeline_rula_final"),
            *[row.get(f"pipeline_reba_{name}") for name in REBA_COMPONENTS],
            row.get("assumed_load_kg"), row.get("assumed_coupling"),
            round(float(row["sampling_weight"]), 3), row["frame_number"],
        ]
        for column, value in enumerate(values, start=1):
            cell = sheet.cell(row=line, column=column, value=value)
            cell.font = Font(name=FONT, size=9)
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = BOX
            if column == 5:
                cell.fill = BAND_FILL.get(str(row["band"]), PatternFill())
                cell.font = Font(name=FONT, size=9, bold=True)
    sheet.freeze_panes = "C3"
    sheet.auto_filter.ref = f"A2:S{2 + len(ordered)}"
    return keep


def _design_sheet(book: Workbook, key: pd.DataFrame, manifest: dict[str, Any]) -> None:
    sheet = book.create_sheet("Design")
    sheet.sheet_view.showGridLines = False
    for letter, width in (("A", 26), ("B", 12), ("C", 12), ("D", 12), ("E", 12), ("F", 12)):
        sheet.column_dimensions[letter].width = width

    row = 1
    sheet.merge_cells("A1:F1")
    _title(sheet.cell(row=1, column=1, value="How the 60 frames are composed"), 13)
    sheet.row_dimensions[1].height = 24
    row = 3

    def table(caption: str, frame: pd.DataFrame) -> None:
        nonlocal row
        sheet.cell(row=row, column=1, value=caption).font = Font(
            name=FONT, size=11, bold=True, color=ACCENT
        )
        row += 1
        sheet.cell(row=row, column=1, value=frame.index.name or "").font = Font(
            name=FONT, size=9, bold=True, color=MUTED
        )
        for index, column in enumerate(frame.columns, start=2):
            cell = sheet.cell(row=row, column=index, value=str(column))
            cell.font = Font(name=FONT, size=9, bold=True, color=MUTED)
            cell.alignment = Alignment(horizontal="center")
        row += 1
        for label, values in frame.iterrows():
            sheet.cell(row=row, column=1, value=str(label)).font = Font(name=FONT, size=10)
            for index, value in enumerate(values, start=2):
                cell = sheet.cell(row=row, column=index, value=int(value))
                cell.font = Font(name=FONT, size=10)
                cell.alignment = Alignment(horizontal="center")
                cell.border = BOX
            row += 1
        row += 2

    bands = ["Low", "Medium", "High"]
    by_stratum = pd.crosstab(key["stratum"], key["band"]).reindex(columns=bands, fill_value=0)
    by_stratum.index.name = "Stratum"
    table("Risk band × stratum — the hard quota the design guarantees", by_stratum)

    by_task = pd.crosstab(key["task"], key["band"]).reindex(columns=bands, fill_value=0)
    by_task.index.name = "Task"
    table("Risk band × task", by_task)

    by_participant = pd.crosstab(key["participant"], key["band"]).reindex(
        columns=bands, fill_value=0
    )
    by_participant.index.name = "Participant"
    table("Risk band × participant — every participant is represented", by_participant)

    notes = manifest.get("design", {}).get("notes") or []
    if notes:
        sheet.cell(row=row, column=1, value="Where the design had to give ground").font = Font(
            name=FONT, size=11, bold=True, color=ACCENT
        )
        row += 1
        for note in notes:
            sheet.cell(row=row, column=1, value=str(note)).font = Font(name=FONT, size=10)
            row += 1
        row += 1

    prevalence = manifest.get("pool_provenance", {}).get("band_prevalence", {})
    if prevalence:
        total = sum(prevalence.values()) or 1
        sheet.cell(row=row, column=1, value="Risk in the pool the frames were drawn from").font = (
            Font(name=FONT, size=11, bold=True, color=ACCENT)
        )
        row += 1
        sheet.cell(row=row, column=1, value=(
            "The sample is balanced on purpose, so it does not look like the population. "
            "The 'Weight' column on the Frames sheet undoes that when agreement is "
            "re-estimated for the population."
        )).font = Font(name=FONT, size=9, color=MUTED)
        row += 1
        for band in bands:
            count = int(prevalence.get(band, 0))
            sheet.cell(row=row, column=1, value=band).font = Font(name=FONT, size=10)
            sheet.cell(row=row, column=2, value=count).font = Font(name=FONT, size=10)
            sheet.cell(row=row, column=3, value=f"{100 * count / total:.1f}%").font = Font(
                name=FONT, size=10
            )
            row += 1


def _column_guide(book: Workbook) -> None:
    sheet = book.create_sheet("Column guide")
    sheet.sheet_view.showGridLines = False
    sheet.column_dimensions["A"].width = 22
    sheet.column_dimensions["B"].width = 96

    sheet.merge_cells("A1:B1")
    _title(sheet.cell(row=1, column=1, value="What each column on the Frames sheet means"), 12)
    sheet.row_dimensions[1].height = 22

    entries = [
        ("Frame", "The anonymous id the reviewers see. Ratings are joined back on this."),
        ("Participant / Task", "Which recording the frame came from. Reviewers never see either."),
        ("Band", "The pipeline's REBA score collapsed to Low (1-3), Medium (4-7), High (8-15)."),
        ("Stratum", "'sustained' — the band holds across the two samples either side, so it "
                    "describes a posture that was held. 'transient' — it does not. Frames at the "
                    "very start or end of a recording are excluded from the study altogether: "
                    "their stability cannot be judged, and calling them transient would put "
                    "stable postures into the stratum that measures instability."),
        ("Samples to band change", "How many samples away the risk band next differs. This is "
                                   "what decides the stratum: 3 or more is sustained, 1 or 2 is "
                                   "transient. At 10 fps a sample is 0.1 s."),
        ("Band run (s)", "How long the run of frames sharing this band lasts in total. Read it "
                         "alongside the column before it, not instead of it — a frame can sit "
                         "inside a long run and still be transient, because a one-frame spike "
                         "two samples away breaks its immediate neighbourhood. Both are exported "
                         "so the sustained/transient line can be re-cut in sensitivity analysis "
                         "rather than taken on trust."),
        ("REBA / RULA", "The pipeline's final scores for this frame."),
        ("Trunk … Wrist", "The pipeline's REBA component subscores. Compare these against a "
                          "reviewer's to see which body segment a disagreement lives in."),
        ("Load / Coupling", "The task inputs the pipeline used. Reviewers are given these same "
                            "values rather than estimating them, so the comparison isolates "
                            "posture."),
        ("Weight", "Inverse inclusion probability. The design over-samples rare bands on "
                   "purpose; weighting by this re-estimates agreement for the population the "
                   "frames were drawn from. Kappa is prevalence-sensitive, so the balanced "
                   "figure alone would not describe the study."),
        ("Source frame", "The frame number in the original recording, for tracing back."),
    ]
    row = 3
    for name, meaning in entries:
        sheet.cell(row=row, column=1, value=name).font = Font(name=FONT, size=10, bold=True)
        cell = sheet.cell(row=row, column=2, value=meaning)
        cell.font = Font(name=FONT, size=10)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        sheet.row_dimensions[row].height = 14 * (1 + len(meaning) // 92)
        row += 1


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build(sample_dir: Path, destination: Path) -> Path:
    key_path = sample_dir / "ground_truth_key.csv"
    manifest_path = sample_dir / "sampling_manifest.json"
    if not key_path.is_file():
        raise ReferenceError(f"ground truth key not found: {key_path}")
    key = pd.read_csv(key_path)
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    )

    book = Workbook()
    book.remove(book.active)
    _how_to_use(book, key, manifest)
    keep = _frames_sheet(book, key, sample_dir / "images")
    _design_sheet(book, key, manifest)
    _column_guide(book)
    book.active = 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    book.save(destination)
    LOGGER.info(
        "Wrote %s (%d frames, %.1f MB)", destination, len(key), destination.stat().st_size / 1e6
    )
    del keep
    return destination


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )
    destination = args.out or (args.sample_dir / "ground_truth_reference.xlsx")
    try:
        build(args.sample_dir, destination)
    except ReferenceError as exc:
        LOGGER.error("%s", exc)
        return 1
    print(f"\nReference document: {destination}")
    print("This is the answer key — keep it away from the reviewers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
