"""Build the Excel scoring workbook that goes out alongside the web app.

Some raters would rather work in Excel than in a browser, so the package carries both.
The two are not separate methods: they ask exactly the same questions in the same
words, and both are read back by :mod:`validation.scoring`, which recomposes REBA and
RULA from the raw observations using the project's own calculators.  Whichever a rater
chooses, the numbers that reach the analysis are produced by identical code.

The workbook is one sheet per image, in presentation order, each carrying the frame,
the given subscores, drop-downs for every posture judgement, and a live score composed
by in-sheet formulas so the rater gets the same immediate feedback the web app gives.
Those formulas exist for the rater's benefit only -- the analysis never depends on
Excel having evaluated them, so a workbook returned from LibreOffice or Google Sheets
is still readable.

Usage
-----
    python -m validation.build_workbook
    python -m validation.build_workbook --out ./scoring_workbook.xlsx
"""

from __future__ import annotations

import argparse
import io
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Protection, Side
from openpyxl.utils import get_column_letter
from openpyxl.workbook.defined_name import DefinedName
from openpyxl.worksheet.datavalidation import DataValidation

from ergonomics.reba import REBACalculator
from ergonomics.rula import RULACalculator
from validation import scoring

LOGGER = logging.getLogger("validation.build_workbook")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLE_DIR = PROJECT_ROOT / "validation_sample"

FONT = "Arial"
EMBED_WIDTH_PX = 1500  # composite is ~5.35:1, so this gives ~500 px per camera panel

INK = "1A1A1A"
MUTED = "5F6672"
ACCENT = "1F4E79"
INPUT_FILL = PatternFill("solid", fgColor="FFF6D5")   # cells the rater fills in
GIVEN_FILL = PatternFill("solid", fgColor="E8EEF7")   # supplied, not to be estimated
CALC_FILL = PatternFill("solid", fgColor="F2F2F2")    # computed, do not type
HEAD_FILL = PatternFill("solid", fgColor="1F4E79")
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

# --- fixed row layout, identical on every frame sheet ---------------------------
R_TITLE, R_NAV = 1, 2
R_IMAGE = 4
R_GIVEN_HEAD = 19
R_GIVEN = 20          # load, coupling, reba activity, rula muscle, rula force -> 20..24
R_OBS_HEAD = 26
R_OBS_COLS = 27
R_OBS = 28            # 19 observation rows -> 28..46
R_CALC_HEAD = 48
R_CALC_COLS = 49
R_CALC = 50           # trunk..wrist twist -> 50..56, score A/B 57..58, total 59, risk 60
R_META_HEAD = 62
R_META = 63           # confidence, scorable, reason, notes -> 63..66

#: (field, label, kind).  ``kind`` is a named option list or "yesno".
QUESTIONS: list[tuple[str, str, str, str]] = [
    ("trunk", "Trunk position", "TrunkOpts", "Flexion or extension of the trunk from upright."),
    ("trunk_twisted", "  …trunk twisted?", "yesno", ""),
    ("trunk_side_bent", "  …trunk side-bent?", "yesno", ""),
    ("neck", "Neck position", "NeckOpts", "Flexion of the neck; choose extension if the head is tipped back."),
    ("neck_twisted", "  …neck twisted?", "yesno", ""),
    ("neck_side_bent", "  …neck side-bent?", "yesno", ""),
    ("legs", "Legs — weight bearing", "LegsOpts", "Of the more loaded leg."),
    ("seated", "  …seated?", "yesno", "If seated, knee flexion is not used."),
    ("legs_supported_balanced", "  …legs supported and balanced?", "yesno", "Used by RULA."),
    ("knee", "Knee flexion", "KneeOpts", "Leave blank only if seated."),
    ("upper_arm", "Upper arm position", "UArmOpts", "Shoulder flexion of the more loaded arm."),
    ("arm_abducted", "  …abducted or rotated?", "yesno", ""),
    ("shoulder_raised", "  …shoulder raised?", "yesno", ""),
    ("arm_supported", "  …arm supported or gravity-assisted?", "yesno", ""),
    ("lower_arm", "Lower arm position", "LArmOpts", "Elbow flexion of the same arm."),
    ("lower_arm_across", "  …across midline or out to the side?", "yesno", "Used by RULA."),
    ("wrist", "Wrist position", "WristOpts", "Wrist flexion or extension of the same arm."),
    ("wrist_deviated", "  …deviated or bent from midline?", "yesno", ""),
    ("wrist_twist_end", "  …twisted at or near end of range?", "yesno", ""),
]

#: Human-readable option labels, and the band key each maps to.  Labels are what the
#: rater picks; keys are what :mod:`validation.scoring` consumes.
OPTION_LABELS: dict[str, list[tuple[str, str]]] = {
    "TrunkOpts": [
        ("Upright / neutral", "upright"),
        ("Flexed 0–20°", "flex_0_20"),
        ("Flexed 20–60°", "flex_20_60"),
        ("Flexed more than 60°", "flex_gt60"),
        ("Extended 0–20°", "ext_0_20"),
        ("Extended more than 20°", "ext_gt20"),
    ],
    "NeckOpts": [
        ("Flexed 0–10°", "flex_0_10"),
        ("Flexed 10–20°", "flex_10_20"),
        ("Flexed more than 20°", "flex_gt20"),
        ("In extension", "ext"),
    ],
    "LegsOpts": [
        ("Bilateral weight bearing, walking or sitting", "bilateral"),
        ("Unilateral, weight on one leg, or unstable", "unilateral"),
    ],
    "KneeOpts": [
        ("Less than 30°", "lt30"),
        ("30–60°", "k30_60"),
        ("More than 60°", "gt60"),
    ],
    "UArmOpts": [
        ("20° extension to 20° flexion", "n20_20"),
        ("More than 20° extension, or 20–45° flexion", "mid"),
        ("Flexed 45–90°", "f45_90"),
        ("Flexed more than 90°", "f_gt90"),
    ],
    "LArmOpts": [
        ("Flexed 60–100°", "f60_100"),
        ("Flexed less than 60° or more than 100°", "outside"),
    ],
    "WristOpts": [
        ("Neutral", "neutral"),
        ("Flexed or extended up to 15°", "le15"),
        ("Flexed or extended more than 15°", "gt15"),
    ],
}

CONFIDENCE_LABELS = ["5 — Very confident", "4 — Confident", "3 — Moderate", "2 — Low", "1 — Very low"]

#: Field -> the option list it draws on, used when reading a workbook back.
FIELD_OPTIONS = {field: kind for field, _label, kind, _help in QUESTIONS if kind != "yesno"}
LABEL_TO_KEY = {
    kind: {label: key for label, key in pairs} for kind, pairs in OPTION_LABELS.items()
}


class WorkbookError(RuntimeError):
    """Raised when the workbook cannot be built."""


# ---------------------------------------------------------------------------
# Reference sheets
# ---------------------------------------------------------------------------


def _base_scores(kind: str, key: str) -> tuple[int, int]:
    """Base component score for a band, before modifiers, for REBA and RULA.

    Derived by running the band's representative angle through the calculators with
    every modifier off, so the workbook cannot drift from them.  ``-1`` marks a band
    the instrument does not define -- RULA has no trunk-extension bin.
    """

    if kind == "TrunkOpts":
        obs = {"trunk": key}
        reba = scoring.reba_components(obs)["trunk"]
        rula = scoring.rula_components(obs)["trunk"]
    elif kind == "NeckOpts":
        obs = {"neck": key}
        reba = scoring.reba_components(obs)["neck"]
        rula = scoring.rula_components(obs)["neck"]
    elif kind == "UArmOpts":
        obs = {"upper_arm": key}
        reba = scoring.reba_components(obs)["upper_arm"]
        rula = scoring.rula_components(obs)["upper_arm"]
    elif kind == "LArmOpts":
        obs = {"lower_arm": key}
        reba = scoring.reba_components(obs)["lower_arm"]
        rula = scoring.rula_components(obs)["lower_arm"]
    elif kind == "WristOpts":
        obs = {"wrist": key}
        reba = scoring.reba_components(obs)["wrist"]
        rula = scoring.rula_components(obs)["wrist"]
    elif kind == "KneeOpts":
        # REBA's knee contribution when standing: the increment over bilateral/neutral.
        standing = scoring.reba_components({"legs": "bilateral", "knee": key, "seated": False})["legs"]
        if standing is None:
            raise WorkbookError(f"knee band {key!r} does not yield a REBA legs score")
        reba = standing - 1
        rula = -1
    elif kind == "LegsOpts":
        reba = 1 if key == "bilateral" else 2
        rula = -1
    else:
        raise WorkbookError(f"unknown option list {kind!r}")
    return (-1 if reba is None else int(reba), -1 if rula is None else int(rula))


def _write_reference(workbook: Workbook) -> None:
    """Hidden sheet holding option labels and their base component scores."""

    sheet = workbook.create_sheet("_ref")
    sheet.sheet_state = "hidden"
    row = 1
    for kind, pairs in OPTION_LABELS.items():
        start = row
        for label, key in pairs:
            reba, rula = _base_scores(kind, key)
            sheet.cell(row=row, column=1, value=label)
            sheet.cell(row=row, column=2, value=key)
            sheet.cell(row=row, column=3, value=reba)
            sheet.cell(row=row, column=4, value=rula)
            row += 1
        end = row - 1
        workbook.defined_names.add(
            DefinedName(kind, attr_text=f"_ref!$A${start}:$A${end}")
        )
        workbook.defined_names.add(
            DefinedName(f"{kind}Map", attr_text=f"_ref!$A${start}:$D${end}")
        )
        row += 1

    start = row
    for label in ("Yes", "No"):
        sheet.cell(row=row, column=1, value=label)
        row += 1
    workbook.defined_names.add(DefinedName("YesNo", attr_text=f"_ref!$A${start}:$A${row - 1}"))
    row += 1

    start = row
    for label in CONFIDENCE_LABELS:
        sheet.cell(row=row, column=1, value=label)
        row += 1
    workbook.defined_names.add(DefinedName("ConfOpts", attr_text=f"_ref!$A${start}:$A${row - 1}"))


def _write_tables(workbook: Workbook) -> None:
    """Hidden sheet holding the REBA and RULA lookup tables, flattened for INDEX.

    Copied straight out of the calculators, so the workbook composes scores with the
    same tables as the pipeline under test.
    """

    reba, rula = REBACalculator(), RULACalculator()
    sheet = workbook.create_sheet("_tables")
    sheet.sheet_state = "hidden"
    row = 1

    def block(name: str, rows: list[list[int]]) -> None:
        nonlocal row
        start = row
        for values in rows:
            for column, value in enumerate(values, start=1):
                sheet.cell(row=row, column=column, value=int(value))
            row += 1
        workbook.defined_names.add(
            DefinedName(
                name,
                attr_text=f"_tables!$A${start}:${get_column_letter(len(rows[0]))}${row - 1}",
            )
        )
        row += 1

    # REBA A: (trunk, neck) rows x legs columns
    block("RebaA", [list(reba.TABLE_A[t][n]) for t in range(5) for n in range(3)])
    # REBA B: (upper, lower) rows x wrist columns
    block("RebaB", [list(reba.TABLE_B[u][lo]) for u in range(6) for lo in range(2)])
    block("RebaC", [list(r) for r in reba.TABLE_C])
    # RULA A: (upper, lower, wrist) rows x twist columns
    block(
        "RulaA",
        [list(rula.TABLE_A[u][lo][w]) for u in range(6) for lo in range(3) for w in range(4)],
    )
    # RULA B: (neck, trunk) rows x legs columns
    block("RulaB", [list(rula.TABLE_B[n][t]) for n in range(6) for t in range(6)])
    block("RulaC", [list(r) for r in rula.TABLE_C])

    # Risk labels, looked up by final score
    start = row
    for score in range(1, 16):
        sheet.cell(row=row, column=1, value=score)
        sheet.cell(row=row, column=2, value=reba.RISK_LEVELS[score][0])
        row += 1
    workbook.defined_names.add(DefinedName("RebaRisk", attr_text=f"_tables!$A${start}:$B${row - 1}"))
    row += 1
    start = row
    for score in range(1, 8):
        sheet.cell(row=row, column=1, value=score)
        sheet.cell(row=row, column=2, value=rula.RISK_LEVELS[score][0])
        row += 1
    workbook.defined_names.add(DefinedName("RulaRisk", attr_text=f"_tables!$A${start}:$B${row - 1}"))


# ---------------------------------------------------------------------------
# Frame sheets
# ---------------------------------------------------------------------------


def _row_of(field: str) -> int:
    return R_OBS + [q[0] for q in QUESTIONS].index(field)


def _yes(field: str) -> str:
    return f'(B{_row_of(field)}="Yes")*1'


def _formulas() -> dict[str, str]:
    """Component and total formulas, in terms of the fixed input rows."""

    trunk, neck = _row_of("trunk"), _row_of("neck")
    legs, knee = _row_of("legs"), _row_of("knee")
    uarm, larm, wrist = _row_of("upper_arm"), _row_of("lower_arm"), _row_of("wrist")
    seated = _row_of("seated")

    # Base score for a chosen band; column 3 is REBA, column 4 is RULA.
    def base(row: int, kind: str, column: int) -> str:
        return f'VLOOKUP(B{row},{kind}Map,{column},FALSE)'

    reba_trunk = f'MIN(5,{base(trunk,"TrunkOpts",3)}+MIN(1,{_yes("trunk_twisted")}+{_yes("trunk_side_bent")}))'
    rula_trunk = (
        f'IF({base(trunk,"TrunkOpts",4)}<0,"n/a",'
        f'MIN(6,{base(trunk,"TrunkOpts",4)}+{_yes("trunk_twisted")}+{_yes("trunk_side_bent")}))'
    )
    reba_neck = f'MIN(3,{base(neck,"NeckOpts",3)}+MIN(1,{_yes("neck_twisted")}+{_yes("neck_side_bent")}))'
    rula_neck = f'MIN(6,{base(neck,"NeckOpts",4)}+{_yes("neck_twisted")}+{_yes("neck_side_bent")})'
    reba_legs = (
        f'MIN(4,{base(legs,"LegsOpts",3)}+'
        f'IF(B{seated}="Yes",0,{base(knee,"KneeOpts",3)}))'
    )
    rula_legs = f'IF(B{_row_of("legs_supported_balanced")}="Yes",1,2)'
    arm = (
        f'MIN(6,MAX(1,{base(uarm,"UArmOpts",3)}+{_yes("arm_abducted")}'
        f'+{_yes("shoulder_raised")}-{_yes("arm_supported")}))'
    )
    reba_larm = base(larm, "LArmOpts", 3)
    rula_larm = f'MIN(3,{base(larm,"LArmOpts",4)}+{_yes("lower_arm_across")})'
    reba_wrist = f'MIN(3,{base(wrist,"WristOpts",3)}+MIN(1,{_yes("wrist_deviated")}+{_yes("wrist_twist_end")}))'
    rula_wrist = f'MIN(4,{base(wrist,"WristOpts",4)}+{_yes("wrist_deviated")})'
    rula_twist = f'IF(B{_row_of("wrist_twist_end")}="Yes",2,1)'
    return {
        "reba_trunk": reba_trunk, "rula_trunk": rula_trunk,
        "reba_neck": reba_neck, "rula_neck": rula_neck,
        "reba_legs": reba_legs, "rula_legs": rula_legs,
        "arm": arm, "reba_larm": reba_larm, "rula_larm": rula_larm,
        "reba_wrist": reba_wrist, "rula_wrist": rula_wrist, "rula_twist": rula_twist,
    }


def _style_title(cell: Any, size: int = 12) -> None:
    cell.font = Font(name=FONT, size=size, bold=True, color="FFFFFF")
    cell.fill = HEAD_FILL
    cell.alignment = Alignment(horizontal="left", vertical="center", indent=1)


def _build_frame_sheet(
    workbook: Workbook,
    index: int,
    total: int,
    item: dict[str, Any],
    image_bytes: io.BytesIO,
    image_size: tuple[int, int],
    sheet_name: str,
    previous_name: str | None,
    next_name: str | None,
) -> None:
    sheet = workbook.create_sheet(sheet_name)
    sheet.sheet_view.showGridLines = False
    widths = {"A": 42, "B": 34, "C": 46, "D": 4, "E": 12, "F": 12}
    for letter, width in widths.items():
        sheet.column_dimensions[letter].width = width

    # --- header and navigation ---
    sheet.merge_cells(start_row=R_TITLE, start_column=1, end_row=R_TITLE, end_column=6)
    title = sheet.cell(row=R_TITLE, column=1, value=f"Image {index} of {total}")
    _style_title(title, 13)
    sheet.row_dimensions[R_TITLE].height = 24

    nav = [("◀ Previous", previous_name), ("Index", "Index"), ("Next ▶", next_name)]
    for offset, (label, target) in enumerate(nav):
        cell = sheet.cell(row=R_NAV, column=1 + offset)
        if target is None:
            cell.value = ""
            continue
        cell.value = label
        cell.hyperlink = f"#'{target}'!A1"
        cell.font = Font(name=FONT, size=10, color=ACCENT, underline="single")
    full = sheet.cell(row=R_NAV, column=5, value="Open full-size image ↗")
    full.hyperlink = f"images/{item['image']}"
    full.font = Font(name=FONT, size=10, color=ACCENT, underline="single")

    # --- image ---
    picture = XLImage(image_bytes)
    picture.width, picture.height = image_size
    sheet.add_image(picture, f"A{R_IMAGE}")
    rows_needed = max(1, int(image_size[1] / 20) + 1)
    for offset in range(rows_needed):
        sheet.row_dimensions[R_IMAGE + offset].height = 20

    # --- given ---
    sheet.merge_cells(start_row=R_GIVEN_HEAD, start_column=1, end_row=R_GIVEN_HEAD, end_column=6)
    _style_title(
        sheet.cell(
            row=R_GIVEN_HEAD, column=1,
            value="GIVEN FOR THIS IMAGE — score these as stated; please do not estimate them",
        ),
        11,
    )
    given = item["given"]
    rows = [
        ("Handled load", given["load_label"]),
        ("Coupling", given["coupling_label"]),
        ("REBA activity score", f"+{given['reba_activity']}"),
        ("RULA muscle-use score", f"+{given['rula_muscle_use']}"),
        ("RULA force/load score", f"+{given['rula_force_load']}"),
    ]
    for offset, (label, value) in enumerate(rows):
        sheet.cell(row=R_GIVEN + offset, column=1, value=label).font = Font(name=FONT, size=10)
        cell = sheet.cell(row=R_GIVEN + offset, column=2, value=value)
        cell.font = Font(name=FONT, size=10, bold=True)
        cell.fill = GIVEN_FILL
        cell.border = BOX

    # --- observations ---
    sheet.merge_cells(start_row=R_OBS_HEAD, start_column=1, end_row=R_OBS_HEAD, end_column=6)
    _style_title(
        sheet.cell(row=R_OBS_HEAD, column=1,
                   value="YOUR OBSERVATIONS — choose from the drop-downs in the shaded column"), 11)
    for column, label in ((1, "Question"), (2, "Your answer"), (3, "Guidance")):
        cell = sheet.cell(row=R_OBS_COLS, column=column, value=label)
        cell.font = Font(name=FONT, size=10, bold=True, color=MUTED)

    for offset, (_field, label, kind, help_text) in enumerate(QUESTIONS):
        row = R_OBS + offset
        name = sheet.cell(row=row, column=1, value=label)
        name.font = Font(name=FONT, size=10, bold=not label.startswith("  "))
        entry = sheet.cell(row=row, column=2)
        entry.fill = INPUT_FILL
        entry.border = BOX
        entry.font = Font(name=FONT, size=10)
        if kind == "yesno":
            entry.value = "No"
        sheet.cell(row=row, column=3, value=help_text).font = Font(name=FONT, size=9, color=MUTED)

    # --- computed ---
    sheet.merge_cells(start_row=R_CALC_HEAD, start_column=1, end_row=R_CALC_HEAD, end_column=6)
    _style_title(
        sheet.cell(row=R_CALC_HEAD, column=1,
                   value="COMPOSED SCORE — calculated from your answers; please do not type here"), 11)
    for column, label in ((1, "Component"), (2, "REBA"), (3, "RULA")):
        cell = sheet.cell(row=R_CALC_COLS, column=column, value=label)
        cell.font = Font(name=FONT, size=10, bold=True, color=MUTED)

    f = _formulas()
    complete = (
        f'AND(COUNTA(B{_row_of("trunk")})>0,COUNTA(B{_row_of("neck")})>0,'
        f'COUNTA(B{_row_of("legs")})>0,COUNTA(B{_row_of("upper_arm")})>0,'
        f'COUNTA(B{_row_of("lower_arm")})>0,COUNTA(B{_row_of("wrist")})>0,'
        f'OR(B{_row_of("seated")}="Yes",COUNTA(B{_row_of("knee")})>0))'
    )
    guard = lambda expr: f'=IF({complete},IFERROR({expr},""),"")'  # noqa: E731

    layout = [
        ("Trunk", f["reba_trunk"], f["rula_trunk"]),
        ("Neck", f["reba_neck"], f["rula_neck"]),
        ("Legs", f["reba_legs"], f["rula_legs"]),
        ("Upper arm", f["arm"], f["arm"]),
        ("Lower arm", f["reba_larm"], f["rula_larm"]),
        ("Wrist", f["reba_wrist"], f["rula_wrist"]),
        ("Wrist twist", None, f["rula_twist"]),
    ]
    for offset, (label, reba_expr, rula_expr) in enumerate(layout):
        row = R_CALC + offset
        sheet.cell(row=row, column=1, value=label).font = Font(name=FONT, size=10)
        for column, expr in ((2, reba_expr), (3, rula_expr)):
            cell = sheet.cell(row=row, column=column)
            cell.value = guard(expr) if expr else "—"
            cell.fill = CALC_FILL
            cell.border = BOX
            cell.font = Font(name=FONT, size=10)
            cell.alignment = Alignment(horizontal="center")

    trunk_r, neck_r, legs_r = R_CALC, R_CALC + 1, R_CALC + 2
    uarm_r, larm_r, wrist_r, twist_r = R_CALC + 3, R_CALC + 4, R_CALC + 5, R_CALC + 6
    row_a, row_b, row_total, row_risk = R_CALC + 7, R_CALC + 8, R_CALC + 9, R_CALC + 10

    reba_table_a = f'INDEX(RebaA,(B{trunk_r}-1)*3+B{neck_r},B{legs_r})'
    reba_table_b = f'INDEX(RebaB,(B{uarm_r}-1)*2+B{larm_r},B{wrist_r})'
    rula_table_a = f'INDEX(RulaA,((C{uarm_r}-1)*3+(C{larm_r}-1))*4+C{wrist_r},C{twist_r})'
    rula_table_b = f'INDEX(RulaB,(C{neck_r}-1)*6+C{trunk_r},C{legs_r})'

    # Given subscores are shown to the rater as text ("+1"), so the numeric value is
    # taken from the item rather than parsed back out of the display cell.
    load_force = given["reba_load_force"]
    coupling = given["reba_coupling"]
    activity = given["reba_activity"]
    muscle = given["rula_muscle_use"]
    force = given["rula_force_load"]

    numeric = f'AND(ISNUMBER(B{trunk_r}),ISNUMBER(B{neck_r}),ISNUMBER(B{legs_r}),' \
              f'ISNUMBER(B{uarm_r}),ISNUMBER(B{larm_r}),ISNUMBER(B{wrist_r}))'
    rula_numeric = f'AND(ISNUMBER(C{trunk_r}),ISNUMBER(C{neck_r}),ISNUMBER(C{legs_r}),' \
                   f'ISNUMBER(C{uarm_r}),ISNUMBER(C{larm_r}),ISNUMBER(C{wrist_r}))'

    sheet.cell(row=row_a, column=1, value="Score A").font = Font(name=FONT, size=10)
    sheet.cell(row=row_a, column=2,
               value=f'=IF({numeric},{reba_table_a}+{load_force},"")')
    sheet.cell(row=row_a, column=3,
               value=f'=IF({rula_numeric},{rula_table_a}+{muscle}+{force},"")')
    sheet.cell(row=row_b, column=1, value="Score B").font = Font(name=FONT, size=10)
    sheet.cell(row=row_b, column=2,
               value=f'=IF({numeric},{reba_table_b}+{coupling},"")')
    sheet.cell(row=row_b, column=3,
               value=f'=IF({rula_numeric},{rula_table_b}+{muscle}+{force},"")')

    sheet.cell(row=row_total, column=1, value="TOTAL").font = Font(name=FONT, size=11, bold=True)
    sheet.cell(row=row_total, column=2, value=(
        f'=IF({numeric},MIN(15,MAX(1,INDEX(RebaC,MEDIAN(1,B{row_a},12),MEDIAN(1,B{row_b},12))+{activity})),"")'
    ))
    sheet.cell(row=row_total, column=3, value=(
        f'=IF({rula_numeric},INDEX(RulaC,MIN(C{row_a},8),MIN(C{row_b},7)),"")'
    ))
    sheet.cell(row=row_risk, column=1, value="Risk level").font = Font(name=FONT, size=10)
    sheet.cell(row=row_risk, column=2,
               value=f'=IFERROR(VLOOKUP(B{row_total},RebaRisk,2,FALSE),"")')
    sheet.cell(row=row_risk, column=3,
               value=f'=IFERROR(VLOOKUP(C{row_total},RulaRisk,2,FALSE),"")')

    for row in (row_a, row_b, row_total, row_risk):
        for column in (2, 3):
            cell = sheet.cell(row=row, column=column)
            cell.fill = CALC_FILL
            cell.border = BOX
            cell.alignment = Alignment(horizontal="center")
            cell.font = Font(name=FONT, size=11 if row == row_total else 10,
                             bold=row == row_total)

    # --- rater metadata ---
    sheet.merge_cells(start_row=R_META_HEAD, start_column=1, end_row=R_META_HEAD, end_column=6)
    _style_title(sheet.cell(row=R_META_HEAD, column=1, value="ABOUT YOUR ASSESSMENT"), 11)
    meta = [
        ("Confidence in this assessment", ""),
        ("Can you score this frame?", "Yes"),
        ("If not, why not?", ""),
        ("Notes (optional)", ""),
    ]
    for offset, (label, default) in enumerate(meta):
        row = R_META + offset
        sheet.cell(row=row, column=1, value=label).font = Font(name=FONT, size=10)
        cell = sheet.cell(row=row, column=2, value=default)
        cell.fill = INPUT_FILL
        cell.border = BOX
        cell.font = Font(name=FONT, size=10)

    # --- data validation ---
    # Excel stores a list source without a leading "=", and openpyxl writes the string
    # through verbatim; a stray "=" makes the file open with a repair prompt.
    for offset, (_field, _label, kind, _help) in enumerate(QUESTIONS):
        row = R_OBS + offset
        source = "YesNo" if kind == "yesno" else kind
        validation = DataValidation(type="list", formula1=source, allow_blank=True)
        validation.error = "Please choose one of the listed options."
        validation.errorTitle = "Not a listed option"
        sheet.add_data_validation(validation)
        validation.add(sheet.cell(row=row, column=2))
    for row, source in ((R_META, "ConfOpts"), (R_META + 1, "YesNo")):
        validation = DataValidation(type="list", formula1=source, allow_blank=True)
        sheet.add_data_validation(validation)
        validation.add(sheet.cell(row=row, column=2))

    # Frame identity, kept where a reader can see it but out of the way.
    tag = sheet.cell(row=R_TITLE, column=8, value=item["key"])
    tag.font = Font(name=FONT, size=9, color="FFFFFF")

    # Formula cells are locked; the shaded answer cells are not.  Protection carries no
    # password, so a rater who needs to get past it can.
    for row in range(1, R_META + 6):
        for column in range(1, 9):
            sheet.cell(row=row, column=column).protection = Protection(locked=True)
    for offset in range(len(QUESTIONS)):
        sheet.cell(row=R_OBS + offset, column=2).protection = Protection(locked=False)
    for offset in range(len(meta)):
        sheet.cell(row=R_META + offset, column=2).protection = Protection(locked=False)
    sheet.protection.sheet = True
    sheet.protection.enable()
    sheet.protection.selectLockedCells = False


# ---------------------------------------------------------------------------
# Index, instructions, results
# ---------------------------------------------------------------------------


def _build_instructions(workbook: Workbook, total: int, unique: int, rater: str = "") -> None:
    sheet = workbook.create_sheet("Instructions", 0)
    sheet.sheet_view.showGridLines = False
    sheet.column_dimensions["A"].width = 3
    sheet.column_dimensions["B"].width = 110

    blocks: list[tuple[str, str]] = [
        ("h", "Posture assessment — expert rating"),
        ("p", "A blind comparison of expert postural assessment against an automated video-based "
              "system. Your ratings are the reference standard."),
        ("h2", "What to do"),
        ("p", f"There are {total} images, one per sheet, in the tabs along the bottom (F001, F002, …). "
              "Work through them in order, or use the Index sheet to jump around and see what is left."),
        ("p", "Each sheet shows the same instant from three synchronised cameras — front, side and "
              "isometric. Answer the drop-downs in the shaded column. The score is composed for you."),
        ("h2", "You record observations, not totals"),
        ("p", "You are not asked to add anything up or to read a REBA or RULA table. You select the "
              "posture band you observe for each body segment, and the sheet composes REBA and RULA "
              "using the same lookup tables as the system being tested."),
        ("p", "This removes clerical error — hand-composing a score through three lookup tables is a "
              "known source of mistakes in reliability studies, and a table slip would otherwise be "
              "recorded as genuine disagreement. It also means any difference that remains comes from "
              "the postural judgement itself, and it lets us see which body segment a disagreement "
              "lives in rather than only that the totals differed."),
        ("h2", "Load, coupling and activity are given to you"),
        ("p", "Each sheet states the handled load, coupling, and the activity / muscle-use scores in "
              "the blue box. Please score them as stated, even where your own reading of the scene "
              "would differ. Load and coupling cannot be judged reliably from a still frame, and "
              "activity depends on what happens before and after the instant shown. If you and the "
              "system assumed different loads, the comparison would measure that instead of posture."),
        ("h2", "Which side to score"),
        ("p", "Score the more severely loaded side — the side you would report if you were assessing "
              "this worker. If the two sides look equivalent, score either."),
        ("h2", "If an image cannot be scored"),
        ("p", "Some frames may be blurred, occluded, or caught mid-movement. Set \"Can you score this "
              "frame?\" to No and give a brief reason rather than guessing. That is a result in itself "
              "and is more useful than a low-confidence score."),
        ("h2", "Seeing the image more clearly"),
        ("p", "Each sheet has an \"Open full-size image\" link that opens the full-resolution frame in "
              "your usual image viewer, where you can zoom freely. The same files are in the images "
              "folder next to this workbook."),
        ("h2", "Which cells you fill in"),
        ("p", "Only the cream-shaded cells. The blue cells are given to you, and the grey cells are "
              "calculated. The sheets are protected so a formula cannot be overwritten by accident; "
              "there is no password if you ever need to unprotect one."),
        ("h2", "When you are finished"),
        ("p", "Save the workbook and send it back. Please check your rater ID in the cell below "
              "before you start. Every reviewer sees the same images in the same order, so image "
              "12 is the same frame for all of you - but please still do not discuss individual "
              "images until everyone has finished, because your independent judgement is the "
              "whole point of the exercise."),
    ]
    if total != unique:
        blocks.append(
            ("p", f"{total - unique} of the {total} images appear more than once by design. Please "
                  "score each on what you see, without trying to recall an earlier answer.")
        )
    row = 2
    for kind, text in blocks:
        cell = sheet.cell(row=row, column=2, value=text)
        if kind == "h":
            cell.font = Font(name=FONT, size=16, bold=True, color=ACCENT)
            row += 2
        elif kind == "h2":
            cell.font = Font(name=FONT, size=12, bold=True, color=ACCENT)
            row += 2
        else:
            cell.font = Font(name=FONT, size=11, color=INK)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            sheet.row_dimensions[row].height = 15 * (1 + len(text) // 105)
            row += 2

    row += 1
    sheet.cell(row=row, column=2, value="Your name or rater ID:").font = Font(name=FONT, size=11, bold=True)
    entry = sheet.cell(row=row + 1, column=2, value=rater or None)
    entry.fill = INPUT_FILL
    entry.border = BOX
    entry.font = Font(name=FONT, size=11)
    workbook.defined_names.add(DefinedName("RaterID", attr_text=f"Instructions!$B${row + 1}"))


def _build_index(workbook: Workbook, sheet_names: list[str], items: list[dict[str, Any]]) -> None:
    sheet = workbook.create_sheet("Index", 1)
    sheet.sheet_view.showGridLines = False
    for letter, width in (("A", 12), ("B", 14), ("C", 14), ("D", 12), ("E", 20), ("F", 12), ("G", 44)):
        sheet.column_dimensions[letter].width = width

    sheet.merge_cells("A1:G1")
    _style_title(sheet.cell(row=1, column=1, value="Index — click a sheet name to open it"), 13)
    sheet.row_dimensions[1].height = 24

    total = len(sheet_names)
    sheet.cell(row=2, column=1, value="Scored:").font = Font(name=FONT, size=11, bold=True)
    sheet.cell(row=2, column=2, value=f'=COUNTIF(D5:D{4 + total},"Done")&" of {total}"').font = \
        Font(name=FONT, size=11, bold=True, color=ACCENT)

    headers = ["Sheet", "REBA total", "RULA total", "Status", "Confidence", "Scorable", "Notes"]
    for column, label in enumerate(headers, start=1):
        cell = sheet.cell(row=4, column=column, value=label)
        cell.font = Font(name=FONT, size=10, bold=True, color=MUTED)
        cell.border = BOX

    for offset, name in enumerate(sheet_names):
        row = 5 + offset
        link = sheet.cell(row=row, column=1, value=name)
        link.hyperlink = f"#'{name}'!A1"
        link.font = Font(name=FONT, size=10, color=ACCENT, underline="single")
        # A reference to an empty cell evaluates to 0 in Excel, which would fill the
        # unanswered columns with zeros; blank them explicitly instead.
        sheet.cell(row=row, column=2, value=f"='{name}'!B{R_CALC + 9}")
        sheet.cell(row=row, column=3, value=f"='{name}'!C{R_CALC + 9}")
        sheet.cell(row=row, column=4,
                   value=f'=IF(OR(ISNUMBER(\'{name}\'!B{R_CALC + 9}),\'{name}\'!B{R_META + 1}="No"),"Done","")')
        for column, source_row in ((5, R_META), (6, R_META + 1), (7, R_META + 3)):
            sheet.cell(
                row=row, column=column,
                value=f"=IF('{name}'!B{source_row}=\"\",\"\",'{name}'!B{source_row})",
            )
        for column in range(1, 8):
            cell = sheet.cell(row=row, column=column)
            cell.border = BOX
            if column > 1:
                cell.font = Font(name=FONT, size=10)
                cell.alignment = Alignment(horizontal="center" if column < 5 else "left")

    sheet.conditional_formatting.add(
        f"D5:D{4 + total}",
        CellIsRule(operator="equal", formula=['"Done"'],
                   fill=PatternFill("solid", fgColor="D6EAD6")),
    )
    sheet.freeze_panes = "A5"


def _build_results(workbook: Workbook, sheet_names: list[str], items: list[dict[str, Any]]) -> None:
    """Flat one-row-per-item table, mirroring what the web app exports."""

    sheet = workbook.create_sheet("Results")
    sheet.sheet_view.showGridLines = False
    headers = (
        ["rater_id", "item", "frame_id", "sheet"]
        + [q[0] for q in QUESTIONS]
        + ["reba_trunk", "reba_neck", "reba_legs", "reba_upper_arm", "reba_lower_arm", "reba_wrist",
           "reba_total", "rula_trunk", "rula_neck", "rula_legs", "rula_upper_arm", "rula_lower_arm",
           "rula_wrist", "rula_wrist_twist", "rula_total", "confidence", "scorable", "reason", "notes"]
    )
    for column, label in enumerate(headers, start=1):
        cell = sheet.cell(row=1, column=column, value=label)
        cell.font = Font(name=FONT, size=9, bold=True, color="FFFFFF")
        cell.fill = HEAD_FILL

    calc_rows = {
        "reba_trunk": ("B", R_CALC), "reba_neck": ("B", R_CALC + 1), "reba_legs": ("B", R_CALC + 2),
        "reba_upper_arm": ("B", R_CALC + 3), "reba_lower_arm": ("B", R_CALC + 4),
        "reba_wrist": ("B", R_CALC + 5), "reba_total": ("B", R_CALC + 9),
        "rula_trunk": ("C", R_CALC), "rula_neck": ("C", R_CALC + 1), "rula_legs": ("C", R_CALC + 2),
        "rula_upper_arm": ("C", R_CALC + 3), "rula_lower_arm": ("C", R_CALC + 4),
        "rula_wrist": ("C", R_CALC + 5), "rula_wrist_twist": ("C", R_CALC + 6),
        "rula_total": ("C", R_CALC + 9),
    }
    meta_rows = {"confidence": R_META, "scorable": R_META + 1, "reason": R_META + 2, "notes": R_META + 3}

    for offset, (name, item) in enumerate(zip(sheet_names, items, strict=True)):
        row = 2 + offset
        sheet.cell(row=row, column=1, value="=RaterID")
        sheet.cell(row=row, column=2, value=offset + 1)
        sheet.cell(row=row, column=3, value=item["frame_id"])
        sheet.cell(row=row, column=4, value=name)
        column = 5
        # Same blank-reads-as-zero guard as the Index: an unanswered drop-down must
        # export as empty, not as 0, or the returned table misreads as data.
        for question_offset in range(len(QUESTIONS)):
            source = f"'{name}'!B{R_OBS + question_offset}"
            sheet.cell(row=row, column=column, value=f'=IF({source}="","",{source})')
            column += 1
        for key in headers[4 + len(QUESTIONS):]:
            if key in calc_rows:
                letter, source_row = calc_rows[key]
                source = f"'{name}'!{letter}{source_row}"
            else:
                source = f"'{name}'!B{meta_rows[key]}"
            sheet.cell(row=row, column=column, value=f'=IF({source}="","",{source})')
            column += 1
        for c in range(1, len(headers) + 1):
            sheet.cell(row=row, column=c).font = Font(name=FONT, size=9)

    sheet.freeze_panes = "E2"


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build(
    items: list[dict[str, Any]], images_dir: Path, destination: Path, rater: str = ""
) -> Path:
    from PIL import Image as PILImage

    workbook = Workbook()
    workbook.remove(workbook.active)

    _write_reference(workbook)
    _write_tables(workbook)

    sheet_names = [f"F{index:03d}" for index in range(1, len(items) + 1)]
    keep_alive: list[io.BytesIO] = []

    for index, (item, name) in enumerate(zip(items, sheet_names, strict=True), start=1):
        source = images_dir / item["image"]
        if not source.is_file():
            raise WorkbookError(f"missing composite: {source}")
        with PILImage.open(source) as picture:
            ratio = EMBED_WIDTH_PX / picture.width
            size = (EMBED_WIDTH_PX, max(1, int(round(picture.height * ratio))))
            resized = picture.convert("RGB").resize(size, PILImage.Resampling.LANCZOS)
            buffer = io.BytesIO()
            resized.save(buffer, format="JPEG", quality=82, optimize=True)
            buffer.seek(0)
        keep_alive.append(buffer)
        _build_frame_sheet(
            workbook, index, len(items), item, buffer, size, name,
            sheet_names[index - 2] if index > 1 else None,
            sheet_names[index] if index < len(items) else None,
        )

    unique = len({item["frame_id"] for item in items})
    _build_instructions(workbook, len(items), unique, rater)
    _build_index(workbook, sheet_names, items)
    _build_results(workbook, sheet_names, items)

    workbook.active = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(destination)
    LOGGER.info(
        "Wrote %s (%d frame sheets, %.1f MB)",
        destination, len(items), destination.stat().st_size / 1e6,
    )
    return destination


# ---------------------------------------------------------------------------
# Reading a returned workbook
# ---------------------------------------------------------------------------


def _read_confidence(value: Any) -> str:
    """Accept numeric responses and exact labels from the template's own scale."""
    if value is None:
        return ""
    text = str(value).strip()
    for label in CONFIDENCE_LABELS:
        number, description = (part.strip() for part in label.split("—", 1))
        if text.casefold() in {number, label.casefold(), description.casefold()}:
            return number
    # Preserve unrecognised text for downstream validation rather than guessing.
    return text


def read_workbook(path: Path) -> list[dict[str, Any]]:
    """Read a returned workbook back into observation records.

    The per-frame sheets are read, not ``Results``: the raw drop-down answers are what
    the analysis needs, and taking them directly means the result does not depend on
    Excel having evaluated any formula.
    """

    from openpyxl import load_workbook

    workbook = load_workbook(path, data_only=True, read_only=False)
    rater = ""
    if "Instructions" in workbook.sheetnames:
        for row in workbook["Instructions"].iter_rows(min_col=2, max_col=2):
            for cell in row:
                if isinstance(cell.value, str) and cell.value.strip().lower().startswith("your name"):
                    below = workbook["Instructions"].cell(row=cell.row + 1, column=2).value
                    rater = str(below).strip() if below else ""
    if not rater:
        rater = path.stem

    records: list[dict[str, Any]] = []
    for name in workbook.sheetnames:
        if not (name.startswith("F") and name[1:].isdigit()):
            continue
        sheet = workbook[name]
        observations: dict[str, Any] = {}
        for offset, (field, _label, kind, _help) in enumerate(QUESTIONS):
            value = sheet.cell(row=R_OBS + offset, column=2).value
            if value is None or str(value).strip() == "":
                continue
            text = str(value).strip()
            if kind == "yesno":
                observations[field] = text.lower() == "yes"
            else:
                key = LABEL_TO_KEY.get(kind, {}).get(text)
                if key is None:
                    LOGGER.warning("%s!%s: unrecognised answer %r", path.name, name, text)
                    continue
                observations[field] = key
        confidence = sheet.cell(row=R_META, column=2).value
        scorable = sheet.cell(row=R_META + 1, column=2).value
        records.append(
            {
                "rater_id": rater,
                "sheet": name,
                "presentation_index": int(name[1:]),
                "frame_id": str(sheet.cell(row=R_TITLE, column=8).value or "").split("#")[0],
                "item_key": str(sheet.cell(row=R_TITLE, column=8).value or ""),
                "observations": observations,
                "confidence": _read_confidence(confidence),
                "confidence_raw": confidence,
                "cannot_score": str(scorable).strip().lower() == "no" if scorable else False,
                "cannot_reason": str(sheet.cell(row=R_META + 2, column=2).value or ""),
                "notes": str(sheet.cell(row=R_META + 3, column=2).value or ""),
            }
        )
    workbook.close()
    return records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_SAMPLE_DIR / "expert_package")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--rater", default=None, help="build this rater's copy, with their own order")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from validation.build_expert_package import build_items, order_for_rater

    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )
    key_path = args.sample_dir / "ground_truth_key.csv"
    if not key_path.is_file():
        LOGGER.error("ground truth key not found: %s", key_path)
        return 1
    key = pd.read_csv(key_path)
    items = build_items(key, 6, 20260801)
    rater = (args.rater or "").strip()
    if rater:
        items = order_for_rater(items, rater, 20260801)
    destination = args.out or (
        args.package_dir / (f"scoring_workbook_{rater}.xlsx" if rater else "scoring_workbook.xlsx")
    )
    build(items, args.sample_dir / "images", destination, rater)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
