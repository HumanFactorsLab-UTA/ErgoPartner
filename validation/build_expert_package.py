"""Build the offline scoring package that goes out to the expert raters.

The package is a folder (and a zip of it) containing the composites and a single
self-contained HTML page.  A rater opens the page by double-clicking it: no install,
no server, no network.  It keeps their progress in the browser and exports JSON and
CSV they send back.

Two things are worth knowing about how it is built.

*The REBA and RULA lookup tables are copied out of the calculators at build time*
rather than transcribed into JavaScript.  Expert and pipeline scores are therefore
composed by identical tables by construction; a change to ``ergonomics/reba.py``
propagates to the next package instead of silently disagreeing with it.

*The page is given no ground truth.*  It receives the image, the frame's anonymous
id, and the subscores the rater is not being asked to judge.  Band, participant,
task, pipeline score and stratum stay in ``ground_truth_key.csv``, which is not part
of the package.

Usage
-----
    python -m validation.build_expert_package
    python -m validation.build_expert_package --repeats 8 --no-zip
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from ergonomics.reba import REBACalculator
from ergonomics.rula import RULACalculator

LOGGER = logging.getLogger("validation.build_expert_package")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLE_DIR = PROJECT_ROOT / "validation_sample"
DEFAULT_PACKAGE_DIR = PROJECT_ROOT / "validation_sample" / "expert_package"
TEMPLATE_PATH = Path(__file__).resolve().parent / "expert_app_template.html"

STUDY_ID = "cv_ergonomics_blind_validation"
PACKAGE_VERSION = "1.0"
TITLE = "Posture assessment — expert rating"
SUBTITLE = (
    "A blind comparison of expert postural assessment against an automated "
    "video-based system. Your ratings are the reference standard."
)

#: Frames shown a second time, later in the sequence, so a rater's agreement with
#: themselves can be measured.  Off by default: it is a useful extra rather than a
#: requirement, and it makes the set longer than the 60 frames that were drawn.
#: Turn it on with ``--repeats 6`` to obtain a within-rater reliability estimate,
#: which bounds every other agreement figure in the report.
DEFAULT_REPEATS = 0

#: Minimum number of items between a frame and its repeat, so the second showing is
#: not answered from memory of the first.  Enforced per rater inside the page, since
#: each rater is dealt their own permutation; applied here only to the default order
#: recorded in the ledger.
MIN_REPEAT_SEPARATION = 12

RANDOM_SEED = 20260801

#: Columns the rater may see.  Everything else in the key is ground truth.
GIVEN_COLUMNS = (
    "given_reba_load_force",
    "given_reba_coupling",
    "given_reba_activity",
    "given_rula_force_load",
    "given_rula_muscle_use",
    "assumed_load_kg",
    "assumed_coupling",
)


class PackageError(RuntimeError):
    """Raised when the package cannot be built from the sample directory."""


def extract_tables() -> dict[str, Any]:
    """Copy the scoring tables out of the calculators so they cannot drift."""

    reba, rula = REBACalculator(), RULACalculator()
    return {
        "reba": {
            "a": [[list(cell) for cell in row] for row in reba.TABLE_A],
            "b": [[list(cell) for cell in row] for row in reba.TABLE_B],
            "c": [list(row) for row in reba.TABLE_C],
            "risk": {int(k): v[0] for k, v in reba.RISK_LEVELS.items()},
        },
        "rula": {
            "a": [[[list(pair) for pair in wrist] for wrist in row] for row in rula.TABLE_A],
            "b": [[list(cell) for cell in row] for row in rula.TABLE_B],
            "c": [list(row) for row in rula.TABLE_C],
            "risk": {int(k): v[0] for k, v in rula.RISK_LEVELS.items()},
        },
    }


def _load_label(load_kg: Any) -> str:
    if load_kg is None or (isinstance(load_kg, float) and pd.isna(load_kg)):
        return "not recorded"
    value = float(load_kg)
    if value == 0:
        return "none (0 kg)"
    return f"{value:g} kg"


def _panel_fractions(row: Any) -> list[list[float]] | None:
    """Each camera panel's span as a fraction of composite width, for the panel zoom.

    Cameras need not share an aspect ratio, so the panels are not thirds; the exact
    pixel boundaries are recorded when the composite is written.
    """

    raw = row.get("panel_bounds")
    width = row.get("composite_width")
    if not raw or pd.isna(width) or (isinstance(raw, float) and pd.isna(raw)):
        return None
    try:
        bounds = json.loads(raw)
        total = float(width)
        return [[round(a / total, 6), round(b / total, 6)] for a, b in bounds]
    except (ValueError, TypeError, ZeroDivisionError):
        LOGGER.warning("Could not read panel bounds for %s", row.get("frame_id"))
        return None


def _int_or_zero(value: Any) -> int:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0
    return int(value)


def build_items(key: pd.DataFrame, repeats: int, seed: int) -> list[dict[str, Any]]:
    """Turn the ground-truth key into the blinded item list the page receives.

    Repeats are drawn one per (stratum, band) cell in rotation, so the
    within-rater consistency estimate is not concentrated in whichever cell happens
    to be easiest to score.
    """

    rng = random.Random(seed)
    base: list[dict[str, Any]] = []
    for _, row in key.iterrows():
        base.append(
            {
                "key": str(row["frame_id"]),
                "frame_id": str(row["frame_id"]),
                "image": str(row["image_filename"]),
                "panels": _panel_fractions(row),
                "given": {
                    "reba_load_force": _int_or_zero(row.get("given_reba_load_force")),
                    "reba_coupling": _int_or_zero(row.get("given_reba_coupling")),
                    "reba_activity": _int_or_zero(row.get("given_reba_activity")),
                    "rula_force_load": _int_or_zero(row.get("given_rula_force_load")),
                    "rula_muscle_use": _int_or_zero(row.get("given_rula_muscle_use")),
                    "load_label": _load_label(row.get("assumed_load_kg")),
                    "coupling_label": str(row.get("assumed_coupling") or "not recorded"),
                },
            }
        )

    if repeats <= 0:
        return base
    if repeats > len(base):
        raise PackageError(f"cannot repeat {repeats} of {len(base)} frames")

    cells: dict[str, list[str]] = {}
    for _, row in key.iterrows():
        cells.setdefault(f"{row['stratum']}:{row['band']}", []).append(str(row["frame_id"]))
    for frames in cells.values():
        rng.shuffle(frames)

    chosen: list[str] = []
    cell_names = sorted(cells)
    while len(chosen) < repeats:
        progressed = False
        for name in cell_names:
            if len(chosen) >= repeats:
                break
            if cells[name]:
                chosen.append(cells[name].pop())
                progressed = True
        if not progressed:
            break

    by_id = {item["frame_id"]: item for item in base}
    items = list(base)
    for frame_id in chosen:
        source = by_id[frame_id]
        items.append({**source, "key": f"{frame_id}#repeat", "repeat_of": frame_id})

    rng.shuffle(items)
    return items


def order_for_rater(
    items: list[dict[str, Any]], rater: str, seed: int, minimum: int = MIN_REPEAT_SEPARATION
) -> list[dict[str, Any]]:
    """Deal one rater their own image order, repeats kept apart.

    Used only when order randomisation is asked for.  The default is a single shared
    order: every reviewer meets image 12 as the same frame, which makes the study
    trivial to audit and costs nothing analytically, because ratings are joined back
    on ``frame_id`` and never on position.  The trade is that any fatigue effect lands
    on the same frames for everyone instead of being spread over the set.
    """

    rng = random.Random(f"{seed}:{rater.strip().upper()}")
    ordered = list(items)
    rng.shuffle(ordered)
    return _enforce_separation(ordered, minimum, rng)


def _enforce_separation(items: list[dict[str, Any]], minimum: int, rng: random.Random) -> list[dict[str, Any]]:
    """Push repeats away from their originals in the given ordering."""

    for _ in range(200):
        positions = {item["key"]: index for index, item in enumerate(items)}
        violations = [
            item
            for item in items
            if item.get("repeat_of")
            and abs(positions[item["key"]] - positions[item["repeat_of"]]) < minimum
        ]
        if not violations:
            break
        for item in violations:
            index = positions[item["key"]]
            target = rng.randrange(len(items))
            items[index], items[target] = items[target], items[index]
    return items


def render_html(
    items: list[dict[str, Any]], images_dir_name: str, randomise_order: bool = False
) -> str:
    if not TEMPLATE_PATH.is_file():
        raise PackageError(f"template not found: {TEMPLATE_PATH}")
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    config = {
        "study_id": STUDY_ID,
        "package_version": PACKAGE_VERSION,
        "images_dir": images_dir_name,
        "order_salt": 0x5F3A21,
        "min_repeat_separation": MIN_REPEAT_SEPARATION,
        "randomise_order": bool(randomise_order),
    }
    replacements = {
        "{{TITLE}}": TITLE,
        "{{SUBTITLE}}": SUBTITLE,
        "{{ITEMS_JSON}}": json.dumps(items, separators=(",", ":")),
        "{{TABLES_JSON}}": json.dumps(extract_tables(), separators=(",", ":")),
        "{{CONFIG_JSON}}": json.dumps(config, separators=(",", ":")),
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    leftover = [token for token in replacements if token in template]
    if leftover:
        raise PackageError(f"template still contains {leftover}")
    return template


RATER_README = """\
Posture assessment - expert rating
==================================

There are two ways to do this. They ask exactly the same questions and produce
exactly the same scores, so pick whichever you prefer. Please use only one.


OPTION 1 - THE RATING PAGE  (recommended)
-----------------------------------------
1. Double-click `index.html`. Any modern browser works (Chrome, Edge, Firefox,
   Safari). Nothing to install, no internet needed.
2. Read the first screen. It matters - especially the note about load, coupling
   and activity being given to you rather than judged.
3. Enter your rater ID and click Start.
4. Score every image. Progress saves automatically as you go.
5. Click "Finish & export" and download BOTH files. Send them back.

Why this one: the image is bigger, you can zoom into a single camera view, and
it keeps your place automatically.


OPTION 2 - THE EXCEL WORKBOOK
-----------------------------
1. Open the workbook with YOUR rater ID in the name:
  {{WORKBOOKS}}
2. Read the Instructions sheet.
3. Work through the sheets F001, F002, ... one image per sheet. Answer the
   drop-downs in the cream-shaded cells; the score is calculated for you.
4. Use the Index sheet to see what is left, or to jump to a particular image.
5. Save and send the workbook back.

Each sheet has an "Open full-size image" link if you want to zoom in.


HOW LONG IT TAKES
-----------------
Roughly 2-3 minutes per image once you settle in - so about 2-3 hours in total.
You do NOT have to do it in one sitting; both options keep your work.

With the rating page: progress is stored in that browser on that machine.
Clicking Start again with the same rater ID picks up where you left off. To move
to another computer, click "Save a copy" and use "Restore from a saved file"
there.


IMPORTANT
---------
- Score the more severely loaded side.
- Load, coupling and activity are given to you per image. Please use the stated
  values even if your own reading of the scene differs.
- If an image cannot be scored, say so and give a reason. Please do not guess.
- Every reviewer sees the same images in the same order, so image 12 is the same
  frame for all of you. Please still do not discuss individual images until
  everyone has finished - your independent judgement is the whole point.


WHAT IS IN THIS FOLDER
----------------------
  index.html               the rating page - start here
  scoring_workbook_*.xlsx  the Excel alternative, one per reviewer
  images/                  the frames, in anonymised order
  README.txt               this file
"""


def build_package(
    sample_dir: Path,
    package_dir: Path,
    repeats: int,
    seed: int,
    make_zip: bool,
    raters: list[str] | None = None,
    randomise_order: bool = False,
) -> Path:
    key_path = sample_dir / "ground_truth_key.csv"
    images_source = sample_dir / "images"
    if not key_path.is_file():
        raise PackageError(f"ground truth key not found: {key_path}. Run validation.sample_frames first.")
    if not images_source.is_dir():
        raise PackageError(f"composites not found: {images_source}. Run validation.sample_frames without --no-images.")

    key = pd.read_csv(key_path)
    missing_columns = [name for name in ("frame_id", "image_filename", "stratum", "band") if name not in key.columns]
    if missing_columns:
        raise PackageError(f"{key_path} is missing columns {missing_columns}")

    missing_images = [
        name for name in key["image_filename"] if not (images_source / str(name)).is_file()
    ]
    if missing_images:
        raise PackageError(
            f"{len(missing_images)} composite(s) are missing from {images_source}, "
            f"e.g. {missing_images[:5]}"
        )

    rng = random.Random(seed)
    items = build_items(key, repeats, seed)
    items = _enforce_separation(items, MIN_REPEAT_SEPARATION, rng)

    if package_dir.exists():
        LOGGER.info("Replacing existing package at %s", package_dir)
        shutil.rmtree(package_dir)
    images_target = package_dir / "images"
    images_target.mkdir(parents=True, exist_ok=True)
    for name in sorted(set(key["image_filename"])):
        shutil.copy2(images_source / str(name), images_target / str(name))

    (package_dir / "index.html").write_text(
        render_html(items, "images", randomise_order), encoding="utf-8"
    )

    # The Excel alternative. One workbook per rater, because a workbook's sheet order is
    # fixed when it is written and every rater should meet the images in a different one.
    from validation.build_workbook import build as build_workbook

    workbooks: list[str] = []
    sheet_map: list[dict[str, Any]] = []
    for rater in raters or [""]:
        ordered = order_for_rater(items, rater, seed) if (rater and randomise_order) else items
        name = f"scoring_workbook_{rater}.xlsx" if rater else "scoring_workbook.xlsx"
        build_workbook(ordered, images_source, package_dir / name, rater)
        workbooks.append(name)
        for position, item in enumerate(ordered, start=1):
            sheet_map.append(
                {
                    "rater": rater or "(shared)",
                    "workbook": name,
                    "sheet": f"F{position:03d}",
                    "position": position,
                    "item_key": item["key"],
                    "frame_id": item["frame_id"],
                    "is_repeat": bool(item.get("repeat_of")),
                }
            )

    (package_dir / "README.txt").write_text(
        RATER_README.replace("{{WORKBOOKS}}", "\n  ".join(workbooks)), encoding="utf-8"
    )

    unique = sum(1 for item in items if not item.get("repeat_of"))
    LOGGER.info(
        "Package: %d items (%d unique frames + %d repeats), %d image file(s), %d workbook(s)",
        len(items),
        unique,
        len(items) - unique,
        len(set(key["image_filename"])),
        len(workbooks),
    )

    if make_zip:
        # One archive holding every reviewer's workbook.  When the order is shared the
        # workbooks differ only by the name on them, so a single link to distribute is
        # simpler and smaller than one 40 MB copy of the frames per reviewer.  With
        # randomised order each reviewer must receive only their own file, so the
        # package is split.
        targets: list[tuple[str, str]] = (
            [(rater, f"scoring_workbook_{rater}.xlsx") for rater in raters or []]
            if randomise_order and raters
            else [("", "")]
        )
        for rater, workbook_name in targets:
            suffix = f"_{rater}" if rater else ""
            archive = package_dir.parent / f"{package_dir.name}{suffix}.zip"
            if archive.exists():
                archive.unlink()
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
                for path in sorted(package_dir.rglob("*")):
                    if not path.is_file():
                        continue
                    name = path.name
                    if name.startswith("scoring_workbook") and workbook_name and name != workbook_name:
                        continue
                    bundle.write(path, Path(f"{package_dir.name}{suffix}") / path.relative_to(package_dir))
            LOGGER.info("Wrote %s (%.1f MB)", archive.name, archive.stat().st_size / 1e6)

    # The mapping from presentation item back to frame, kept with the study lead.
    ledger = pd.DataFrame(
        [
            {
                "default_order": index + 1,
                "item_key": item["key"],
                "frame_id": item["frame_id"],
                "is_repeat": bool(item.get("repeat_of")),
                "repeat_of": item.get("repeat_of", ""),
                "image": item["image"],
            }
            for index, item in enumerate(items)
        ]
    )
    ledger_path = sample_dir / "package_item_ledger.csv"
    ledger.to_csv(ledger_path, index=False)
    LOGGER.info("Wrote %s", ledger_path)

    # Which workbook sheet holds which frame, per rater.  The analysis does not need it
    # -- every sheet carries its own frame id -- but it lets the study lead audit a
    # returned workbook without opening it.
    map_path = sample_dir / "workbook_sheet_map.csv"
    pd.DataFrame(sheet_map).to_csv(map_path, index=False)
    LOGGER.info("Wrote %s", map_path)
    return package_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument(
        "--repeats", type=int, default=DEFAULT_REPEATS,
        help="frames shown twice, for within-rater consistency (0 disables)",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--raters", default=None,
        help="comma-separated rater IDs; builds one Excel workbook each, in its own order",
    )
    parser.add_argument(
        "--randomise-order", action="store_true",
        help="give each reviewer their own image order instead of one shared sequence",
    )
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )
    raters = [name.strip() for name in (args.raters or "").split(",") if name.strip()]
    try:
        package = build_package(
            args.sample_dir, args.package_dir, args.repeats, args.seed, not args.no_zip,
            raters, args.randomise_order,
        )
    except PackageError as exc:
        LOGGER.error("%s", exc)
        return 1
    print(f"\nExpert package: {package}")
    print("Reviewers open index.html, or the workbook with their own ID — both ask the")
    print("same questions and are scored by the same code.")
    print(f"Keep {args.sample_dir / 'ground_truth_key.csv'} back until scoring is complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
