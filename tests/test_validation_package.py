"""What goes out to a rater must be blinded, and must score the way the pipeline does.

Two failures here would be invisible until the study was over: a package that quietly
discloses the answer, and a workbook whose formulas index the lookup tables one row
off.  Both are cheap to pin down, so they are pinned down.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ergonomics.reba import REBACalculator
from ergonomics.rula import RULACalculator
from validation import scoring
from validation.build_expert_package import build_items, order_for_rater

openpyxl = pytest.importorskip("openpyxl", reason="the workbook is optional")
PIL = pytest.importorskip("PIL", reason="the workbook is optional")

from validation import build_workbook  # noqa: E402

REBA = REBACalculator()
RULA = RULACalculator()

SEED = 20260801


def _key(frames: int = 12) -> pd.DataFrame:
    """A ground-truth key shaped like the real one, with every stratum/band cell used."""

    cells = [(s, b) for s in ("sustained", "transient") for b in ("Low", "Medium", "High")]
    rows = []
    for index in range(1, frames + 1):
        stratum, band = cells[(index - 1) % len(cells)]
        rows.append(
            {
                "frame_id": f"frame_{index:03d}",
                "image_filename": f"frame_{index:03d}.jpg",
                "stratum": stratum,
                "band": band,
                "given_reba_load_force": index % 4,
                "given_reba_coupling": index % 3,
                "given_reba_activity": index % 2,
                "given_rula_force_load": index % 3,
                "given_rula_muscle_use": index % 2,
                "assumed_load_kg": 0.0,
                "assumed_coupling": "good",
                "panel_bounds": "[[0, 100], [106, 206], [212, 312]]",
                "composite_width": 312,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Blinding
# ---------------------------------------------------------------------------


def test_items_sent_to_a_rater_disclose_nothing_but_the_frame_and_its_givens():
    items = build_items(_key(), repeats=2, seed=SEED)
    allowed = {"key", "frame_id", "image", "panels", "given", "repeat_of"}
    for item in items:
        assert set(item) <= allowed, f"unexpected field leaked to the rater: {set(item) - allowed}"
        assert set(item["given"]) == {
            "reba_load_force", "reba_coupling", "reba_activity",
            "rula_force_load", "rula_muscle_use", "load_label", "coupling_label",
        }
    serialised = repr(items)
    for secret in ("band", "stratum", "participant", "task", "pipeline", "sustained", "transient"):
        assert secret not in serialised.lower().replace("frame_id", ""), secret


def test_repeats_are_added_and_spread_over_the_cells():
    items = build_items(_key(), repeats=6, seed=SEED)
    repeats = [item for item in items if item.get("repeat_of")]
    assert len(repeats) == 6
    assert len({item["repeat_of"] for item in repeats}) == 6, "a frame was repeated twice"
    assert len({item["frame_id"] for item in items}) == 12


def test_each_rater_gets_their_own_order_with_repeats_held_apart():
    items = build_items(_key(frames=30), repeats=4, seed=SEED)
    orders = {}
    for rater in ("R01", "R02", "R03"):
        ordered = order_for_rater(items, rater, SEED, minimum=8)
        orders[rater] = [item["key"] for item in ordered]
        position = {item["key"]: index for index, item in enumerate(ordered)}
        for item in ordered:
            if item.get("repeat_of"):
                gap = abs(position[item["key"]] - position[item["repeat_of"]])
                assert gap >= 8, f"{rater}: repeat only {gap} items from its original"
    assert len({tuple(order) for order in orders.values()}) == 3, "raters share an order"
    assert all(sorted(order) == sorted(orders["R01"]) for order in orders.values())


def test_a_rater_order_is_stable_for_the_same_id():
    items = build_items(_key(), repeats=2, seed=SEED)
    first = [item["key"] for item in order_for_rater(items, "R07", SEED)]
    second = [item["key"] for item in order_for_rater(items, "R07", SEED)]
    assert first == second


def test_the_set_is_sixty_frames_unless_repeats_are_asked_for():
    """Repeats are an opt-in extra; the default set is exactly what was drawn."""

    from validation.build_expert_package import DEFAULT_REPEATS

    assert DEFAULT_REPEATS == 0
    items = build_items(_key(frames=60), repeats=DEFAULT_REPEATS, seed=SEED)
    assert len(items) == 60
    assert len({item["frame_id"] for item in items}) == 60
    assert not any(item.get("repeat_of") for item in items)


# ---------------------------------------------------------------------------
# Workbook
# ---------------------------------------------------------------------------


def test_option_labels_map_onto_bands_scoring_understands():
    for kind, pairs in build_workbook.OPTION_LABELS.items():
        keys = [key for _label, key in pairs]
        assert len(set(keys)) == len(keys), f"{kind} repeats a band"
        field = next(f for f, k in build_workbook.FIELD_OPTIONS.items() if k == kind)
        assert set(keys) == set(scoring.OBSERVATION_CHOICES[field]), kind


def test_workbook_base_scores_agree_with_the_scoring_module():
    """The drop-down base scores are what the sheet formulas add modifiers to."""

    for kind, pairs in build_workbook.OPTION_LABELS.items():
        for _label, key in pairs:
            reba, rula = build_workbook._base_scores(kind, key)
            if kind == "TrunkOpts":
                assert reba == scoring.reba_components({"trunk": key})["trunk"]
                expected = scoring.rula_components({"trunk": key})["trunk"]
                assert rula == (-1 if expected is None else expected)
            elif kind == "NeckOpts":
                assert reba == scoring.reba_components({"neck": key})["neck"]
                assert rula == scoring.rula_components({"neck": key})["neck"]
            elif kind == "WristOpts":
                assert reba == scoring.reba_components({"wrist": key})["wrist"]
                assert rula == scoring.rula_components({"wrist": key})["wrist"]


def _build_small_workbook(tmp_path, frames: int = 3, rater: str = "R01"):
    from PIL import Image

    images = tmp_path / "images"
    images.mkdir()
    key = _key(frames)
    for name in key["image_filename"]:
        Image.new("RGB", (312, 72), (90, 90, 90)).save(images / name, "JPEG")
    items = build_items(key, repeats=0, seed=SEED)
    destination = tmp_path / f"scoring_workbook_{rater}.xlsx"
    build_workbook.build(items, images, destination, rater)
    return destination, items


def test_workbook_lookup_tables_are_indexed_the_way_the_formulas_index_them(tmp_path):
    """The flattened tables plus the formulas' row arithmetic must reproduce the tables.

    An off-by-one here would still recalculate cleanly and produce plausible scores.
    """

    destination, _items = _build_small_workbook(tmp_path, frames=1)
    book = openpyxl.load_workbook(destination)
    tables = book["_tables"]

    def block(name: str) -> list[list[int]]:
        ref = book.defined_names[name].attr_text.split("!")[1].replace("$", "")
        return [[cell.value for cell in row] for row in tables[ref]]

    reba_a, reba_b, reba_c = block("RebaA"), block("RebaB"), block("RebaC")
    rula_a, rula_b, rula_c = block("RulaA"), block("RulaB"), block("RulaC")

    for trunk in range(1, 6):
        for neck in range(1, 4):
            for legs in range(1, 5):
                row = (trunk - 1) * 3 + neck
                assert reba_a[row - 1][legs - 1] == REBA.lookup_table_a(trunk, neck, legs)
    for upper in range(1, 7):
        for lower in range(1, 3):
            for wrist in range(1, 4):
                row = (upper - 1) * 2 + lower
                assert reba_b[row - 1][wrist - 1] == REBA.lookup_table_b(upper, lower, wrist)
    for score_a in range(1, 13):
        for score_b in range(1, 13):
            assert reba_c[score_a - 1][score_b - 1] == REBA.lookup_table_c(score_a, score_b)

    for upper in range(1, 7):
        for lower in range(1, 4):
            for wrist in range(1, 5):
                for twist in range(1, 3):
                    row = ((upper - 1) * 3 + (lower - 1)) * 4 + wrist
                    assert rula_a[row - 1][twist - 1] == RULA.lookup_table_a(
                        upper, lower, wrist, twist
                    )
    for neck in range(1, 7):
        for trunk in range(1, 7):
            for legs in range(1, 3):
                row = (neck - 1) * 6 + trunk
                assert rula_b[row - 1][legs - 1] == RULA.lookup_table_b(neck, trunk, legs)
    for score_a in range(1, 14):
        for score_b in range(1, 13):
            assert rula_c[min(score_a, 8) - 1][min(score_b, 7) - 1] == RULA.lookup_table_c(
                score_a, score_b
            )
    book.close()


def test_workbook_has_a_sheet_per_frame_named_in_order(tmp_path):
    destination, items = _build_small_workbook(tmp_path, frames=4)
    book = openpyxl.load_workbook(destination, read_only=True)
    frame_sheets = [n for n in book.sheetnames if n.startswith("F") and n[1:].isdigit()]
    assert frame_sheets == ["F001", "F002", "F003", "F004"]
    assert frame_sheets == sorted(frame_sheets), "tab order must follow presentation order"
    for name in ("Instructions", "Index", "Results"):
        assert name in book.sheetnames
    for hidden in ("_ref", "_tables"):
        assert book[hidden].sheet_state == "hidden"
    book.close()


def test_workbook_round_trips_a_rater_answer(tmp_path):
    """Labels a rater picks must come back as the bands scoring.py consumes."""

    destination, items = _build_small_workbook(tmp_path, frames=2)
    book = openpyxl.load_workbook(destination)
    sheet = book["F001"]
    answers = {
        "trunk": "Flexed 20–60°", "neck": "Flexed 10–20°",
        "legs": "Bilateral weight bearing, walking or sitting", "knee": "30–60°",
        "upper_arm": "Flexed 45–90°", "lower_arm": "Flexed 60–100°",
        "wrist": "Flexed or extended up to 15°",
    }
    for offset, (field, _label, kind, _help) in enumerate(build_workbook.QUESTIONS):
        if field in answers:
            sheet.cell(row=build_workbook.R_OBS + offset, column=2, value=answers[field])
        elif kind == "yesno":
            sheet.cell(row=build_workbook.R_OBS + offset, column=2,
                       value="Yes" if field == "trunk_twisted" else "No")
    sheet.cell(row=build_workbook.R_META, column=2, value="4 — Confident")
    sheet.cell(row=build_workbook.R_META + 3, column=2, value="clear view")
    book.save(destination)
    book.close()

    records = build_workbook.read_workbook(destination)
    assert len(records) == 2
    first = next(r for r in records if r["sheet"] == "F001")
    assert first["rater_id"] == "R01"
    assert first["observations"]["trunk"] == "flex_20_60"
    assert first["observations"]["neck"] == "flex_10_20"
    assert first["observations"]["trunk_twisted"] is True
    assert first["observations"]["neck_twisted"] is False
    assert first["confidence"] == "4"
    assert first["notes"] == "clear view"
    assert first["cannot_score"] is False
    assert scoring.is_complete(first["observations"])
    assert scoring.compose_reba(first["observations"], dict.fromkeys(scoring.GIVEN_FIELDS, 0))[
        "final"
    ] is not None


def test_workbook_records_an_unscorable_frame(tmp_path):
    destination, _items = _build_small_workbook(tmp_path, frames=2)
    book = openpyxl.load_workbook(destination)
    sheet = book["F002"]
    sheet.cell(row=build_workbook.R_META + 1, column=2, value="No")
    sheet.cell(row=build_workbook.R_META + 2, column=2, value="torso occluded")
    book.save(destination)
    book.close()

    record = next(r for r in build_workbook.read_workbook(destination) if r["sheet"] == "F002")
    assert record["cannot_score"] is True
    assert record["cannot_reason"] == "torso occluded"


def _delivery(tmp_path, raters, randomise=False, repeats=0):
    """Assemble a small delivery end to end, so its structure can be asserted."""

    from PIL import Image

    from validation import assemble_delivery

    sample = tmp_path / "sample"
    (sample / "images").mkdir(parents=True)
    key = _key(6)
    key["participant"] = [1, 2, 3, 1, 2, 3]
    key["task"] = ["task1_niosh_lifting"] * 3 + ["task2_reba_stocking"] * 3
    key["frame_number"] = range(6)
    key["band_run_sec"] = 1.0
    key["sampling_weight"] = 1.0
    key["pipeline_reba_final"] = [6, 2, 9, 5, 3, 10]
    key["pipeline_rula_final"] = [5, 2, 7, 4, 3, 6]
    for name in ("trunk", "neck", "legs", "upper_arm", "lower_arm", "wrist"):
        key[f"pipeline_reba_{name}"] = 2
    key.to_csv(sample / "ground_truth_key.csv", index=False)
    for name in key["image_filename"]:
        Image.new("RGB", (312, 72), (70, 70, 70)).save(sample / "images" / name, "JPEG")

    delivery = tmp_path / "delivery"
    assemble_delivery.assemble(sample, delivery, raters, repeats, SEED, randomise)
    return delivery


def test_the_delivery_keeps_the_answer_key_out_of_what_reviewers_receive(tmp_path):
    delivery = _delivery(tmp_path, ["R01", "R02", "R03", "R04", "R05"])
    send = delivery / "1_send_to_reviewers"
    private = delivery / "2_keep_private"

    assert (delivery / "START_HERE.html").is_file()
    assert (private / "ground_truth_reference.xlsx").is_file()
    assert (private / "ground_truth_key.csv").is_file()
    assert (private / "design_record.md").is_file()

    sent = [path.name for path in send.rglob("*") if path.is_file()]
    for secret in ("ground_truth_key.csv", "ground_truth_reference.xlsx",
                   "sampling_manifest.json", "design_record.md", "workbook_sheet_map.csv"):
        assert secret not in sent, f"{secret} would have been sent to the reviewers"

    import zipfile
    with zipfile.ZipFile(send / "expert_package.zip") as bundle:
        inside = bundle.namelist()
    assert not [n for n in inside if "ground_truth" in n or "manifest" in n]
    assert sum(1 for n in inside if n.endswith(".xlsx")) == 5


def test_every_reviewer_receives_the_same_frames_in_the_same_order(tmp_path):
    """The reviewers are compared to each other, so they must see one shared sequence."""

    raters = ["R01", "R02", "R03", "R04", "R05"]
    delivery = _delivery(tmp_path, raters)
    package = delivery / "1_send_to_reviewers" / "expert_package"

    orders = {}
    for rater in raters:
        book = openpyxl.load_workbook(package / f"scoring_workbook_{rater}.xlsx", read_only=True)
        orders[rater] = [
            str(book[name].cell(row=build_workbook.R_TITLE, column=8).value)
            for name in book.sheetnames
            if name.startswith("F") and name[1:].isdigit()
        ]
        book.close()
    assert len({tuple(order) for order in orders.values()}) == 1, "reviewers differ in order"
    assert len(orders["R01"]) == 6


def test_randomising_the_order_splits_the_package_one_zip_per_reviewer(tmp_path):
    raters = ["R01", "R02", "R03"]
    delivery = _delivery(tmp_path, raters, randomise=True)
    send = delivery / "1_send_to_reviewers"
    for rater in raters:
        assert (send / f"expert_package_{rater}.zip").is_file()
    assert not (send / "expert_package.zip").is_file()


def test_workbook_ignores_an_answer_that_is_not_a_listed_option(tmp_path, caplog):
    destination, _items = _build_small_workbook(tmp_path, frames=1)
    book = openpyxl.load_workbook(destination)
    trunk_row = build_workbook.R_OBS + [q[0] for q in build_workbook.QUESTIONS].index("trunk")
    book["F001"].cell(row=trunk_row, column=2, value="bent over a lot")
    book.save(destination)
    book.close()

    record = build_workbook.read_workbook(destination)[0]
    assert "trunk" not in record["observations"]
    assert not scoring.is_complete(record["observations"])
