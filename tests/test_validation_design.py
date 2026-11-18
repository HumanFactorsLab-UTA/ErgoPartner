"""The two-stratum design must hold its hard quotas and give ground only where it must.

The design exists because stratifying on risk band alone selects pose-estimation
transients for the rare cells.  Two properties keep it honest: the (stratum, band)
counts are exact, and anything it could not achieve -- task balance, in practice --
is reported rather than absorbed.  Both are asserted here, along with the pool
construction rules that decide what is a candidate at all.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from ergonomics.reba import REBACalculator
from validation import allocate, pool

# ---------------------------------------------------------------------------
# Banding
# ---------------------------------------------------------------------------


def test_bands_are_derived_from_the_calculator_not_hardcoded():
    lookup = pool.reba_band_lookup()
    assert set(lookup) == set(REBACalculator.RISK_LEVELS)
    for score, level in REBACalculator.RISK_LEVELS.items():
        risk = level[0]
        if risk in ("Negligible", "Low"):
            assert lookup[score] == "Low"
        elif risk == "Medium":
            assert lookup[score] == "Medium"
        else:
            assert lookup[score] == "High", f"REBA {score} ({risk})"


@pytest.mark.parametrize("score,band", [(1, "Low"), (3, "Low"), (4, "Medium"),
                                        (7, "Medium"), (8, "High"), (15, "High")])
def test_band_of_matches_the_published_boundaries(score, band):
    assert pool.band_of(score) == band


def test_band_of_rejects_scores_outside_the_scale():
    assert pool.band_of(None) is None
    assert pool.band_of(float("nan")) is None
    assert pool.band_of(0) is None
    assert pool.band_of(16) is None


@pytest.mark.parametrize("folder,index", [
    ("task1_niosh_lifting", 1), ("task2_reba_stocking", 2),
    ("task3_rula_screwdriving", 3), ("unnumbered", 0),
])
def test_task_index_reads_the_leading_number(folder, index):
    assert pool.task_index(folder) == index


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _write_results_tree(root):
    """A results tree with one live run and one archived, superseded run."""

    live = root / "10fps" / "study_a" / "001" / "task1_niosh_lifting"
    archived = root / "10fps" / "study_a" / "001" / ".task1_niosh_lifting.archive-pre-fixes"
    other = root / "native" / "study_a" / "001" / "task1_niosh_lifting"
    for path in (live, archived, other):
        path.mkdir(parents=True)
        (path / "frame_results.csv").write_text("frame_number\n0\n", encoding="utf-8")
    return root


def test_archived_runs_are_not_candidates(tmp_path):
    """A retired run still parses as a task, and would compete for the task quota."""

    root = _write_results_tree(tmp_path / "results")
    tables = pool.discover_result_tables(root, "10fps")
    assert [table["task"] for table in tables] == ["task1_niosh_lifting"]
    assert not any(table["task"].startswith(".") for table in tables)


def test_variant_selection_prefers_the_widest_coverage(tmp_path):
    root = _write_results_tree(tmp_path / "results")
    extra = root / "10fps" / "study_b" / "002" / "task2_reba_stocking"
    extra.mkdir(parents=True)
    (extra / "frame_results.csv").write_text("frame_number\n0\n", encoding="utf-8")

    summary = pool.summarise_variants(root)
    assert summary["10fps"] == {"recordings": 2, "participants": 2}
    assert summary["native"] == {"recordings": 1, "participants": 1}
    assert pool.choose_variant(root, None) == "10fps"
    assert pool.choose_variant(root, "native") == "native"


def test_asking_for_a_variant_with_no_results_names_the_alternatives(tmp_path):
    root = _write_results_tree(tmp_path / "results")
    with pytest.raises(pool.PoolError, match="30fps"):
        pool.choose_variant(root, "30fps")


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------


def _recording(bands: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame_number": range(len(bands)),
            "timestamp_sec": [index * 0.1 for index in range(len(bands))],
            "band": bands,
        }
    )


def test_a_frame_is_sustained_only_when_its_band_holds_either_side():
    # A single "High" inside a run of "Medium" is the transient the design must isolate.
    frame = _recording(["Medium"] * 5 + ["High"] + ["Medium"] * 5)
    marked = pool._mark_stability(frame, half_window=2, expected_step_sec=0.1)
    assert not marked.loc[5, "sustained"], "a one-frame spike must not read as sustained"
    assert marked.loc[2, "sustained"]
    # Frames adjacent to the spike see it inside their window.
    assert not marked.loc[4, "sustained"]
    assert not marked.loc[6, "sustained"]


def test_run_length_is_recorded_so_the_threshold_can_be_varied_later():
    frame = _recording(["Low"] * 3 + ["High"] * 7)
    marked = pool._mark_stability(frame, half_window=2, expected_step_sec=0.1)
    assert marked.loc[0, "band_run_frames"] == 3
    assert marked.loc[5, "band_run_frames"] == 7
    assert marked.loc[5, "band_run_sec"] == pytest.approx(0.6)


def test_edges_of_a_recording_are_unverifiable_rather_than_transient():
    """An edge frame in a stable run is not unstable -- it simply cannot be judged.

    Filing it as transient would put frames from the middle of long stable runs into
    the stratum whose whole purpose is to isolate pose instability.
    """

    frame = _recording(["Medium"] * 10)
    marked = pool._mark_stability(frame, half_window=2, expected_step_sec=0.1)
    for edge in (0, 1, 8, 9):
        assert not marked.loc[edge, "sustained"]
        assert not marked.loc[edge, "stability_verifiable"]
    assert marked.loc[5, "sustained"]
    assert marked.loc[5, "stability_verifiable"]


def test_a_decoding_gap_does_not_masquerade_as_a_held_posture():
    frame = _recording(["Medium"] * 5)
    frame.loc[3:, "timestamp_sec"] += 30.0          # a long gap mid-window
    marked = pool._mark_stability(frame, half_window=2, expected_step_sec=0.1)
    assert not marked.loc[2, "sustained"]
    assert not marked.loc[2, "stability_verifiable"]


def test_distance_to_the_next_band_change_is_what_decides_the_stratum():
    """A frame can sit inside a long run and still be transient, so record the distance."""

    frame = _recording(["Medium"] * 20 + ["High"] + ["Medium"] * 20)
    marked = pool._mark_stability(frame, half_window=2, expected_step_sec=0.1)

    # Two samples before the spike: inside a 20-frame run, but not a held posture.
    assert marked.loc[18, "frames_to_band_change"] == 2
    assert not marked.loc[18, "sustained"]
    assert marked.loc[18, "stability_verifiable"]
    assert marked.loc[18, "band_run_frames"] == 20      # the run is long regardless

    # Three samples away, the band holds either side.
    assert marked.loc[17, "frames_to_band_change"] == 3
    assert marked.loc[17, "sustained"]

    verifiable = marked[marked["stability_verifiable"]]
    assert (verifiable.loc[verifiable["sustained"], "frames_to_band_change"] >= 3).all()
    assert (verifiable.loc[~verifiable["sustained"], "frames_to_band_change"] <= 2).all()


def test_a_band_that_never_changes_has_no_distance_to_record():
    frame = _recording(["Medium"] * 10)
    marked = pool._mark_stability(frame, half_window=2, expected_step_sec=0.1)
    assert marked["frames_to_band_change"].isna().all()
    assert marked.loc[5, "sustained"]


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------

CELLS = ["sustained:Low", "sustained:Medium", "sustained:High",
         "transient:Low", "transient:Medium", "transient:High"]


def _plentiful(participants, tasks, cells=CELLS, count=50):
    return {(p, t, c): count for p in participants for t in tasks for c in cells}


def test_cell_quotas_are_met_exactly():
    participants = [f"{n:03d}" for n in range(1, 11)]
    tasks = ["task1", "task2", "task3"]
    quota = dict.fromkeys(CELLS, 5)
    solved = allocate.solve(
        availability=_plentiful(participants, tasks),
        participants=participants, tasks=tasks, cell_types=CELLS, cell_quota=quota,
        task_target={task: 10 for task in tasks},
        min_per_participant=2, max_per_participant=4, pair_cap=3, seed=1,
    )
    drawn: dict[str, int] = {}
    for (_p, _t, cell), n in solved.counts.items():
        drawn[cell] = drawn.get(cell, 0) + n
    assert drawn == quota
    assert sum(solved.counts.values()) == 30


def test_every_participant_meets_the_floor():
    participants = [f"{n:03d}" for n in range(1, 11)]
    tasks = ["task1", "task2", "task3"]
    solved = allocate.solve(
        availability=_plentiful(participants, tasks),
        participants=participants, tasks=tasks, cell_types=CELLS,
        cell_quota=dict.fromkeys(CELLS, 5),
        task_target={task: 10 for task in tasks},
        min_per_participant=2, max_per_participant=4, pair_cap=3, seed=2,
    )
    assert set(solved.participant_totals) == set(participants)
    assert min(solved.participant_totals.values()) >= 2
    assert max(solved.participant_totals.values()) <= 4


def test_task_balance_is_relaxed_rather_than_failing_when_a_task_cannot_supply_a_cell():
    """Seated screwdriving has no sustained high-risk posture; the design must survive."""

    participants = [f"{n:03d}" for n in range(1, 11)]
    tasks = ["task1", "task2", "task3"]
    availability = _plentiful(participants, tasks)
    for participant in participants:
        del availability[(participant, "task3", "sustained:High")]

    solved = allocate.solve(
        availability=availability,
        participants=participants, tasks=tasks, cell_types=CELLS,
        cell_quota=dict.fromkeys(CELLS, 5),
        task_target={task: 10 for task in tasks},
        min_per_participant=2, max_per_participant=4, pair_cap=3, seed=3,
    )
    drawn = {(t, c): 0 for t in tasks for c in CELLS}
    for (_p, task, cell), n in solved.counts.items():
        drawn[(task, cell)] += n
    assert drawn[("task3", "sustained:High")] == 0
    assert sum(v for (_t, c), v in drawn.items() if c == "sustained:High") == 5
    assert sum(solved.task_totals.values()) == 30


def test_an_impossible_quota_reports_why_instead_of_hanging():
    participants = ["001", "002"]
    tasks = ["task1"]
    availability = {("001", "task1", "sustained:Low"): 1}
    with pytest.raises(allocate.AllocationError) as excinfo:
        allocate.solve(
            availability=availability, participants=participants, tasks=tasks,
            cell_types=["sustained:Low"], cell_quota={"sustained:Low": 40},
            task_target={"task1": 40},
            min_per_participant=1, max_per_participant=4, pair_cap=3, seed=4,
        )
    message = str(excinfo.value)
    assert "sustained:Low" in message
    assert "quota 40" in message


def test_the_floor_cannot_exceed_the_budget():
    with pytest.raises(allocate.AllocationError, match="minimum"):
        allocate.solve(
            availability={}, participants=[f"{n:03d}" for n in range(20)],
            tasks=["task1"], cell_types=["sustained:Low"], cell_quota={"sustained:Low": 10},
            task_target={"task1": 10},
            min_per_participant=3, max_per_participant=4, pair_cap=3, seed=5,
        )


def test_a_recording_never_supplies_more_than_the_cap():
    participants = ["001", "002", "003"]
    tasks = ["task1", "task2"]
    solved = allocate.solve(
        availability=_plentiful(participants, tasks),
        participants=participants, tasks=tasks, cell_types=CELLS,
        cell_quota=dict.fromkeys(CELLS, 2),
        task_target={task: 6 for task in tasks},
        min_per_participant=2, max_per_participant=4, pair_cap=2, seed=6,
    )
    per_recording: dict[tuple[str, str], int] = {}
    for (participant, task, _cell), n in solved.counts.items():
        per_recording[(participant, task)] = per_recording.get((participant, task), 0) + n
    assert max(per_recording.values()) <= 2


def test_camera_references_come_back_in_panel_order():
    record = {
        "camera_frame_references": [
            {"camera_index": 2, "camera_role": "iso", "source_frame_number": 7},
            {"camera_index": 0, "camera_role": "front", "source_frame_number": 7},
            {"camera_index": 1, "camera_role": "side", "source_frame_number": 7},
        ]
    }
    assert [ref["camera_role"] for ref in pool.camera_references(record)] == [
        "front", "side", "iso"
    ]


def test_camera_references_fall_back_to_index_order_when_roles_are_missing():
    record = {
        "camera_frame_references": [
            {"camera_index": 1, "source_frame_number": 3},
            {"camera_index": 0, "source_frame_number": 3},
        ]
    }
    assert [ref["camera_index"] for ref in pool.camera_references(record)] == [0, 1]


def test_frame_details_are_pulled_only_for_the_frames_asked_for(tmp_path):
    path = tmp_path / "frame_results.jsonl"
    path.write_text(
        "\n".join(json.dumps({"frame_number": n, "payload": n * 2}) for n in range(20)),
        encoding="utf-8",
    )
    found = pool.load_frame_details(path, {3, 11})
    assert set(found) == {3, 11}
    assert found[11]["payload"] == 22
