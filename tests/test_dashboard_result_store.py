import json
import os

import pytest

from dashboard.result_store import (
    ResultStoreError,
    build_legacy_camera_refs,
    clear_result_store_cache,
    open_result_store,
)


def _record(
    frame: int,
    *,
    angle: float,
    name: str = "Participant α",
    score: int | None = None,
) -> dict:
    selected = {"final_score": score} if score is not None else None
    return {
        "frame_number": frame,
        "timestamp_sec": frame / 10,
        "person_identifier": name,
        "angles": {"trunk_flexion": angle},
        "pose_quality": {"mean_keypoint_visibility": 0.95},
        "rula": {"selected": selected},
        "reba": {"selected": selected},
        "pose_landmarks_normalized_by_camera": {"0": {"11": [0.25, 0.5, 0.0, 0.95]}},
        "pose_landmarks_world_by_camera": {"0": {"11": [-0.2, 0.4, 0.1, 0.95]}},
        "note": "ergonomie – Δ",
    }


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def _clear_store_cache():
    clear_result_store_cache()
    yield
    clear_result_store_cache()


def test_indexes_byte_offsets_and_retrieves_exact_unicode_record(tmp_path):
    path = tmp_path / "frame_results.jsonl"
    records = [_record(1, angle=5.0), _record(2, angle=25.0, name="参与者 01")]
    _write_jsonl(path, records)

    store = open_result_store(path)

    assert len(store) == 2
    assert store.entries[1].offset > store.entries[0].offset
    assert store.load_record_at_offset(store.entries[1].offset) == records[1]
    assert store.load_record(1)["person_identifier"] == "参与者 01"
    assert store.load_record(1)["note"] == "ergonomie – Δ"


def test_lightweight_records_exclude_heavy_landmarks_and_are_defensive_copies(tmp_path):
    path = tmp_path / "frame_results.jsonl"
    _write_jsonl(path, [_record(1, angle=15.0)])
    store = open_result_store(path)

    first = store.lightweight_records()
    assert "pose_landmarks_normalized_by_camera" not in first[0]
    assert "pose_landmarks_world_by_camera" not in first[0]
    assert first[0]["angles"] == {"trunk_flexion": 15.0}

    first[0]["angles"]["trunk_flexion"] = 999
    assert store.lightweight_records()[0]["angles"]["trunk_flexion"] == 15.0


def test_lightweight_schema_keeps_analysis_fields_and_discards_full_trace(tmp_path):
    path = tmp_path / "frame_results.jsonl"
    method_result = {
        "final_score": 6,
        "validity_status": "valid_with_limitations",
        "risk_level": "high",
        "action_level": "Investigate soon",
        "action_category": 3,
        "intermediate_scores": {"trunk": 4, "upper_arm": 3},
        "missing_inputs": ["load_kg"],
        "invalid_inputs": ["camera_tilt"],
        "warnings": ["Model-estimated angle"],
        "side": "left",
        "raw_measurements": {"trunk_flexion_deg": 42.0},
        "units": {"trunk_flexion_deg": "degrees"},
        "context": {"input_source": "trace-only path"},
        "method_version": "trace-only version",
    }
    record = _record(1, angle=42.0)
    record.update(
        {
            "software_version": "4.0.0-audit",
            "calibration_status": "not_calibrated",
            "timestamp_source": ["decoder_pts"] * 3,
            "multi_view_selection": {"trunk_flexion": "side"},
            "camera_frame_refs": [{"camera_index": 0, "source_frame_number": 1}],
            "synchronization": {"status": "within_tolerance", "spread_ms": 4.2},
            "input_source": "discarded duplicate source path",
            "coordinate_convention": {"units": "model-estimated metres"},
        }
    )
    method = {
        "assessment_type": "RULA",
        "selected_side": "left",
        "selection_reason": "Left score is higher.",
        "left": method_result,
        "right": {**method_result, "final_score": 4, "side": "right"},
        "selected": method_result,
        "method_version": "RULA 1993",
        "full_trace_only": {"table_a": [1, 2, 3]},
    }
    record["rula"] = method
    record["reba"] = {**method, "assessment_type": "REBA"}
    _write_jsonl(path, [record])

    store = open_result_store(path)
    lightweight = store.lightweight_records()[0]

    assert lightweight["frame_number"] == 1
    assert lightweight["timestamp_sec"] == 0.1
    assert lightweight["person_identifier"] == "Participant α"
    assert lightweight["angles"] == {"trunk_flexion": 42.0}
    assert lightweight["pose_quality"]["mean_keypoint_visibility"] == 0.95
    assert lightweight["multi_view_selection"] == {"trunk_flexion": "side"}
    assert lightweight["camera_frame_refs"][0]["source_frame_number"] == 1
    assert lightweight["synchronization"]["spread_ms"] == 4.2
    assert lightweight["rula"]["left"]["final_score"] == 6
    assert lightweight["rula"]["right"]["final_score"] == 4
    assert lightweight["rula"]["left"]["intermediate_scores"]["trunk"] == 4
    assert lightweight["rula"]["left"]["missing_inputs"] == ["load_kg"]
    assert lightweight["rula"]["left"]["invalid_inputs"] == ["camera_tilt"]
    assert lightweight["rula"]["left"]["warnings"] == ["Model-estimated angle"]
    assert lightweight["rula"]["selected"]["action_level"] == "Investigate soon"
    assert lightweight["rula"]["selected_side"] == "left"
    assert lightweight["rula"]["selection_reason"] == "Left score is higher."

    assert "pose_landmarks_world_by_camera" not in lightweight
    assert "pose_landmarks_normalized_by_camera" not in lightweight
    assert "input_source" not in lightweight
    assert "coordinate_convention" not in lightweight
    assert "raw_measurements" not in lightweight["rula"]["left"]
    assert "units" not in lightweight["rula"]["left"]
    assert "context" not in lightweight["rula"]["left"]
    assert "method_version" not in lightweight["rula"]["left"]
    assert "method_version" not in lightweight["rula"]
    assert "full_trace_only" not in lightweight["rula"]

    assert store.load_record(0) == record


def test_cache_reuses_unchanged_file_and_invalidates_on_file_change(tmp_path):
    path = tmp_path / "frame_results.jsonl"
    _write_jsonl(path, [_record(1, angle=10.0)])

    first = open_result_store(path)
    assert open_result_store(path) is first

    _write_jsonl(path, [_record(1, angle=10.0), _record(2, angle=20.0)])
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    second = open_result_store(path)
    assert second is not first
    assert len(second) == 2
    with pytest.raises(ResultStoreError, match="stale"):
        first.load_record(0)


def test_malformed_json_reports_physical_line_number(tmp_path):
    path = tmp_path / "frame_results.jsonl"
    path.write_text(
        json.dumps(_record(1, angle=10.0)) + "\n\n{not json}\n",
        encoding="utf-8",
    )

    with pytest.raises(ResultStoreError, match=r"malformed JSON on line 3"):
        open_result_store(path)


def test_landmark_subset_is_deterministic_stride_sample_and_preserves_peaks(tmp_path):
    path = tmp_path / "frame_results.jsonl"
    records = [
        _record(0, angle=0.0, score=1),
        _record(1, angle=10.0, score=2),
        _record(2, angle=90.0, score=7),
        _record(3, angle=20.0, score=3),
        _record(4, angle=5.0, score=2),
        _record(5, angle=15.0, score=2),
    ]
    _write_jsonl(path, records)
    store = open_result_store(path)

    first = store.load_landmark_subset(stride=4)
    second = store.load_landmark_subset(stride=4)

    # Stride contributes 0 and 4; endpoints contribute 5; angle and score
    # peaks both preserve frame 2.
    assert [record["frame_number"] for record in first] == [0, 2, 4, 5]
    assert first == second


def test_landmark_subset_filters_person_and_rejects_invalid_stride(tmp_path):
    path = tmp_path / "frame_results.jsonl"
    _write_jsonl(
        path,
        [
            _record(0, angle=5.0, name="person-a"),
            _record(1, angle=10.0, name="person-b"),
            _record(2, angle=15.0, name="person-a"),
        ],
    )
    store = open_result_store(path)

    subset = store.load_landmark_subset(stride=10, person_identifier="person-a")
    assert [record["frame_number"] for record in subset] == [0, 2]
    with pytest.raises(ResultStoreError, match="positive integer"):
        store.load_landmark_subset(stride=0)


def test_legacy_camera_refs_never_fabricate_exact_offsets():
    refs = build_legacy_camera_refs(
        {"frame_number": 42, "timestamp_sec": 1.4},
        [{"path": "front.mp4"}, {"path": "side.mp4"}, {"path": "iso.mp4"}],
        {"front": 0, "side": 1, "iso": 2},
    )

    assert [item["role"] for item in refs] == ["front", "side", "iso"]
    assert all(item["source_frame_number"] == 42 for item in refs)
    assert all(item["sync_delta_ms"] is None for item in refs)
    assert all(item["synchronization_status"] == "exact_offsets_unknown" for item in refs)


def test_lightweight_store_preserves_canonical_camera_evidence(tmp_path):
    path = tmp_path / "frame_results.jsonl"
    record = _record(1, angle=15.0)
    record["camera_frame_references"] = [
        {
            "camera_index": 0,
            "camera_role": "front",
            "source_frame_number": 1,
            "timestamp_sec": 0.1,
        }
    ]
    record["angles_by_camera"] = {
        "0": {"trunk_flexion": 15.0},
        "1": {"trunk_flexion": 19.0},
    }
    record["synchronization"] = {"status": "within_tolerance"}
    _write_jsonl(path, [record])

    lightweight = open_result_store(path).lightweight_records()[0]

    assert lightweight["camera_frame_references"][0]["camera_role"] == "front"
    assert lightweight["angles_by_camera"]["1"]["trunk_flexion"] == 19.0
    assert lightweight["synchronization"]["status"] == "within_tolerance"


def test_task_conditions_reach_the_lightweight_record(tmp_path):
    """The Results page reads conditions from the index, not the full record."""

    from dashboard.result_store import _lightweight_record

    record = {
        "frame_number": 1,
        "task_conditions": {
            "values": {"trunk_twisted": True, "neck_side_bent": False, "seated": None},
            "sources": {"trunk_twisted": "derived", "load_kg": "assumed"},
            "conflicts": [],
            "evidence": {
                "trunk_twisted": {
                    "measured_deg": 40.0,
                    "neutral_deg": 10.0,
                    "departure_deg": 30.0,
                    "threshold_deg": 20.0,
                    "measurement": "trunk_twist",
                }
            },
        },
    }
    light = _lightweight_record(record)["task_conditions"]

    # Only conditions that actually fired are kept; false and unknown are the
    # overwhelming majority and carry no information for this view.
    assert light["values"] == {"trunk_twisted": True}
    assert light["sources"]["load_kg"] == "assumed"
    # The departure survives; the rest of the evidence stays in the JSONL.
    assert light["evidence"]["trunk_twisted"]["departure_deg"] == 30.0
    assert "threshold_deg" not in light["evidence"]["trunk_twisted"]


def test_a_record_without_conditions_is_handled():
    from dashboard.result_store import _lightweight_record

    assert "task_conditions" not in _lightweight_record({"frame_number": 1})
