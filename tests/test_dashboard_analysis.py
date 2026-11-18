import json

import cv2
import numpy as np
import pytest

from dashboard.analysis_service import (
    DashboardDataError,
    body_map_figure,
    body_region_risk,
    derive_findings,
    ergonomic_summary_metrics,
    interpret_frame,
    load_result_bundle,
    read_video_frame,
    record_dataframe,
    region_heatmap_figure,
    risk_distribution_figure,
    risk_indicator_figure,
    severity_label,
    stick_figure_3d,
)
from dashboard.viz_theme import INK_MUTED, STATUS


def _score(score, side, intermediate):
    return {
        "final_score": score,
        "risk_level": "high",
        "action_level": "Investigate and change soon",
        "validity_status": "valid_with_limitations",
        "intermediate_scores": intermediate,
        "raw_measurements": {"trunk_flexion_deg": 42.0},
        "units": {"trunk_flexion_deg": "degrees"},
        "missing_inputs": [],
        "invalid_inputs": [],
        "warnings": [],
        "side": side,
    }


def _record(frame=12, rula_score=6, reba_score=9):
    rula_left = _score(
        rula_score,
        "left",
        {"neck": 4, "trunk": 5, "legs": 1, "upper_arm": 5, "lower_arm": 2, "wrist": 3},
    )
    rula_right = _score(
        rula_score - 1,
        "right",
        {"neck": 4, "trunk": 5, "legs": 1, "upper_arm": 2, "lower_arm": 1, "wrist": 1},
    )
    reba_left = _score(
        reba_score,
        "left",
        {"neck": 3, "trunk": 4, "legs": 3, "upper_arm": 5, "lower_arm": 2, "wrist": 2},
    )
    reba_right = _score(
        reba_score - 1,
        "right",
        {"neck": 3, "trunk": 4, "legs": 3, "upper_arm": 2, "lower_arm": 1, "wrist": 1},
    )
    return {
        "frame_number": frame,
        "timestamp_sec": frame / 10,
        "person_identifier": "person-0001",
        "pose_quality": {"mean_keypoint_visibility": 0.91},
        "angles": {"trunk_flexion": 42.0, "neck_flexion": 18.0},
        "rula": {
            "left": rula_left,
            "right": rula_right,
            "selected": rula_left,
            "selected_side": "left",
            "selection_reason": "Left score is higher.",
        },
        "reba": {
            "left": reba_left,
            "right": reba_right,
            "selected": reba_left,
            "selected_side": "left",
            "selection_reason": "Left score is higher.",
        },
        "pose_landmarks_normalized": {
            "11": [0.4, 0.3, 0.0, 0.9],
            "12": [0.6, 0.3, 0.0, 0.9],
            "13": [0.35, 0.5, 0.0, 0.9],
        },
        "pose_landmarks_world": {
            "11": [-0.2, 0.4, 0.0, 0.9],
            "12": [0.2, 0.4, 0.0, 0.9],
            "13": [-0.3, 0.1, 0.0, 0.9],
            "14": [0.3, 0.1, 0.0, 0.9],
            "23": [-0.15, 0.0, 0.0, 0.9],
            "24": [0.15, 0.0, 0.0, 0.9],
            "25": [-0.15, -0.4, 0.0, 0.9],
            "26": [0.15, -0.4, 0.0, 0.9],
            "27": [-0.15, -0.8, 0.0, 0.9],
            "28": [0.15, -0.8, 0.0, 0.9],
        },
    }


def test_record_dataframe_preserves_bilateral_and_angle_fields():
    dataframe = record_dataframe([_record()])
    assert dataframe.loc[0, "rula_score"] == 6
    assert dataframe.loc[0, "rula_left_score"] == 6
    assert dataframe.loc[0, "rula_right_score"] == 5
    assert dataframe.loc[0, "angle_trunk_flexion"] == 42.0


def test_body_region_mapping_and_findings_reference_supporting_frame():
    record = _record()
    regions = body_region_risk(record)
    findings = derive_findings([record])
    assert regions["left_shoulder"] > regions["right_shoulder"]
    assert regions["neck"] == 1.0
    assert any(item["frame_number"] == 12 for item in findings)
    assert any(item["title"] == "Peak REBA score: 9" for item in findings)


def test_interpretation_exposes_measurements_intermediates_and_side_reason():
    result = interpret_frame(_record(), "RULA", "trunk")
    assert result["measurements"]["trunk_flexion_deg"] == 42.0
    assert result["intermediate_scores"]["trunk"] == 5
    assert result["selection_reason"] == "Left score is higher."
    assert "not a diagnosis" in result["meaning"]


def test_incomplete_interpretation_never_returns_numeric_score():
    record = _record()
    record["rula"]["selected"] = None
    record["rula"]["left"]["missing_inputs"] = ["load_kg"]
    record["rula"]["left"]["final_score"] = None
    result = interpret_frame(record, "RULA")
    assert "incomplete" in result["headline"].lower()
    assert result["missing"] == ["load_kg"]


def test_result_bundle_requires_current_traceability_files(tmp_path):
    (tmp_path / "video_summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(DashboardDataError, match="frame_results.jsonl"):
        load_result_bundle(tmp_path)

    (tmp_path / "frame_results.jsonl").write_text(json.dumps(_record()) + "\n", encoding="utf-8")
    (tmp_path / "run_config.json").write_text("{}", encoding="utf-8")
    bundle = load_result_bundle(tmp_path)
    assert bundle["records"][0]["frame_number"] == 12


def test_frame_review_draws_recorded_landmarks(tmp_path):
    path = tmp_path / "frame.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (100, 100))
    writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
    writer.release()

    image = read_video_frame(path, 0, _record(frame=0), "left_shoulder")
    assert image.shape == (100, 100, 3)
    assert np.count_nonzero(image) > 0


def test_3d_stick_figure_preserves_frames_and_uses_canonical_y_up():
    first = _record(frame=12)
    second = _record(frame=13)
    figure = stick_figure_3d([first, second], selected_region="left_shoulder")

    # floor plane + skeleton lines + landmark markers
    assert len(figure.data) == 3
    assert figure.data[0].type == "surface"
    assert len(figure.frames) == 2
    marker_trace = figure.data[2]
    assert max(marker_trace.z) > min(marker_trace.z)
    assert figure.layout.scene.zaxis.title.text == "Vertical Y (model m)"


def test_3d_stick_figure_grounds_the_lowest_foot_on_the_floor():
    """Hip-centred landmarks put the floor wherever the feet happen to be; the
    display must translate each frame so the stance foot rests at y = 0."""

    figure = stick_figure_3d([_record(frame=12), _record(frame=13)])
    marker_trace = figure.data[2]
    lowest = min(marker_trace.z)
    assert lowest == pytest.approx(0.0, abs=1e-6)
    # The floor plane itself sits at zero.
    assert float(np.asarray(figure.data[0].z).min()) == 0.0


def test_3d_stick_figure_reports_missing_landmarks_without_numeric_fallback():
    record = _record()
    record.pop("pose_landmarks_world")
    figure = stick_figure_3d([record])

    assert not figure.data
    assert "No stored 3-D landmarks" in figure.layout.annotations[0].text


def test_analysis_visuals_and_metrics_are_derived_from_valid_observations():
    records = [_record(frame=12), _record(frame=13, rula_score=4, reba_score=7)]
    summary = {
        "rula_summary": {"sustained_high_risk_periods": [{"duration_sec": 1.5}]},
        "reba_summary": {"sustained_high_risk_periods": []},
    }
    metrics = ergonomic_summary_metrics(records, summary)

    assert metrics["rula_high_percent"] == 50.0
    assert metrics["reba_high_percent"] == 50.0
    assert metrics["peak_trunk_flexion_deg"] == 42.0
    assert metrics["rula_sustained_duration_sec"] == 1.5
    assert len(region_heatmap_figure(records).data) == 1
    assert len(risk_distribution_figure(records, "RULA").data) == 1
    assert risk_indicator_figure(6, "RULA").data[0].value == 6


def test_missing_risk_score_is_labeled_unavailable_not_zero():
    figure = risk_indicator_figure(None, "RULA")
    assert not figure.data
    assert "Not available" in figure.layout.annotations[0].text


def test_unavailable_components_remain_unavailable_in_body_map():
    record = _record()
    for method in ("rula", "reba"):
        record[method]["left"]["intermediate_scores"].pop("wrist")

    regions = body_region_risk(record)
    assert regions["left_wrist"] is None
    assert severity_label(regions["left_wrist"]) == "Unavailable"

    figure = body_map_figure(regions)
    marker_trace = figure.data[-1]
    wrist_index = list(marker_trace.customdata).index("left_wrist")
    # Unavailable is the muted ink from the shared palette, deliberately not a
    # colour that could be misread as a low-risk band.
    assert marker_trace.marker.color[wrist_index] == INK_MUTED
    assert marker_trace.marker.size[wrist_index] == 24
    assert "Unavailable" in marker_trace.text[wrist_index]


def test_3d_highlights_only_the_explicitly_selected_region():
    selected = stick_figure_3d([_record()], selected_region="left_shoulder")
    marker_trace = selected.data[2]
    colors_by_landmark = dict(zip(marker_trace.customdata, marker_trace.marker.color, strict=True))
    assert colors_by_landmark[11] == STATUS["critical"]
    assert colors_by_landmark[13] == STATUS["critical"]
    assert colors_by_landmark[12] != "#ff334f"
    assert colors_by_landmark[14] != "#ff334f"

    unselected = stick_figure_3d([_record()])
    assert "#ff334f" not in unselected.data[2].marker.color


def test_risk_gauges_use_exact_official_discrete_action_bands():
    rula = risk_indicator_figure(5, "RULA").data[0]
    assert tuple(rula.gauge.axis.range) == (0.5, 7.5)
    assert tuple(rula.gauge.axis.tickvals) == tuple(range(1, 8))
    assert [tuple(step.range) for step in rula.gauge.steps] == [
        (0.5, 2.5),
        (2.5, 4.5),
        (4.5, 6.5),
        (6.5, 7.5),
    ]

    reba = risk_indicator_figure(8, "REBA").data[0]
    assert tuple(reba.gauge.axis.range) == (0.5, 15.5)
    assert tuple(reba.gauge.axis.tickvals) == tuple(range(1, 16))
    assert [tuple(step.range) for step in reba.gauge.steps] == [
        (0.5, 1.5),
        (1.5, 3.5),
        (3.5, 7.5),
        (7.5, 10.5),
        (10.5, 15.5),
    ]


def test_summary_percentages_preserve_missing_denominators():
    empty = ergonomic_summary_metrics([], {})
    assert empty["rula_valid_percent"] is None
    assert empty["rula_high_percent"] is None
    assert empty["reba_valid_percent"] is None
    assert empty["reba_high_percent"] is None

    incomplete = _record()
    incomplete["rula"]["selected"] = None
    incomplete["reba"]["selected"] = None
    metrics = ergonomic_summary_metrics([incomplete], {})
    assert metrics["rula_valid_percent"] == 0.0
    assert metrics["rula_high_percent"] is None
    assert metrics["reba_valid_percent"] == 0.0
    assert metrics["reba_high_percent"] is None


def test_incomplete_interpretation_combines_bilateral_evidence():
    record = _record()
    record["rula"]["selected"] = None
    left = record["rula"]["left"]
    right = record["rula"]["right"]
    left["validity_status"] = "missing_required_task_input"
    left["missing_inputs"] = ["load_kg"]
    left["invalid_inputs"] = ["left_invalid"]
    left["warnings"] = ["left warning"]
    right["validity_status"] = "insufficient_pose_quality"
    right["missing_inputs"] = ["wrist_flexion_deg"]
    right["invalid_inputs"] = ["right_invalid"]
    right["warnings"] = ["right warning"]

    result = interpret_frame(record, "RULA")
    assert result["status"] == "bilateral_incomplete"
    assert result["bilateral_statuses"] == {
        "left": "missing_required_task_input",
        "right": "insufficient_pose_quality",
    }
    assert result["missing_inputs"] == ["load_kg", "wrist_flexion_deg"]
    assert result["invalid_inputs"] == ["left_invalid", "right_invalid"]
    assert result["missing"] == [
        "left_invalid",
        "load_kg",
        "right_invalid",
        "wrist_flexion_deg",
    ]
    assert result["warnings"] == ["left warning", "right warning"]


def test_findings_tolerate_records_with_no_component_scores():
    record = _record()
    for method in ("rula", "reba"):
        for side in ("left", "right"):
            record[method][side]["intermediate_scores"] = {}

    findings = derive_findings([record])
    assert any(item["title"] == "Peak RULA score: 6" for item in findings)
    assert not any("posture component is elevated" in item["title"] for item in findings)
