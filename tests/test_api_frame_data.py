from __future__ import annotations

import csv
import json
from io import StringIO

import pytest

from api import frame_data
from dashboard.result_store import clear_result_store_cache


def _side(
    score: int | None,
    side: str,
    *,
    validity: str = "valid",
    components: dict[str, int] | None = None,
) -> dict:
    return {
        "final_score": score,
        # Deliberately non-canonical: the API must derive the published label
        # from the score instead of trusting historic wording in the trace.
        "risk_level": "historic wording",
        "action_level": "historic action",
        "validity_status": validity,
        "intermediate_scores": components or {},
        "missing_inputs": [] if score is not None else ["wrist_flexion_deg"],
        "invalid_inputs": [],
        "warnings": [],
        "side": side,
    }


def _method(
    score: int | None,
    method: str,
    *,
    validity: str = "valid",
) -> dict:
    components = (
        {"trunk": 4, "upper_arm": 5, "table_c": score or 0}
        if method == "reba"
        else {"trunk": 5, "upper_arm": 4, "table_c": score or 0}
    )
    left = _side(score, "left", validity=validity, components=components)
    right_score = None if score is None else max(1, score - 1)
    right = _side(right_score, "right", validity=validity, components=components)
    return {
        "assessment_type": method.upper(),
        "left": left,
        "right": right,
        "selected": left if score is not None else None,
        "selected_side": "left" if score is not None else None,
        "selection_reason": "Left score is higher." if score is not None else "",
    }


def _record(
    frame: int,
    timestamp: float,
    *,
    person: str = "person-a",
    reba: int | None = 4,
    rula: int | None = 3,
    validity: str = "valid",
) -> dict:
    return {
        "software_version": "4.0-test",
        "frame_number": frame,
        "timestamp_sec": timestamp,
        "person_identifier": person,
        "input_source": r"D:\private\participant\camera.mp4",
        "calibration_status": "not_calibrated",
        "pose_quality": {
            "mean_keypoint_visibility": 0.91,
            "status": "usable",
            "camera_view_status": "side",
            "wrist_data_source": "pose",
            "warnings": ["model-estimated"],
        },
        "angles": {
            "trunk_flexion": 42.5,
            "neck_flexion": 18.0,
            "left_upper_arm_flexion": 55.0,
        },
        "task_conditions": {
            "values": {"trunk_twisted": True, "seated": False},
            "sources": {"trunk_twisted": "derived"},
        },
        "rula": _method(rula, "rula", validity=validity),
        "reba": _method(reba, "reba", validity=validity),
        "pose_landmarks_world_by_camera": {
            "0": {"11": [0.1, 0.2, 0.3, 0.9]},
        },
    }


@pytest.fixture
def result_session(tmp_path, monkeypatch):
    records = [
        _record(0, 0.0, reba=2, rula=1),
        _record(1, 0.1, reba=3, rula=2),
        _record(2, 0.2, reba=4, rula=4),
        _record(
            3,
            0.3,
            reba=None,
            rula=None,
            validity="insufficient_pose_quality",
        ),
        _record(
            4,
            0.4,
            reba=None,
            rula=None,
            validity="insufficient_pose_quality",
        ),
        _record(5, 0.5, reba=9, rula=7),
        _record(6, 2.0, reba=9, rula=7),
        _record(100, 0.0, person="person-b", reba=12, rula=7),
    ]
    path = tmp_path / "frame_results.jsonl"
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    monkeypatch.setattr(frame_data, "resolve_result_dir", lambda _: str(tmp_path))
    monkeypatch.setattr(
        frame_data,
        "load_run_config",
        lambda _: {"temporal_aggregation": {"maximum_gap_sec": 0.25}},
    )
    clear_result_store_cache()
    yield "study__001__task__native", records
    clear_result_store_cache()


def test_observations_are_complete_paginated_and_never_use_chart_decimation(result_session):
    session_id, _ = result_session

    first = frame_data.observations_payload(
        session_id,
        "REBA",
        page=1,
        limit=3,
        person="person-a",
    )
    second = frame_data.observations_payload(
        session_id,
        "reba",
        page=2,
        limit=3,
        person="person-a",
    )

    assert first["total"] == 7
    assert first["count"] == 3
    assert first["hasNext"] is True
    assert [row["frame"] for row in first["rows"]] == [0, 1, 2]
    assert [row["frame"] for row in second["rows"]] == [3, 4, 5]
    assert first["rows"][2]["components"]["trunk"] == 4
    assert first["rows"][2]["sides"]["right"]["score"] == 3
    assert first["rows"][2]["angles"]["trunk_flexion"] == pytest.approx(42.5)


def test_band_and_shared_or_validity_status_filters_use_official_labels(result_session):
    session_id, _ = result_session

    very_high = frame_data.observations_payload(
        session_id,
        "reba",
        band="very-high",
        page=1,
        limit=20,
    )
    critical = frame_data.observations_payload(
        session_id,
        "reba",
        status="critical",
        page=1,
        limit=20,
    )
    incomplete = frame_data.observations_payload(
        session_id,
        "reba",
        status="insufficient_pose_quality",
        page=1,
        limit=20,
    )
    rula = frame_data.observations_payload(
        session_id,
        "rula",
        band="change_now",
        page=1,
        limit=20,
    )

    assert very_high["total"] == 1
    assert very_high["rows"][0]["classification"] == "Very high"
    assert very_high["rows"][0]["score"] == 12
    assert critical["total"] == 1
    assert [row["classification"] for row in incomplete["rows"]] == [
        "Unscored",
        "Unscored",
    ]
    # Stored RULA says "historic wording"; the shared official band says this.
    assert rula["total"] == 3
    assert {row["classification"] for row in rula["rows"]} == {"Change now"}


def test_risk_periods_cover_every_band_and_keep_unscored_gaps(result_session):
    session_id, _ = result_session

    payload = frame_data.risk_periods_payload(
        session_id,
        "reba",
        person="person-a",
    )
    periods = payload["periods"]

    assert payload["maximumGapSec"] == pytest.approx(0.25)
    assert [period["classification"] for period in periods] == [
        "Low",
        "Medium",
        "Unscored",
        "High",
        "High",
    ]
    assert periods[0]["startFrame"] == 0
    assert periods[0]["endFrame"] == 1
    assert periods[0]["durationSec"] == pytest.approx(0.2)
    assert periods[2]["startFrame"] == 3
    assert periods[2]["endFrame"] == 4
    assert periods[2]["durationSec"] == pytest.approx(0.2)
    assert periods[2]["band"] is None
    assert periods[2]["peakFrame"] == 3
    # A 1.5-second evidence gap splits otherwise identical High observations.
    assert periods[3]["endSec"] == pytest.approx(0.5)
    assert periods[4]["startSec"] == pytest.approx(2.0)


def test_periods_never_join_people_and_can_filter_unscored(result_session):
    session_id, _ = result_session

    all_people = frame_data.risk_periods_payload(session_id, "reba")
    unscored = frame_data.risk_periods_payload(
        session_id,
        "reba",
        band="unscored",
    )

    assert {period["person"] for period in all_people["periods"]} == {
        "person-a",
        "person-b",
    }
    assert unscored["count"] == 1
    assert unscored["periods"][0]["scored"] is False
    assert unscored["periods"][0]["representativeFrame"] == 3


def test_enriched_csv_contains_bilateral_calculations_but_no_raw_paths(result_session):
    session_id, _ = result_session

    payload = "".join(
        frame_data.iter_enriched_csv(
            session_id,
            person="person-a",
            method="reba",
            band="high",
        )
    )
    rows = list(csv.DictReader(StringIO(payload)))

    assert len(rows) == 2
    assert rows[0]["reba_classification"] == "High"
    assert rows[0]["rula_classification"] == "Change now"
    assert rows[0]["reba_selected_side"] == "left"
    assert rows[0]["reba_selected_trunk"] == "4"
    assert rows[0]["reba_left_upper_arm"] == "5"
    assert rows[0]["reba_right_score"] == "8"
    assert rows[0]["trunk_flexion_deg"] == "42.5"
    assert rows[0]["pose_mean_visibility"] == "0.91"
    assert rows[0]["task_conditions_true"] == "trunk_twisted"
    assert "input_source" not in rows[0]
    assert "camera.mp4" not in payload
    assert "pose_landmarks" not in payload


def test_csv_response_has_a_safe_download_name(result_session):
    session_id, _ = result_session

    response = frame_data.csv_export_response(
        session_id,
        filename=r"..\unsafe participant results",
    )

    assert response.media_type == "text/csv"
    assert response.headers["content-disposition"] == (
        'attachment; filename="unsafe_participant_results.csv"'
    )
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    ("page", "limit"),
    [(0, 10), (1, 0), (1, frame_data.MAX_PAGE_SIZE + 1)],
)
def test_invalid_pagination_is_rejected(result_session, page, limit):
    session_id, _ = result_session

    with pytest.raises(ValueError):
        frame_data.observations_payload(
            session_id,
            "reba",
            page=page,
            limit=limit,
        )


def test_export_filters_require_a_method(result_session):
    session_id, _ = result_session

    with pytest.raises(ValueError, match="method is required"):
        list(frame_data.iter_enriched_csv(session_id, band="high"))


def test_all_is_the_unfiltered_ui_sentinel(result_session):
    session_id, _ = result_session

    observations = frame_data.observations_payload(
        session_id,
        "reba",
        band="all",
        page=1,
        limit=20,
        person="person-a",
    )
    exported = "".join(
        frame_data.iter_enriched_csv(
            session_id,
            method="reba",
            band="all",
            person="person-a",
        )
    )

    assert observations["total"] == 7
    assert len(list(csv.DictReader(StringIO(exported)))) == 7
