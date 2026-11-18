import inspect

from dashboard.professional_page import (
    RESULT_VIEWS,
    comparison_rows,
    filter_analysis_region,
    method_snapshot,
    render_professional_results,
    scope_records,
)


def test_public_renderer_has_explicit_injected_dependencies():
    parameters = inspect.signature(render_professional_results).parameters

    assert list(parameters) == [
        "context",
        "task",
        "catalog",
        "get_processed_fps",
        "detect_current_results",
        "load_bundle",
        "load_summary",
        "analyze_cached",
        "session",
    ]
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in parameters.values()
    )


def test_person_and_inclusive_frame_scope_is_real_not_display_only():
    records = [
        {"person_identifier": "p1", "frame_number": 1},
        {"person_identifier": "p1", "frame_number": 2},
        {"person_identifier": "p1", "frame_number": 3},
        {"person_identifier": "p2", "frame_number": 2},
        {"person_identifier": "p1", "frame_number": None},
    ]

    scoped = scope_records(records, person_identifier="p1", frame_range=(2, 3))

    assert [record["frame_number"] for record in scoped] == [2, 3]


def test_region_filter_narrows_every_measurement_stream_without_mutating_validity():
    quality = {"status": "valid_with_limitations", "selected_record_count": 5}
    readiness = [{"method": "RULA", "status": "ready"}]
    payload = {
        "data_quality": quality,
        "method_readiness": readiness,
        "angle_analyses": [
            {"measurement": "trunk_flexion"},
            {"measurement": "neck_flexion"},
        ],
        "repetition_candidates": [
            {"measurement": "trunk_flexion"},
            {"measurement": "neck_flexion"},
        ],
        "camera_sources": {
            "contributions": [
                {"measurement": "trunk_flexion"},
                {"measurement": "neck_flexion"},
            ],
            "agreement": [
                {"measurement": "trunk_flexion"},
                {"measurement": "neck_flexion"},
            ],
        },
        "recommendations": [
            {"title": "Quality", "evidence": []},
            {"title": "Trunk", "evidence": [{"measurement": "trunk_flexion"}]},
            {"title": "Neck", "evidence": [{"measurement": "neck_flexion"}]},
        ],
    }

    filtered = filter_analysis_region(payload, "trunk")

    assert [item["measurement"] for item in filtered["angle_analyses"]] == ["trunk_flexion"]
    assert [item["measurement"] for item in filtered["repetition_candidates"]] == ["trunk_flexion"]
    assert [item["measurement"] for item in filtered["camera_sources"]["contributions"]] == [
        "trunk_flexion"
    ]
    assert [item["title"] for item in filtered["recommendations"]] == ["Quality", "Trunk"]
    assert filtered["data_quality"] is quality
    assert filtered["method_readiness"] is readiness
    assert len(payload["angle_analyses"]) == 2


def test_method_snapshot_preserves_unavailable_and_selects_worst_valid_frame():
    unavailable = method_snapshot([{"frame_number": 1, "rula": {"selected": None}}], "RULA")
    records = [
        {
            "frame_number": 9,
            "timestamp_sec": 0.9,
            "reba": {
                "selected_side": "left",
                "selected": {"final_score": 8, "risk_level": "high"},
            },
        },
        {
            "frame_number": 4,
            "timestamp_sec": 0.4,
            "reba": {
                "selected_side": "right",
                "selected": {"final_score": 10, "action_category": "necessary soon"},
            },
        },
    ]

    snapshot = method_snapshot(records, "REBA")

    assert unavailable.maximum_score is None
    assert unavailable.valid_result_count == 0
    assert snapshot.maximum_score == 10
    assert snapshot.frame_number == 4
    assert snapshot.selected_side == "right"
    assert snapshot.interpretation == "necessary soon"


def test_compare_uses_summary_only_and_keeps_missing_methods_unavailable():
    calls = []
    catalog = [
        {
            "key": "study::001",
            "participant_id": "001",
            "study": {"folder_name": "internal-study"},
            "tasks": [{"folder_name": "trial-a"}],
        }
    ]

    def detect(study, participant, trial, sampling):
        calls.append(("detect", study, participant, trial, sampling))
        return {"dir": "result-directory"}

    def load_summary(result_dir):
        calls.append(("summary", result_dir))
        return {
            "processed_frames": 10,
            "pose_detected_frames": 8,
            "rula_summary": {"maximum_score": None, "observation_count": 0},
            "reba_summary": {"maximum_score": 7, "observation_count": 3},
            "rnle": {"lifting_index": None, "validity_status": "missing_required_task_input"},
        }

    rows, errors = comparison_rows(
        catalog=catalog,
        selected_context_keys=["study::001"],
        task={"folder_name": "trial-a"},
        sampling=None,
        detect_current_results=detect,
        load_summary=load_summary,
    )

    assert errors == []
    assert calls == [
        ("detect", "internal-study", "001", "trial-a", None),
        ("summary", "result-directory"),
    ]
    assert rows[0]["Participant"] == "001"
    assert rows[0]["RULA maximum"] == "Unavailable"
    assert rows[0]["REBA maximum"] == "7"
    assert rows[0]["RNLE LI (task-level)"] == "Unavailable"
    assert rows[0]["Pose-detected share"] == "80.0%"


def test_page_source_contains_all_professional_views_and_explicit_cycle_limitations():
    source = inspect.getsource(render_professional_results)
    module_source = inspect.getsource(inspect.getmodule(render_professional_results))

    for title in RESULT_VIEWS:
        assert title in module_source
    assert "Candidate movement cycles are not lift events" in module_source
    assert "not RNLE lifting frequency" in module_source
    assert "selection_analysis = dict(analysis)" in source
    assert "filter_analysis_region(selection_analysis, region)" in source


def test_the_default_workflow_reads_one_page_instead_of_choosing_a_tab():
    """Six equally weighted tabs of whole-trial aggregates hid the answer.

    Summary mode now renders the verdict, the timeline, and the body-part
    ranking in sequence, with the older evidence views demoted into expanders
    below rather than removed.
    """

    source = inspect.getsource(render_professional_results)

    verdict = source.index("_render_verdict_view")
    when = source.index("_render_when")
    where = source.index("_render_where")
    assert verdict < when < where

    for demoted in ("_render_why", "_render_exposure", "_render_movement", "_render_trust"):
        assert demoted in source
    assert "st.expander" in source


# ----------------------------------------------------------------------
# The question-ordered views
# ----------------------------------------------------------------------


def test_views_are_named_as_the_questions_an_assessor_asks():
    assert RESULT_VIEWS == (
        "The verdict",
        "When in the task",
        "Where in the body",
        "Why this score",
        "Can I trust it",
        "Report & compare",
    )


def test_a_stored_legacy_view_name_still_resolves():
    """A session opened before the rename must not land on nothing."""

    from dashboard.professional_page import _LEGACY_VIEW_NAMES, RESULT_VIEWS

    for old, new in _LEGACY_VIEW_NAMES.items():
        assert new in RESULT_VIEWS, old


def test_condition_labels_read_as_plain_english():
    from dashboard.professional_page import _condition_label

    assert _condition_label("trunk_twisted") == "Trunk twisted"
    assert _condition_label("left_upper_arm_abducted") == "Left upper arm abducted"
    assert _condition_label("some_unmapped_thing") == "Some unmapped thing"


def test_condition_values_keep_unknown_distinct_from_no():
    from dashboard.professional_page import _condition_value

    assert _condition_value(None) == "Unknown"
    assert _condition_value(False) == "No"
    assert _condition_value(True) == "Yes"
    assert _condition_value("static_or_repeated") == "static or repeated"
