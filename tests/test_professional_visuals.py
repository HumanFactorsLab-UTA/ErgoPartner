import json

import pytest

from dashboard.professional_visuals import (
    CSV_COLUMNS,
    RED,
    angle_summary_figure,
    camera_evidence_figure,
    data_quality_coverage_figure,
    method_readiness_figure,
    posture_band_exposure_figure,
    professional_csv_rows,
    professional_markdown_report,
)


def evidence(measurement="trunk_flexion", value=40.0, frame=9, timestamp=0.9):
    return {
        "frame_number": frame,
        "timestamp_sec": timestamp,
        "person_identifier": "person-0001",
        "measurement": measurement,
        "value_deg": value,
        "camera_source": "side",
        "detail": "Observed evidence",
    }


def provenance(method, url):
    return {
        "identifier": method,
        "title": f"{method} authoritative source",
        "version": "verified edition",
        "url": url,
        "scope": f"{method} method evidence",
    }


def time_coverage():
    return {
        "timestamp_count": 10,
        "span_sec": 0.9,
        "covered_duration_sec": 0.9,
        "gap_count": 0,
        "gap_duration_sec": 0.0,
        "longest_gap_sec": None,
        "coverage_percent": 100.0,
        "maximum_gap_sec": 0.25,
    }


def analysis_payload():
    rula_source = provenance("RULA", "https://pubmed.ncbi.nlm.nih.gov/15676903/")
    reba_source = provenance("REBA", "https://pubmed.ncbi.nlm.nih.gov/10711982/")
    rnle_source = provenance("RNLE", "https://www.cdc.gov/niosh/publications/numbered/94-110.html")
    observation_source = provenance(
        "NIOSH-POSTURE-OBSERVATION", "https://stacks.cdc.gov/view/cdc/24085"
    )
    return {
        "schema_version": "professional-analysis/1.0",
        "data_quality": {
            "status": "valid_with_limitations",
            "input_record_count": 10,
            "selected_record_count": 10,
            "usable_angle_observation_count": 10,
            "invalid_timestamp_count": 0,
            "duplicate_timestamp_count": 0,
            "out_of_order_timestamp_count": 0,
            "time_coverage": time_coverage(),
            "mean_pose_visibility": 0.94,
            "minimum_pose_visibility": 0.82,
            "pose_visibility_below_threshold_count": 0,
            "calibration_statuses": ["calibrated"],
            "camera_view_statuses": ["suitable"],
            "model_coordinate_types": ["calibrated_multiview_3d"],
            "timestamp_sources": ["decoder_pts"],
            "camera_count": 3,
            "warnings": ["Agreement remains descriptive."],
        },
        "angle_analyses": [
            {
                "measurement": "trunk_flexion",
                "units": "degrees",
                "status": "valid",
                "statistics": {
                    "valid_observation_count": 10,
                    "excluded_observation_count": 0,
                    "minimum_deg": 0.0,
                    "maximum_deg": 40.0,
                    "mean_deg": 18.0,
                    "median_deg": 20.0,
                    "standard_deviation_deg": 12.0,
                    "percentile_05_deg": 2.0,
                    "percentile_95_deg": 38.0,
                    "absolute_peak": evidence(),
                },
                "time_coverage": time_coverage(),
                "exclusion_reasons": [],
                "band_exposure": [
                    {
                        "method": "RULA",
                        "measurement": "trunk_flexion",
                        "band_id": "flexion_gt_20_to_60",
                        "label": "Flexion >20° to 60°",
                        "is_non_neutral": True,
                        "observation_count": 4,
                        "duration_sec": 0.4,
                        "percent_of_covered_duration": 44.444,
                        "evidence": evidence(),
                        "provenance": rula_source,
                    },
                    {
                        "method": "REBA",
                        "measurement": "trunk_flexion",
                        "band_id": "flexion_gt_20_to_60",
                        "label": "Flexion >20° to 60°",
                        "is_non_neutral": True,
                        "observation_count": 4,
                        "duration_sec": 0.4,
                        "percent_of_covered_duration": 44.444,
                        "evidence": evidence(),
                        "provenance": reba_source,
                    },
                ],
                "sustained_bouts": [],
            },
            {
                "measurement": "left_wrist_flexion",
                "units": "degrees",
                "status": "unavailable",
                "statistics": {
                    "valid_observation_count": 0,
                    "excluded_observation_count": 10,
                    "minimum_deg": None,
                    "maximum_deg": None,
                    "mean_deg": None,
                    "median_deg": None,
                    "standard_deviation_deg": None,
                    "percentile_05_deg": None,
                    "percentile_95_deg": None,
                    "absolute_peak": None,
                },
                "time_coverage": {
                    "timestamp_count": 0,
                    "span_sec": None,
                    "covered_duration_sec": None,
                    "gap_count": None,
                    "gap_duration_sec": None,
                    "longest_gap_sec": None,
                    "coverage_percent": None,
                    "maximum_gap_sec": 0.25,
                },
                "exclusion_reasons": ["unsupported_wrist_source:pose_only (10)"],
                "band_exposure": [],
                "sustained_bouts": [],
            },
        ],
        "repetition_candidates": [
            {
                "measurement": "trunk_flexion",
                "status": "candidate_events_available",
                "candidate_only": True,
                "algorithm": "scipy.signal.find_peaks",
                "candidate_count": 2,
                "rate_per_minute": 12.0,
                "analysis_duration_sec": 10.0,
                "segments_analyzed": 1,
                "prominence_deg": 10.0,
                "minimum_interval_sec": 0.5,
                "event_evidence": [evidence()],
                "limitations": ["Task intent requires confirmation."],
                "provenance": observation_source,
            }
        ],
        "method_readiness": [
            {
                "method": "RULA",
                "status": "ready",
                "assessed_record_count": 10,
                "valid_result_count": 10,
                "missing_inputs": [],
                "invalid_inputs": [],
                "unsupported_conditions": [],
                "observed_validity_statuses": ["valid"],
                "detail": "Independent RULA results are available.",
                "provenance": rula_source,
            },
            {
                "method": "REBA",
                "status": "missing_required_inputs",
                "assessed_record_count": 10,
                "valid_result_count": 0,
                "missing_inputs": ["coupling"],
                "invalid_inputs": [],
                "unsupported_conditions": [],
                "observed_validity_statuses": ["missing_required_task_input"],
                "detail": "Coupling must be supplied.",
                "provenance": reba_source,
            },
            {
                "method": "RNLE",
                "status": "not_evaluated",
                "assessed_record_count": None,
                "valid_result_count": None,
                "missing_inputs": [],
                "invalid_inputs": [],
                "unsupported_conditions": [],
                "observed_validity_statuses": [],
                "detail": "No RNLE result supplied.",
                "provenance": rnle_source,
            },
        ],
        "camera_sources": {
            "contribution_status": "available",
            "contributions": [
                {
                    "measurement": "trunk_flexion",
                    "camera_source": "front",
                    "observation_count": 6,
                    "percent_of_valid_observations": 60.0,
                },
                {
                    "measurement": "trunk_flexion",
                    "camera_source": "side",
                    "observation_count": 4,
                    "percent_of_valid_observations": 40.0,
                },
            ],
            "agreement_status": "available",
            "agreement": [
                {
                    "measurement": "trunk_flexion",
                    "paired_observation_count": 10,
                    "mean_range_deg": 4.0,
                    "median_range_deg": 3.5,
                    "percentile_95_range_deg": 8.0,
                    "maximum_range_deg": 10.0,
                    "maximum_range_evidence": evidence(value=10.0),
                    "interpretation": "Descriptive range; no universal threshold applied.",
                }
            ],
        },
        "recommendations": [
            {
                "recommendation_id": "review-trunk",
                "priority": "review",
                "title": "Review task reach",
                "rationale": "Evaluate work height and horizontal reach.",
                "observed_fact": "Trunk flexion entered a non-neutral band.",
                "evidence": [evidence()],
                "related_methods": ["RULA", "REBA"],
                "source_urls": [rula_source["url"], reba_source["url"]],
            }
        ],
        "warnings": [
            "Official posture bands are descriptive; no new score is synthesized.",
            "Pose-model confidence is separate from ergonomic validity.",
        ],
        "provenance": [rula_source, reba_source, rnle_source, observation_source],
    }


@pytest.mark.parametrize(
    "factory",
    [
        method_readiness_figure,
        angle_summary_figure,
        posture_band_exposure_figure,
        data_quality_coverage_figure,
        camera_evidence_figure,
    ],
)
def test_empty_inputs_show_explicit_not_available_annotation(factory):
    figure = factory({})

    assert figure.layout.annotations
    assert "Not available" in str(figure.layout.annotations[0].text)
    # Figures are transparent so they sit on the app surface rather than in a
    # box of a slightly different shade. The page supplies the background.
    assert figure.layout.paper_bgcolor == "rgba(0,0,0,0)"
    json.loads(figure.to_json())


def test_method_readiness_table_preserves_unavailable_and_separate_methods():
    figure = method_readiness_figure(analysis_payload())
    table = figure.data[0]
    rendered = str(table.cells.values)

    assert table.type == "table"
    assert "RULA" in rendered
    assert "REBA" in rendered
    assert "RNLE" in rendered
    assert "Not available" in rendered
    assert "None reported" in rendered


def test_angle_summary_uses_only_validated_angles_and_shows_percentiles_and_rom():
    figure = angle_summary_figure(analysis_payload())

    assert [trace.name for trace in figure.data] == [
        "Observed range (min–max)",
        "5th–95th percentile",
        "Median",
    ]
    assert list(figure.data[0].y) == ["Trunk Flexion"]
    assert list(figure.data[0].x) == [40.0]
    assert list(figure.data[0].base) == [0.0]
    assert list(figure.data[1].x) == [36.0]
    assert "Wrist" not in str(figure.to_dict())


def test_posture_band_figure_keeps_rula_and_reba_separate_and_uses_time_exposure():
    figure = posture_band_exposure_figure(analysis_payload())
    trace = figure.data[0]

    assert list(trace.x) == pytest.approx([44.444, 44.444])
    assert {row[0] for row in trace.customdata} == {"RULA", "REBA"}
    assert all(color == RED for color in trace.marker.color)
    assert "Share of covered duration" in figure.layout.xaxis.title.text
    assert "score" in figure.layout.title.text.lower()


def test_posture_band_figure_falls_back_to_observation_count_not_zero_duration():
    payload = analysis_payload()
    for band in payload["angle_analyses"][0]["band_exposure"]:
        band["duration_sec"] = None
        band["percent_of_covered_duration"] = None

    figure = posture_band_exposure_figure(payload)

    assert list(figure.data[0].x) == [4, 4]
    assert figure.layout.xaxis.title.text == "Observed frame count"


def test_data_quality_figure_displays_coverage_without_inventing_missing_gap_duration():
    payload = analysis_payload()
    payload["data_quality"]["time_coverage"]["gap_duration_sec"] = None
    figure = data_quality_coverage_figure(payload)

    assert figure.data[0].type == "table"
    bars = [trace for trace in figure.data if trace.type == "bar"]
    assert len(bars) == 1
    assert bars[0].name == "Covered duration"
    assert list(bars[0].x) == [0.9]


def test_data_quality_zero_selected_records_is_explicitly_empty():
    payload = analysis_payload()
    payload["data_quality"]["selected_record_count"] = 0

    figure = data_quality_coverage_figure(payload)

    assert not figure.data
    assert "No records matched" in figure.layout.annotations[0].text


def test_camera_figure_shows_contribution_and_descriptive_agreement_without_threshold():
    figure = camera_evidence_figure(analysis_payload())

    names = [trace.name for trace in figure.data]
    assert names == ["Front", "Side", "95th-percentile range", "Maximum range"]
    assert list(figure.data[0].x) == [60.0]
    assert list(figure.data[2].x) == [8.0]
    assert "range" in figure.layout.xaxis2.title.text.lower()


def test_missing_camera_agreement_stays_unavailable_not_zero():
    payload = analysis_payload()
    payload["camera_sources"]["agreement"] = None
    payload["camera_sources"]["agreement_status"] = "unavailable"

    figure = camera_evidence_figure(payload)

    assert [trace.name for trace in figure.data] == ["Front", "Side"]
    assert any(
        "not available" in annotation.text.lower() for annotation in figure.layout.annotations
    )
    assert "0°" not in str(figure.layout.annotations)


def test_markdown_report_is_deterministic_sourced_and_explicit_about_limitations():
    payload = analysis_payload()
    first = professional_markdown_report(payload)
    second = professional_markdown_report(payload)

    assert first == second
    assert "# Professional Ergonomics Analysis Evidence Report" in first
    assert "RULA" in first and "REBA" in first and "RNLE" in first
    assert "Not available" in first
    assert "not a synthetic overall score" in first
    assert "https://pubmed.ncbi.nlm.nih.gov/15676903/" in first
    assert "Pose-model confidence is separate" in first
    assert "Generated" not in first


def test_csv_rows_have_stable_columns_preserve_none_and_contain_no_overall_score():
    rows = professional_csv_rows(analysis_payload())

    assert rows
    assert all(tuple(row) == CSV_COLUMNS for row in rows)
    assert any(
        row["section"] == "method_readiness"
        and row["method"] == "RNLE"
        and row["metric"] == "valid_result_count"
        and row["value"] is None
        for row in rows
    )
    assert {row["method"] for row in rows if row["section"] == "method_readiness"} == {
        "RULA",
        "REBA",
        "RNLE",
    }
    assert not any(row["metric"] == "overall_score" for row in rows)
    json.dumps(rows)


def test_mapping_or_to_dict_object_is_accepted_without_mutation():
    payload = analysis_payload()
    before = json.dumps(payload, sort_keys=True)

    class Result:
        def to_dict(self):
            return payload

    method_readiness_figure(Result())
    professional_markdown_report(Result())

    assert json.dumps(payload, sort_keys=True) == before


def test_invalid_analysis_type_raises_clear_error():
    with pytest.raises(TypeError, match="mapping"):
        method_readiness_figure(object())
