"""Regression guards for removed unsafe pre-v4 execution paths.

The legacy modules themselves (flat CSV exporter, unvalidated trajectory
filter, LLM reinterpreter, archived multipage dashboard) are deleted rather
than stubbed; Git history retains them. These tests keep the import paths dead
so a stale checkout or editor cache cannot silently resurrect one.
"""

import importlib

import pytest

from core.angle_calculator import BodyAngles
from pipeline.video_processor import ConfigurationError, VideoProcessor


@pytest.mark.parametrize("config", [{}, {"schema_version": 3}])
def test_video_processor_rejects_pre_v4_configuration(config):
    with pytest.raises(ConfigurationError, match="schema_version"):
        VideoProcessor(config)


@pytest.mark.parametrize(
    "module",
    ["core.temporal_filter", "export.csv_exporter", "analysis.llm_analyzer", "batch_fixup"],
)
def test_removed_legacy_modules_stay_removed(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_pipeline_refuses_to_mix_with_legacy_output(tmp_path):
    (tmp_path / "scores").mkdir()
    processor = VideoProcessor({"schema_version": 4})
    with pytest.raises(ConfigurationError, match="clean directory"):
        processor.process(["missing.mp4"], str(tmp_path))


def test_non_numeric_camera_tilt_is_rejected():
    processor = VideoProcessor({"schema_version": 4, "calibration": {"camera_tilt_deg": "unknown"}})
    with pytest.raises(ConfigurationError, match="camera_tilt_deg"):
        processor._score_angles(
            BodyAngles(),
            frame_number=0,
            timestamp_sec=0,
            person_identifier="person-0001",
            input_source="synthetic",
            calibration_status="not_calibrated",
            mean_visibility=1.0,
        )


def test_partial_rnle_applicability_review_is_rejected():
    processor = VideoProcessor(
        {"schema_version": 4, "rnle": {"applicability": {"one_handed": False}}}
    )
    with pytest.raises(ConfigurationError, match="omitted fields"):
        processor._compute_rnle("synthetic", "not_calibrated")
