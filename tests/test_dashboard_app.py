import json
from pathlib import Path
from shutil import copyfile

import pytest
from streamlit.testing.v1 import AppTest

from dashboard import data_loader


@pytest.fixture
def app(tmp_path, monkeypatch):
    studies = tmp_path / "studies"
    study = studies / "study_synthetic"
    participant = study / "participant_001"
    raw = participant / "reach" / "raw"
    raw.mkdir(parents=True)
    config = {"study_name": "Synthetic study", "tasks": [{"folder_name": "reach", "name": "Reach"}]}
    (study / "study_config.json").write_text(json.dumps(config), encoding="utf-8")
    (participant / "participant_info.json").write_text('{"participant_id":"001"}', encoding="utf-8")
    for name in ("cam_0.mp4", "cam_1.mp4"):
        copyfile("test_synthetic_video.mp4", raw / name)
    monkeypatch.setattr(data_loader, "RESULTS_DIR_STR", str(tmp_path / "results"))
    app_path = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
    dashboard = AppTest.from_file(str(app_path), default_timeout=120)
    dashboard.session_state["studies_root"] = str(studies)
    return dashboard.run()


def test_frame_review_is_a_core_section_not_an_advanced_one(app):
    """Every "open this evidence" button on Results lands in Frame Review.

    While it sat behind the Advanced toggle those buttons pointed at a section
    the default workflow could not reach, so the drill-through from a high score
    to the recorded frames silently did nothing.
    """

    assert not app.exception
    assert app.title[0].value == "Participant 001"
    assert app.segmented_control[0].label == "Dashboard section"
    assert app.segmented_control[0].options == [
        "Study Overview",
        "Run Analysis",
        "Results",
        "Frame Review",
        "3-D Activity",
    ]
    assert app.segmented_control[0].value == "Study Overview"
    visible_text = " ".join(element.value for element in [*app.title, *app.caption, *app.markdown])
    assert "Legacy values" not in visible_text


def test_advanced_mode_reveals_capture(app):
    """Capture is a choice, not a result, so it stays behind the toggle."""

    app.toggle(key="advanced_mode").set_value(True).run()

    assert not app.exception
    assert app.segmented_control[0].options == [
        "Study Overview",
        "Run Analysis",
        "Results",
        "Frame Review",
        "3-D Activity",
        "Record Activity",
    ]


def test_a_parked_section_request_is_applied_before_navigation_is_built(app):
    """A Results button parks the destination; the next run must honour it.

    Streamlit rejects writing a widget's own key after that widget exists, so the
    request cannot be applied at the click site. If the parked key were ever
    ignored, every drill-through would fail silently again.
    """

    app.session_state["pending_dashboard_section"] = "Frame Review"
    app.session_state["selected_frame"] = 47
    app.run()

    assert not app.exception
    assert app.segmented_control[0].value == "Frame Review"
    assert "pending_dashboard_section" not in app.session_state


def test_the_ergopartner_wordmark_is_shown(app):
    markup = " ".join(element.value for element in app.markdown)

    assert "ErgoPartner" in markup
    assert 'alt="ErgoPartner"' in markup
    assert "data:image/svg+xml;base64," in markup


def test_analysis_camera_setup_exposes_roles_only_for_multiple_views(app):
    app.segmented_control[0].set_value("Run Analysis").run()
    assert not app.exception
    assert app.radio(key="run_camera_mode").value == "Single camera"
    assert any(control.label == "Recording" for control in app.selectbox)
    assert not any(control.label.endswith(" role") for control in app.selectbox)
    app.radio(key="run_camera_mode").set_value("Multiple cameras").run()
    assert not app.exception
    assert any(control.label == "Recordings from the same take" for control in app.multiselect)
    assert any(control.label == "Camera 1 role" for control in app.selectbox)


def test_dashboard_has_responsive_tab_and_mobile_sidebar_contract():
    source = Path("dashboard/app.py").read_text(encoding="utf-8")
    assert 'initial_sidebar_state="auto"' in source
    assert "@media (max-width: 900px)" in source
    assert "@media (max-width: 600px)" in source
    assert '[data-testid="stSegmentedControl"] [role="radiogroup"]' in source
    assert "flex-wrap: wrap !important" in source
