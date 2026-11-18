"""
Paths - Centralized project path management.

All modules should import from here instead of computing PROJECT_ROOT
repeatedly using os.path.dirname(os.path.abspath(__file__)).
"""

import base64
import json
import logging
import os
import tempfile
from pathlib import Path


def resolve_studies_root(project: Path) -> Path:
    """Honor an explicit location and recognize both supported study layouts."""
    settings = project / ".posture-settings.json"
    if settings.is_file():
        try:
            saved = json.loads(settings.read_text(encoding="utf-8"))
            if isinstance(saved, dict) and isinstance(saved.get("studies_root"), str):
                selected = Path(saved["studies_root"])
                if selected.is_absolute():
                    return selected
        except (OSError, ValueError):
            logging.getLogger(__name__).warning(
                "Could not read saved studies folder from %s", settings
            )
    configured = os.environ.get("POSTURE_STUDIES_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    sibling = project.parent / "studies"
    if sibling.is_dir():
        return sibling
    ancestor = project.parent.parent / "studies"
    if any(manifest.is_file() for manifest in ancestor.glob("study_*/study_config.json")):
        return ancestor
    return sibling


def save_studies_root(root: Path, project: Path | None = None) -> None:
    """Persist a local folder preference atomically without modifying study data."""
    settings = (project if project is not None else PROJECT_ROOT) / ".posture-settings.json"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=settings.parent,
        prefix=".posture-settings-",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump({"studies_root": str(root)}, handle, indent=2)
            handle.write("\n")
        except BaseException:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(settings)
    finally:
        temporary.unlink(missing_ok=True)


# Project root (parent of this file's directory)
PROJECT_ROOT = Path(__file__).resolve().parent
BRAND_DIR = PROJECT_ROOT / "web" / "public" / "brand"


def brand_logo_uri(*, dark_surface: bool = False) -> str:
    """Embed the shipped vector wordmark in offline and printable surfaces."""
    name = "ergopartner-horizontal-dark.svg" if dark_surface else "ergopartner-horizontal.svg"
    encoded = base64.b64encode((BRAND_DIR / name).read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


# Key directories
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "output"
STUDIES_DIR = resolve_studies_root(PROJECT_ROOT)

# Configuration files
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
CONFIG_TEST_30FPS_PATH = PROJECT_ROOT / "config_test_30fps.yaml"

# Legacy string paths for compatibility with code expecting str
PROJECT_ROOT_STR = str(PROJECT_ROOT)
MODELS_DIR_STR = str(MODELS_DIR)
RESULTS_DIR_STR = str(RESULTS_DIR)
OUTPUT_DIR_STR = str(OUTPUT_DIR)
STUDIES_DIR_STR = str(STUDIES_DIR)
