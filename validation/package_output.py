"""Publish completed reviewer folders without replacing an existing output."""

import os
from pathlib import Path


def require_new_output(path: Path) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"{path} already exists. Choose a new output path.")


def publish_directory(staged: Path, destination: Path) -> None:
    require_new_output(destination)
    try:
        staged.rename(destination)
    except OSError:
        # A concurrent build may have published its complete directory first.
        require_new_output(destination)
        raise
