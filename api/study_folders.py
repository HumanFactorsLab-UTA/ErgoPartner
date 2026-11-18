"""Local folder navigation for the study library. No file contents are returned."""

from __future__ import annotations

import ipaddress
import os
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request


def require_local_request(request: Request) -> None:
    """Keep local disk navigation and configuration off remote web pages."""
    try:
        local_client = bool(
            request.client and ipaddress.ip_address(request.client.host).is_loopback
        )
    except ValueError:
        local_client = False
    allowed_hosts = {"localhost", "127.0.0.1", "::1"}
    origin = request.headers.get("origin")
    allowed_origins = {
        str(request.base_url).rstrip("/"),
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    }
    if (
        not local_client
        or request.url.hostname not in allowed_hosts
        or (origin is not None and origin not in allowed_origins)
        or request.headers.get("sec-fetch-site") == "cross-site"
    ):
        raise HTTPException(403, "Folder selection is available from the local dashboard only.")


def directory_path(value: str) -> Path:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1]
    path = Path(value).expanduser()
    if not value or not path.is_absolute():
        raise HTTPException(422, "Enter a full folder path.")
    try:
        path = path.resolve(strict=True)
        if not path.is_dir():
            raise HTTPException(422, "Choose a folder, rather than a file.")
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(
            422, "This folder could not be opened. Check its path and access permissions."
        ) from exc
    return path


def folder_listing(value: str) -> dict[str, Any]:
    path = directory_path(value)
    folders = []
    try:
        for child in path.iterdir():
            try:
                if child.is_dir():
                    folders.append(
                        {
                            "name": child.name,
                            "path": str(child),
                            "isStudy": (child / "study_config.json").is_file(),
                        }
                    )
            except OSError:
                continue
    except OSError as exc:
        raise HTTPException(422, "This folder is not readable. Choose another folder.") from exc
    folders.sort(key=lambda item: item["name"].casefold())
    return {
        "path": str(path),
        "parent": str(path.parent) if path.parent != path else None,
        "folders": folders,
        "studyCount": sum(item["isStudy"] for item in folders),
        "drives": os.listdrives() if os.name == "nt" else ["/"],
    }


def validate_studies_folder(value: str) -> Path:
    listing = folder_listing(value)
    if not listing["studyCount"]:
        raise HTTPException(
            422,
            "No studies found. Choose the parent folder containing study folders with study_config.json.",
        )
    return Path(listing["path"])
