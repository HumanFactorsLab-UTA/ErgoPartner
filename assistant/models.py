"""Discover installed Ollama models and resolve a model for one assessment."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlsplit

import requests

from assistant.config import AssistantConfig, load_config
from dashboard.cache import ttl_cache


def is_local(endpoint: str) -> bool:
    return urlsplit(endpoint).hostname in {"127.0.0.1", "localhost", "::1"}


def _root(config: AssistantConfig) -> str:
    return config.endpoint.rstrip("/").removesuffix("/v1")


def _tags(root: str) -> list[dict]:
    response = requests.get(f"{root}/api/tags", timeout=2, allow_redirects=False)
    response.raise_for_status()
    tags = response.json()["models"]
    if not isinstance(tags, list):
        raise ValueError("Invalid installed model list")
    return [item for item in tags if isinstance(item, dict)]


@ttl_cache(ttl=30)
def _catalog(root: str) -> dict:
    try:
        tags = _tags(root)
    except (requests.RequestException, ValueError, KeyError, TypeError):
        return {
            "state": "unreachable",
            "detail": "Start Ollama on this computer and refresh the list.",
            "models": [],
        }
    options = []
    for item in tags:
        if not isinstance(item, dict) or not (tag := item.get("name")) or not isinstance(tag, str):
            continue
        if (
            item.get("remote_model")
            or item.get("remote_host")
            or tag.endswith(("-cloud", ":cloud"))
        ):
            continue
        try:
            details = _details(root, tag, item.get("digest") or "")
        except (requests.RequestException, ValueError, TypeError):
            continue
        if details.get("remote_model") or details.get("remote_host"):
            continue
        capabilities = details.get("capabilities") or []
        if item.get("digest") and "vision" in capabilities and "completion" in capabilities:
            options.append(
                {
                    "tag": tag,
                    "digest": item.get("digest"),
                    "size": item.get("size"),
                    "capabilities": capabilities,
                }
            )
    options.sort(key=lambda option: (option["size"] or 0, option["tag"]))
    return {
        "state": "ready",
        "models": options,
        "detail": "" if options else "No installed model reports local vision support.",
    }


@ttl_cache(ttl=120)
def _details(root: str, tag: str, digest: str) -> dict:
    response = requests.post(
        f"{root}/api/show", json={"model": tag}, timeout=3, allow_redirects=False
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Invalid model metadata")
    return {key: data.get(key) for key in ("capabilities", "remote_model", "remote_host")}


def inventory(config: AssistantConfig, *, refresh: bool = False) -> dict:
    if not is_local(config.endpoint) or config.dialect == "openai":
        return {
            "state": "unavailable",
            "detail": "Local model discovery requires a local Ollama endpoint.",
            "models": [],
        }
    if refresh:
        _details.cache_clear()  # type: ignore[attr-defined]  # supplied by ttl_cache
        _catalog.cache_clear()  # type: ignore[attr-defined]
    return _catalog(_root(config))


def _session_path(session_id: str) -> Path:
    from api.sessions import resolve_result_dir

    return Path(resolve_result_dir(session_id)) / "assistant_model.json"


def session_selection(session_id: str) -> dict:
    path = _session_path(session_id)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _find(options: list[dict], tag: str) -> dict | None:
    return next(
        (
            entry
            for entry in options
            if entry["tag"] == tag
            or (":" not in tag.rsplit("/", 1)[-1] and entry["tag"] == f"{tag}:latest")
        ),
        None,
    )


def effective_config(
    session_id: str | None = None, *, requested_model: str | None = None
) -> AssistantConfig:
    from api.assessments import preferences
    from api.sessions import load_run_config

    config = load_config()
    chosen = requested_model
    saved_digest = None
    if session_id and not chosen:
        selection = session_selection(session_id)
        workflow = load_run_config(session_id).get("assessment_workflow") or {}
        chosen = selection.get("model") or workflow.get("vision_model")
        saved_digest = (
            selection.get("digest")
            if selection.get("model")
            else workflow.get("vision_model_digest")
        )
    chosen = chosen or preferences().visionModel
    if not chosen and (config.dialect == "openai" or not is_local(config.endpoint)):
        return config
    catalog = inventory(config)
    # An existing explicitly configured non-Ollama service still uses its own model roles.
    if (
        not chosen
        and config.dialect == "auto"
        and urlsplit(config.endpoint).port != 11434
        and catalog["state"] != "ready"
    ):
        return config
    tag = chosen or config.scene_model
    model = _find(catalog["models"], tag)
    issue = (
        None
        if model
        else (catalog["detail"] or f"{tag} is unavailable. Select an installed vision model.")
    )
    return replace(
        config,
        chat_model=(model["tag"] if model else tag) if chosen else config.chat_model,
        scene_model=model["tag"] if model else tag,
        model_digest=model["digest"] if model else saved_digest,
        model_capabilities=tuple(model["capabilities"]) if model else None,
        dialect="ollama" if catalog["state"] == "ready" else config.dialect,
        model_issue=issue,
    )


def require_vision(config: AssistantConfig) -> None:
    from assistant.backend import BackendError

    if not config.enabled:
        raise BackendError("The assessment assistant is disabled.")
    if config.model_issue:
        raise BackendError(config.model_issue)


def verify_installed(config: AssistantConfig, tag: str) -> None:
    """Recheck the manifest before transmitting images to a selected local tag."""
    from assistant.backend import BackendError

    require_vision(config)
    if not config.model_digest or tag != config.scene_model:
        return
    try:
        current = next((item for item in _tags(_root(config)) if item.get("name") == tag), {})
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        raise BackendError("Ollama is unavailable. Start it on this computer and retry.") from exc
    if (
        current.get("digest") != config.model_digest
        or current.get("remote_model")
        or current.get("remote_host")
    ):
        _catalog.cache_clear()  # type: ignore[attr-defined]
        raise BackendError(
            f"The installed model {tag} changed or was removed. Refresh the model list and retry."
        )


def validate_choice(tag: str | None) -> None:
    """Validate a freshly chosen tag, leaving offline edits to unrelated fields possible."""
    if tag:
        from fastapi import HTTPException

        catalog = inventory(load_config())
        if not _find(catalog["models"], tag):
            raise HTTPException(
                422, catalog["detail"] or "Choose a locally installed vision model."
            )


def save_selection(session_id: str, tag: str | None) -> None:
    from api.assessments import atomic_json

    path = _session_path(session_id)
    value = {}
    if tag:
        config = effective_config(requested_model=tag)
        require_vision(config)
        value = {"model": config.scene_model, "digest": config.model_digest}
    atomic_json(path, value)
