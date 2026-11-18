"""The model client, in two dialects over one interface.

Written against a wire protocol rather than a runtime library so the serving
stack stays a configuration choice. Nothing here imports torch, transformers, or
a model loader, and ``requests`` -- already a pinned dependency -- is enough for
both streaming formats.

**openai** is the portable dialect: llama.cpp's ``llama-server``, vLLM, and
Ollama's ``/v1`` compatibility layer all speak it.

**ollama** is its native ``/api/chat``. It exists here for one measured reason.
Ollama's compatibility layer silently drops the ``options`` block, so the model
loads at its full declared 262144-token window: 46 GB, spilling onto the CPU on a
24 GB card. The native endpoint honours ``num_ctx``, and the same model then
loads in 22 GB entirely on the GPU. Same weights, same answers, about twice the
speed, and no turn here needs more than a fraction of that window.

The dialect is detected once from the endpoint and can be pinned in config.
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import requests

from assistant.config import AssistantConfig, load_config

LOGGER = logging.getLogger(__name__)


class BackendError(RuntimeError):
    """The model endpoint could not be reached or answered unusably."""


@dataclass(slots=True)
class StreamChunk:
    """One piece of a streamed completion."""

    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finished: bool = False


def image_message(caption: str, jpeg: bytes) -> dict[str, Any]:
    """A user message carrying one rendered frame.

    Tool results have no image channel in either protocol, so a tool that renders
    a frame returns a text receipt and the image arrives as the next user
    message. Written in the OpenAI shape and translated for the native dialect by
    :func:`_to_ollama_messages`.
    """

    encoded = base64.b64encode(jpeg).decode("ascii")
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": caption},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
        ],
    }


def _headers(config: AssistantConfig) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
    }


def _base_url(config: AssistantConfig) -> str:
    """The server root, with any ``/v1`` suffix removed."""

    endpoint = config.endpoint.rstrip("/")
    return endpoint[:-3].rstrip("/") if endpoint.endswith("/v1") else endpoint


@lru_cache(maxsize=8)
def _detect_dialect(endpoint: str) -> str:
    """``ollama`` when the server answers ``/api/tags``, else ``openai``."""

    root = endpoint.rstrip("/")
    root = root[:-3].rstrip("/") if root.endswith("/v1") else root
    try:
        response = requests.get(f"{root}/api/tags", timeout=3)
        if response.status_code == 200 and "models" in response.json():
            return "ollama"
    except (requests.RequestException, ValueError):
        pass
    return "openai"


def dialect(config: AssistantConfig) -> str:
    if config.dialect in ("openai", "ollama"):
        return config.dialect
    return _detect_dialect(config.endpoint)


def _to_ollama_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate OpenAI-shaped messages into the native schema.

    The two differences that matter: images ride in a sibling ``images`` array of
    bare base64 rather than inside the content parts, and a tool result is keyed
    by ``tool_name`` instead of ``tool_call_id``.
    """

    converted: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            text_parts: list[str] = []
            images: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    text_parts.append(str(part.get("text", "")))
                elif part.get("type") == "image_url":
                    url = str((part.get("image_url") or {}).get("url", ""))
                    if "base64," in url:
                        images.append(url.split("base64,", 1)[1])
            entry: dict[str, Any] = {
                "role": message.get("role", "user"),
                "content": "\n".join(text_parts),
            }
            if images:
                entry["images"] = images
            converted.append(entry)
            continue

        entry = {"role": message.get("role", "user"), "content": content or ""}
        if message.get("tool_calls"):
            entry["tool_calls"] = [
                {
                    "function": {
                        "name": call["function"]["name"],
                        # Native expects an object; the OpenAI shape carries a
                        # JSON string.
                        "arguments": _as_object(call["function"].get("arguments")),
                    }
                }
                for call in message["tool_calls"]
            ]
        if message.get("role") == "tool":
            entry["tool_name"] = message.get("name") or message.get("tool_call_id", "")
        converted.append(entry)
    return converted


def _as_object(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    try:
        parsed = json.loads(str(arguments or "{}"))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _options(config: AssistantConfig, temperature: float | None) -> dict[str, Any]:
    options: dict[str, Any] = {
        "temperature": config.temperature if temperature is None else temperature,
        "seed": config.seed,
    }
    if config.context_tokens > 0:
        options["num_ctx"] = int(config.context_tokens)
    return options


def health(config: AssistantConfig | None = None) -> dict[str, Any]:
    """Is the endpoint up, and is the configured model actually present?"""

    config = config or load_config()
    if not config.enabled:
        return {"ok": False, "state": "disabled", "detail": "The assistant is disabled in config.yaml."}
    try:
        response = requests.get(
            f"{config.endpoint}/models", headers=_headers(config), timeout=5
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        return {
            "ok": False,
            "state": "unreachable",
            "detail": f"No model server at {config.endpoint}. Start Ollama, or set "
            f"POSTURE_ASSISTANT_ENDPOINT. ({exc.__class__.__name__})",
            "endpoint": config.endpoint,
        }
    except ValueError:
        return {"ok": False, "state": "unreachable", "detail": "The endpoint returned a non-JSON reply."}

    names = {str(item["id"]) for item in payload.get("data", [])
             if isinstance(item, dict) and item.get("id")}
    wanted = config.chat_model
    required = {wanted, config.scene_model}
    # An explicitly tagged request must be routable. A sibling tag can have
    # entirely different weights and capabilities.
    missing = sorted(model for model in required if model not in names
                     and not (":" not in model and f"{model}:latest" in names
                              and dialect(config) == "ollama"))
    if missing:
        return {
            "ok": False,
            "state": "model_missing",
            "detail": f"Missing configured models at {config.endpoint}: {', '.join(missing)}. Available: "
            f"{', '.join(sorted(names)) or 'none'}.",
            "endpoint": config.endpoint,
            "model": wanted,
            "available": sorted(names),
            "missing": missing,
        }
    return {
        "ok": True,
        "state": "ready",
        "detail": f"{wanted} ready at {config.endpoint}.",
        "endpoint": config.endpoint,
        "model": wanted,
        "available": sorted(names),
        "dialect": dialect(config),
        "contextTokens": config.context_tokens,
    }


def _merge_tool_call(store: dict[int, dict[str, Any]], delta: dict[str, Any]) -> None:
    """Accumulate a streamed tool call, which arrives in fragments."""

    index = int(delta.get("index", 0))
    slot = store.setdefault(
        index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
    )
    if delta.get("id"):
        slot["id"] = delta["id"]
    function = delta.get("function") or {}
    if function.get("name"):
        slot["function"]["name"] = function["name"]
    if function.get("arguments"):
        slot["function"]["arguments"] += function["arguments"]


def stream_completion(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    config: AssistantConfig | None = None,
    model: str | None = None,
    temperature: float | None = None,
    think: bool = False,
) -> Iterator[StreamChunk]:
    """Stream one completion, yielding text deltas and then any tool calls.

    ``think`` is off by default. The reasoning trace roughly triples latency and
    the answer format here is already constrained; a reviewer waiting on a chart
    would rather have the answer.
    """

    config = config or load_config()
    from assistant.models import verify_installed

    verify_installed(config, model or config.chat_model)
    if tools and config.model_capabilities is not None and "tools" not in config.model_capabilities:
        raise BackendError(f"{config.chat_model} supports image review. Select a model with tool support for assessment chat.")
    native = dialect(config) == "ollama"

    if native:
        payload: dict[str, Any] = {
            "model": model or config.chat_model,
            "messages": _to_ollama_messages(messages),
            "stream": True,
            "options": _options(config, temperature),
        }
        if config.model_capabilities is None or "thinking" in config.model_capabilities:
            payload["think"] = bool(think)
        url = f"{_base_url(config)}/api/chat"
    else:
        payload = {
            "model": model or config.chat_model,
            "messages": messages,
            "stream": True,
            "temperature": config.temperature if temperature is None else temperature,
            "chat_template_kwargs": {"enable_thinking": bool(think)},
        }
        url = f"{config.endpoint}/chat/completions"
    if tools:
        payload["tools"] = tools

    try:
        response = requests.post(
            url,
            headers=_headers(config),
            data=json.dumps(payload),
            stream=True,
            allow_redirects=config.model_digest is None,
            timeout=(10, config.request_timeout_sec),
        )
    except requests.RequestException as exc:
        raise BackendError(f"Could not reach the model server at {url}: {exc}") from exc

    if response.status_code >= 400:
        detail = response.text[:400]
        response.close()
        raise BackendError(f"The model server returned {response.status_code}: {detail}")

    pending: dict[int, dict[str, Any]] = {}
    collected: list[dict[str, Any]] = []
    try:
        for raw in response.iter_lines():
            if not raw:
                continue
            # The native endpoint sends no charset, so requests declines to
            # decode and hands back bytes regardless of decode_unicode.
            line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
            if line.startswith("data:"):
                line = line[5:].strip()
            if not line or line == "[DONE]":
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.debug("Skipping unparsable stream line: %.120s", line)
                continue

            if native:
                if event.get("error"):
                    raise BackendError(f"The model server interrupted generation: {event['error']}")
                # Newline-delimited JSON, one whole message fragment per line.
                message = event.get("message") or {}
                text = message.get("content")
                if text:
                    yield StreamChunk(text=text)
                for index, call in enumerate(message.get("tool_calls") or []):
                    function = call.get("function") or {}
                    collected.append(
                        {
                            "id": call.get("id") or f"call_{len(collected)}_{index}",
                            "type": "function",
                            "function": {
                                "name": function.get("name", ""),
                                "arguments": json.dumps(function.get("arguments") or {}),
                            },
                        }
                    )
                if event.get("done"):
                    break
                continue

            choices = event.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            text = delta.get("content")
            if text:
                yield StreamChunk(text=text)
            for call in delta.get("tool_calls") or []:
                _merge_tool_call(pending, call)
            if choice.get("finish_reason"):
                break
    finally:
        response.close()

    calls = collected if native else [pending[key] for key in sorted(pending)]
    yield StreamChunk(tool_calls=calls, finished=True)


def complete(
    messages: list[dict[str, Any]],
    *,
    config: AssistantConfig | None = None,
    model: str | None = None,
    temperature: float | None = None,
    think: bool = False,
    max_tokens: int | None = None,
    schema: dict[str, Any] | None = None,
) -> str:
    """One non-streamed completion. Used by the scene analyst."""

    config = config or load_config()
    from assistant.models import verify_installed

    verify_installed(config, model or config.scene_model)
    native = dialect(config) == "ollama"

    if native:
        options = _options(config, temperature)
        if max_tokens:
            options["num_predict"] = int(max_tokens)
        payload: dict[str, Any] = {
            "model": model or config.chat_model,
            "messages": _to_ollama_messages(messages),
            "stream": False,
            "options": options,
        }
        if config.model_capabilities is None or "thinking" in config.model_capabilities:
            payload["think"] = bool(think)
        url = f"{_base_url(config)}/api/chat"
        if schema is not None:
            payload["format"] = schema
    else:
        payload = {
            "model": model or config.chat_model,
            "messages": messages,
            "stream": False,
            "temperature": config.temperature if temperature is None else temperature,
            "chat_template_kwargs": {"enable_thinking": bool(think)},
        }
        if max_tokens:
            payload["max_tokens"] = int(max_tokens)
        url = f"{config.endpoint}/chat/completions"
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "scene_reading", "strict": True, "schema": schema},
            }

    try:
        response = requests.post(
            url,
            headers=_headers(config),
            data=json.dumps(payload),
            allow_redirects=config.model_digest is None,
            timeout=(10, config.request_timeout_sec),
        )
        response.raise_for_status()
        document = response.json()
    except requests.Timeout as exc:
        raise BackendError("The model took too long to respond. Retry after it loads or choose a smaller local model.") from exc
    except requests.HTTPError as exc:
        detail = exc.response.text[:300] if exc.response is not None else str(exc)
        raise BackendError(f"The model could not complete this visual review. {detail}") from exc
    except requests.RequestException as exc:
        raise BackendError(f"Could not reach the model server: {exc}") from exc
    except ValueError as exc:
        raise BackendError("The model server returned a non-JSON reply.") from exc

    if native:
        message = document.get("message") or {}
        content = str(message.get("content") or "")
        if not content:
            # Qwen3-VL is a reasoning model and Ollama routes its reasoning to a
            # separate `thinking` field. `think: false` does not suppress it --
            # measured at ~10 600 characters either way -- so a token cap that
            # looks generous can be spent entirely before the answer starts.
            # Without this the failure surfaces as "did not return JSON", which
            # sends you looking at the schema instead of the budget.
            thinking = str(message.get("thinking") or "")
            if thinking:
                raise BackendError(
                    f"The model spent its whole token budget reasoning "
                    f"({document.get('eval_count')} tokens) and produced no answer. "
                    "Raise max_tokens for this call."
                )
        return content
    choices = document.get("choices") or []
    if not choices:
        raise BackendError("The model server returned no completion.")
    return str((choices[0].get("message") or {}).get("content") or "")


__all__ = [
    "BackendError",
    "StreamChunk",
    "complete",
    "health",
    "image_message",
    "stream_completion",
]
