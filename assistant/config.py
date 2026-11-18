"""Assistant settings, read from ``config.yaml`` with environment overrides.

The endpoint is deliberately a URL rather than a library binding, so the serving
stack stays a configuration choice: llama.cpp's ``llama-server`` and vLLM speak
the OpenAI shape, and Ollama speaks both that and its own. Moving between them is
a config edit, not a code change, and nothing in this package imports a model
runtime.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import yaml

from assistant.prompts import PROMPT_VERSION
from paths import CONFIG_PATH

#: Frames pushed into the model's context for each kind of question. Video here
#: runs about 110 s across three cameras; feeding every sampled frame would cost
#: tens of thousands of visual tokens to answer a question about one moment.
DEFAULT_FRAME_BUDGET = {"scene": 24, "episode": 8, "moment": 2}
DEFAULT_MODEL = "gemma4:26b"


@dataclass(frozen=True, slots=True)
class AssistantConfig:
    enabled: bool = True
    endpoint: str = "http://127.0.0.1:11434/v1"
    chat_model: str = DEFAULT_MODEL
    scene_model: str = DEFAULT_MODEL
    api_key_env: str = "POSTURE_ASSISTANT_KEY"
    temperature: float = 0.1
    scene_temperature: float = 0.0
    seed: int = 20260804
    max_tool_iterations: int = 8
    request_timeout_sec: int = 600
    frame_width: int = 640
    #: Sent as ``options.num_ctx``. Ollama otherwise loads this model at its full
    #: 262144-token window, which needs 46 GB and spills onto the CPU on a 24 GB
    #: card -- roughly halving throughput to buy context no turn here uses. Set 0
    #: to send nothing, which is what a server that rejects unknown fields wants.
    context_tokens: int = 32768
    #: ``auto`` detects Ollama and uses its native API, which is the only way to
    #: apply ``context_tokens``; anything else speaks OpenAI. Pin to ``openai``
    #: to force the portable path.
    dialect: str = "auto"
    #: Independent readings of the recording for the scene card. Self-reported
    #: confidence is poorly calibrated; agreement between passes over different
    #: frames is measured. Three is the smallest number that can show a majority.
    scene_passes: int = 3
    frame_budget: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_FRAME_BUDGET))
    prompt_version: str = PROMPT_VERSION
    log_conversations: bool = True
    model_digest: str | None = None
    model_capabilities: tuple[str, ...] | None = None
    model_issue: str | None = None

    @property
    def api_key(self) -> str:
        # Ollama ignores it; a hosted endpoint would not. Never read a literal
        # key out of config.yaml -- that file is committed.
        return os.environ.get(self.api_key_env, "not-needed")

    def budget(self, kind: str) -> int:
        return int(self.frame_budget.get(kind, DEFAULT_FRAME_BUDGET.get(kind, 4)))

    def describe(self) -> dict[str, Any]:
        """The provenance stamp written into every conversation and scene card."""

        return {
            "endpoint": self.endpoint,
            "chatModel": self.chat_model,
            "sceneModel": self.scene_model,
            "modelDigest": self.model_digest,
            "temperature": self.temperature,
            "seed": self.seed,
            "promptVersion": self.prompt_version,
        }


def _coerce(raw: Any) -> AssistantConfig:
    if not isinstance(raw, dict):
        raw = {}
    budget = raw.get("frame_budget")
    merged = dict(DEFAULT_FRAME_BUDGET)
    if isinstance(budget, dict):
        for key, value in budget.items():
            try:
                merged[str(key)] = int(value)
            except (TypeError, ValueError):
                continue

    def pick(name: str, default: Any, cast: Any) -> Any:
        value = raw.get(name, default)
        try:
            return cast(value)
        except (TypeError, ValueError):
            return default

    return AssistantConfig(
        enabled=bool(raw.get("enabled", True)),
        endpoint=str(raw.get("endpoint") or "http://127.0.0.1:11434/v1").rstrip("/"),
        chat_model=str(raw.get("chat_model") or DEFAULT_MODEL),
        scene_model=str(raw.get("scene_model") or raw.get("chat_model") or DEFAULT_MODEL),
        api_key_env=str(raw.get("api_key_env") or "POSTURE_ASSISTANT_KEY"),
        temperature=pick("temperature", 0.1, float),
        scene_temperature=pick("scene_temperature", 0.0, float),
        seed=pick("seed", 20260804, int),
        max_tool_iterations=pick("max_tool_iterations", 8, int),
        request_timeout_sec=pick("request_timeout_sec", 600, int),
        frame_width=pick("frame_width", 640, int),
        context_tokens=pick("context_tokens", 32768, int),
        dialect=str(raw.get("dialect") or "auto").lower(),
        scene_passes=pick("scene_passes", 3, int),
        frame_budget=merged,
        prompt_version=str(raw.get("prompt_version") or PROMPT_VERSION),
        log_conversations=bool(raw.get("log_conversations", True)),
    )


@lru_cache(maxsize=1)
def load_config() -> AssistantConfig:
    """Read the ``assistant:`` block. A missing block yields working defaults.

    An unreadable or malformed config must not take the dashboard down with it;
    the assistant is an addition to the application, not a prerequisite for it.
    """

    raw: Any = {}
    try:
        with open(CONFIG_PATH, encoding="utf-8") as handle:
            document = yaml.safe_load(handle) or {}
        if isinstance(document, dict):
            raw = document.get("assistant") or {}
    except (OSError, UnicodeError, yaml.YAMLError):
        raw = {}

    config = _coerce(raw)
    override = os.environ.get("POSTURE_ASSISTANT_ENDPOINT")
    model = os.environ.get("POSTURE_ASSISTANT_MODEL")
    if override or model:
        config = AssistantConfig(
            enabled=config.enabled,
            endpoint=(override or config.endpoint).rstrip("/"),
            chat_model=model or config.chat_model,
            scene_model=model or config.scene_model,
            api_key_env=config.api_key_env,
            temperature=config.temperature,
            scene_temperature=config.scene_temperature,
            seed=config.seed,
            max_tool_iterations=config.max_tool_iterations,
            request_timeout_sec=config.request_timeout_sec,
            frame_width=config.frame_width,
            context_tokens=config.context_tokens,
            dialect=config.dialect,
            scene_passes=config.scene_passes,
            frame_budget=config.frame_budget,
            prompt_version=config.prompt_version,
            log_conversations=config.log_conversations,
        )
    return config


__all__ = ["AssistantConfig", "DEFAULT_FRAME_BUDGET", "load_config"]
