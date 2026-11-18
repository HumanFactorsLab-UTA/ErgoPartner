"""Conversational assistant layer over a completed ergonomic assessment.

Nothing in this package computes an ergonomic score. It reads what the
deterministic pipeline already wrote, shows the model the frames the pipeline
already rendered, and keeps the two apart in every answer it produces.

The scoring packages -- ``core``, ``ergonomics``, ``pipeline``, ``temporal`` --
must never import this one. ``tests/test_assistant_isolation.py`` enforces that.
"""

from __future__ import annotations

__all__ = ["SCHEMA_VERSION"]

#: Bumped whenever the wire shape of a conversation or scene card changes.
SCHEMA_VERSION = "assistant/1.0"
