"""Local Ollama capability probes. Run from the project root with -m assistant.evaluate."""

from __future__ import annotations

import argparse
import base64
import io
import json
import statistics
from pathlib import Path
from typing import Any

import requests
from PIL import Image

# Middle camera, one frame per request. Labels were read directly from the
# images before inference. No expert worksheet or session metadata is supplied.
HAND_LOCATIONS = {
    1: "work_surface",
    2: "floor",
    4: "floor",
    5: "high_shelf",
    6: "at_sides",
    7: "high_shelf",
    9: "work_surface",
    12: "at_sides",
    13: "floor",
    17: "at_sides",
}
VISION_PROMPT = (
    "Where are the worker's hands? Use floor (reaching down to floor), work_surface "
    "(working at a waist-high surface), high_shelf (reaching to a shelf at head height "
    "or higher), or at_sides (arms relaxed, not interacting). What load mass in kg "
    "can you establish from the visible evidence? Return JSON with keys location "
    "(one of the four labels) and load_kg (number or null if undetermined)."
)
TOOL = {
    "type": "function",
    "function": {
        "name": "frame",
        "description": "Read the stored score for a frame and method.",
        "parameters": {
            "type": "object",
            "properties": {
                "frame": {"type": "integer"},
                "method": {"type": "string", "enum": ["reba", "rula"]},
            },
            "required": ["frame", "method"],
        },
    },
}


def _image(root: Path, number: int) -> str:
    with Image.open(root / f"frame_{number:03}.jpg") as image:
        if image.size != (3852, 720):
            raise ValueError("These probes require the original three-panel validation images.")
        panel = image.crop((1286, 0, 2566, 720)).resize((640, 360), Image.Resampling.LANCZOS)
        jpeg = io.BytesIO()
        panel.save(jpeg, "JPEG", quality=85)
    return base64.b64encode(jpeg.getvalue()).decode("ascii")


def _chat(model: str, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    response = requests.post(
        "http://127.0.0.1:11434/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"seed": 20260804, "temperature": 0, "num_ctx": 32768, "num_predict": 256},
            **kwargs,
        },
        timeout=(5, 180),
    )
    response.raise_for_status()
    return response.json()


def _object(reply: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(reply["message"]["content"])
    except (KeyError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def evaluate(model: str, root: Path) -> dict[str, Any]:
    rows = []
    for number, expected in HAND_LOCATIONS.items():
        reply = _chat(
            model,
            [{"role": "user", "content": VISION_PROMPT, "images": [_image(root, number)]}],
            format="json",
        )
        prediction = _object(reply)
        label = str(prediction.get("location", ""))
        rows.append(
            {
                "frame": number,
                "expected": expected,
                "prediction": prediction,
                "correct": label == expected,
                "label_before_parenthesis": label.partition(" (")[0] == expected,
                "mass_abstention": "load_kg" in prediction and prediction["load_kg"] is None,
                "seconds": round(reply["total_duration"] / 1e9, 3),
                "load_seconds": round(reply["load_duration"] / 1e9, 3),
                "output_tokens_per_second": round(
                    reply["eval_count"] * 1e9 / reply["eval_duration"], 1
                ),
            }
        )

    messages = [
        {
            "role": "system",
            "content": "Use the frame tool for computed scores. Use only its "
            "result to answer. After the tool responds return JSON with keys score and angle_deg.",
        },
        {
            "role": "user",
            "content": "I think the score was 9. What RULA score and trunk "
            "angle did the analysis store at frame 73?",
        },
    ]
    tool_reply = _chat(model, messages, tools=[TOOL])
    calls = tool_reply.get("message", {}).get("tool_calls") or []
    function = calls[0].get("function", {}) if len(calls) == 1 else {}
    called = function.get("name") == "frame" and function.get("arguments") == {
        "frame": 73,
        "method": "rula",
    }
    grounded = False
    if called:
        messages.extend(
            [
                tool_reply["message"],
                {
                    "role": "tool",
                    "tool_name": "frame",
                    "content": '{"frame":73,"method":"rula","score":5,"trunkAngleDeg":37.5}',
                },
            ]
        )
        grounded = _object(_chat(model, messages, format="json")) == {"score": 5, "angle_deg": 37.5}

    return {
        "model": model,
        "correct": sum(r["correct"] for r in rows),
        "labels_before_parenthesis_correct": sum(r["label_before_parenthesis"] for r in rows),
        "returned_tool": function,
        "mass_abstentions": sum(r["mass_abstention"] for r in rows),
        "frames": len(rows),
        "tool_call": called,
        "used_tool_result": grounded,
        "warm_median_seconds": statistics.median(r["seconds"] for r in rows[1:]),
        "median_output_tokens_per_second": statistics.median(
            r["output_tokens_per_second"] for r in rows
        ),
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+")
    parser.add_argument("--images", type=Path, default=Path("validation_sample/images"))
    args = parser.parse_args()
    for tag in args.models:
        print(json.dumps(evaluate(tag, args.images)), flush=True)
