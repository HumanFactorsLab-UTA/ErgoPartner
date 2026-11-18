"""Measure end-to-end pipeline throughput, and refuse to report a hollow result.

    python tools/benchmark_pipeline.py --video path\\to\\cam_0.mp4 --frames 200

The retained artifact in ``audit_runs/p001_task1_30frame_benchmark_20260716``
recorded 8.46 s for 30 frames while detecting zero poses and writing an empty
``frame_results.jsonl``: every frame had been discarded by the multi-camera
ambiguity gate. A throughput number measured over frames that were never
analysed is not a throughput number, so this tool reports pose yield alongside
timing and exits non-zero when nothing was actually assessed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import yaml  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", nargs="+", required=True, help="one path per camera")
    parser.add_argument("--frames", type=int, default=200, help="source frames to process")
    parser.add_argument("--start-frame", type=int, default=300, help="first source frame")
    parser.add_argument("--config", default=str(PROJECT / "config.yaml"))
    parser.add_argument(
        "--keep",
        help="write the run to this directory instead of a temporary one",
    )
    arguments = parser.parse_args()

    for path in arguments.video:
        if not Path(path).is_file():
            print(f"Video not found: {path}", file=sys.stderr)
            return 2

    from pipeline.video_processor import VideoProcessor

    config = yaml.safe_load(Path(arguments.config).read_text(encoding="utf-8"))
    config["video"]["start_frame"] = arguments.start_frame
    config["video"]["end_frame"] = arguments.start_frame + arguments.frames - 1

    output = Path(arguments.keep) if arguments.keep else Path(tempfile.mkdtemp(prefix="benchmark_"))
    shutil.rmtree(output, ignore_errors=True)
    try:
        started = time.perf_counter()
        VideoProcessor(config).process([str(path) for path in arguments.video], str(output))
        elapsed = time.perf_counter() - started
        summary = json.loads((output / "video_summary.json").read_text(encoding="utf-8"))

        processed = summary.get("processed_frames") or 0
        detected = summary.get("pose_detected_frames") or 0
        cameras = len(arguments.video)
        yield_share = (detected / processed) if processed else 0.0

        print(f"\ncameras                 : {cameras}")
        print(f"source frames processed : {processed:,}")
        print(f"frames with a pose      : {detected:,} ({yield_share:.0%})")
        print(f"wall time               : {elapsed:.2f} s")
        if processed:
            print(f"per processed frame     : {elapsed / processed * 1000:.1f} ms")
            print(f"per camera-frame        : {elapsed / processed / cameras * 1000:.1f} ms")
            print(f"throughput              : {processed / elapsed:.2f} frames/s")

        completion = summary.get("method_completion") or {}
        for entry in completion.values():
            if isinstance(entry, dict):
                scored = entry.get("scored_observations") or 0
                total = entry.get("total_observations") or 0
                blocker = entry.get("primary_blocker")
                suffix = f", first blocker {blocker}" if blocker else ""
                print(f"{entry.get('method', '?'):<24}: scored {scored:,}/{total:,}{suffix}")

        if detected == 0:
            print(
                "\nNo pose was detected on any processed frame. This timing measures "
                "decoding and rejected frames only and must not be quoted as pipeline "
                "throughput. Check the multi-camera ambiguity gate and the chosen "
                "frame range.",
                file=sys.stderr,
            )
            return 1
        if yield_share < 0.5:
            print(
                f"\nOnly {yield_share:.0%} of processed frames yielded a pose; the timing "
                "is dominated by rejected frames and understates real analysis cost.",
                file=sys.stderr,
            )
        return 0
    finally:
        if not arguments.keep:
            shutil.rmtree(output, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
