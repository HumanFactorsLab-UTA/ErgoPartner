"""
Ergonomic Assessment System v4 - CLI Entry Point.

Usage:
    python main.py process --video front.mp4
    python main.py process --video front.mp4 side.mp4 iso.mp4
    python main.py process --video front.mp4 --output results/
    python main.py process --video "d:/path/to/video.mp4" --config config.yaml

Features:
    - REBA (Rapid Entire Body Assessment) continuous scoring
    - RULA (Rapid Upper Limb Assessment) continuous scoring
    - RNLE with explicit completeness and applicability validation
    - Multi-view per-angle best-view fusion (when multiple cameras)
    - Timestamp-aware temporal summaries
    - JSON Lines, CSV, JSON summary, and Markdown report output
"""

import argparse
import os
import sys

import yaml

# Add project root to Python path
from paths import PROJECT_ROOT, PROJECT_ROOT_STR

if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def cmd_process(args, config):
    """Process recorded video(s) and generate ergonomic assessment."""
    from pipeline.video_processor import VideoProcessor

    video_paths = args.video
    output_dir = args.output

    # Validate video paths
    for vp in video_paths:
        if not os.path.exists(vp):
            print(f"ERROR: Video file not found: {vp}")
            sys.exit(1)

    # Apply CLI overrides to config
    task_cfg = config.setdefault("task_inputs", {})
    if args.load_kg is not None:
        task_cfg["load_kg"] = args.load_kg
        config.setdefault("rnle", {})["load_weight_kg"] = args.load_kg
    if args.coupling:
        task_cfg["coupling"] = args.coupling
        if args.coupling != "unacceptable":
            config.setdefault("rnle", {})["coupling"] = args.coupling

    # Print configuration summary
    print("=" * 60)
    print("  Ergonomic Assessment System v4")
    print("=" * 60)
    print(f"  Videos: {len(video_paths)}")
    for i, vp in enumerate(video_paths):
        print(f"    [{i}] {vp}")
    print(f"  Output: {output_dir}")
    print("  Sides:  left and right independently; worst side reported")
    print(f"  Load:   {task_cfg.get('load_kg')} kg")
    print(f"  Coupling: {task_cfg.get('coupling')}")

    niosh_cfg = config.get("rnle", {})
    if niosh_cfg:
        print(
            f"  RNLE:   H={niosh_cfg.get('horizontal_cm')}cm "
            f"V={niosh_cfg.get('vertical_cm')}cm "
            f"D={niosh_cfg.get('travel_distance_cm')}cm "
            f"A={niosh_cfg.get('asymmetry_deg')}deg "
            f"F={niosh_cfg.get('frequency_lifts_per_min')}/min"
        )
    print()

    # Process
    processor = VideoProcessor(config)
    result_dir = processor.process(video_paths, output_dir)

    print(f"\n{'=' * 60}")
    print(f"  Results saved to: {result_dir}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Ergonomic Assessment System v4 - RULA/REBA/RNLE from recorded video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single camera:
    python main.py process --video recording.mp4

  Three cameras (front, side, isometric):
    python main.py process --video front.mp4 side.mp4 iso.mp4

  With overrides:
    python main.py process --video video.mp4 --load-kg 5.0 --coupling poor

  Custom config and output:
    python main.py process --video video.mp4 --config my_config.yaml --output results/
        """,
    )

    parser.add_argument(
        "command",
        choices=["process"],
        help="Command to run",
    )
    parser.add_argument(
        "--video",
        "-v",
        nargs="+",
        required=True,
        help="Video file path(s). For multi-camera, provide in order: front side iso",
    )
    parser.add_argument(
        "--config",
        "-c",
        default=str(PROJECT_ROOT / "config.yaml"),
        help="Configuration YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output",
        help="Output directory (default: output/)",
    )
    parser.add_argument(
        "--load-kg",
        type=float,
        default=None,
        help="Override load weight in kg",
    )
    parser.add_argument(
        "--coupling",
        choices=["good", "fair", "poor", "unacceptable"],
        default=None,
        help="Override coupling/grip quality",
    )
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "process":
        cmd_process(args, config)


if __name__ == "__main__":
    main()
