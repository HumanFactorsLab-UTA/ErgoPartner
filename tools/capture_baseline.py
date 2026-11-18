"""Capture or compare the pipeline regression baseline.

    python tools/capture_baseline.py             # compare against the recorded baseline
    python tools/capture_baseline.py --write     # record the current output as the baseline

Comparing is the useful mode during a change: it reports which angles moved, by
how much, and how many scores changed, so an intentional correction can be
reviewed on its numbers rather than accepted on faith.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tests.regression import baseline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="record the current output as the new baseline instead of comparing",
    )
    arguments = parser.parse_args()

    available = [item for item in baseline.SCENARIOS if baseline.footage_available(item)]
    if not available:
        print(baseline.missing_footage_reason(), file=sys.stderr)
        return 2

    skipped = [item.name for item in baseline.SCENARIOS if item not in available]
    if skipped:
        print(f"Skipping scenarios without footage: {', '.join(skipped)}\n")

    print(
        f"Running {len(available)} scenario(s) over frames "
        f"{baseline.START_FRAME}-{baseline.END_FRAME}..."
    )
    for scenario in available:
        print(f"  - {scenario.name}: {scenario.description}")
    snapshot = baseline.capture(tuple(available))

    if arguments.write:
        baseline.save_baseline(snapshot)
        print(f"\nBaseline written to {baseline.BASELINE_PATH}")
        for name, entry in sorted(snapshot["scenarios"].items()):
            counts = entry["counts"]
            print(
                f"  {name}: {len(entry['frames'])} frames, "
                f"RULA {counts['valid_rula_observations']}, "
                f"REBA {counts['valid_reba_observations']}"
            )
        return 0

    stored = baseline.load_baseline()
    if stored is None:
        print("\nNo baseline recorded yet. Run with --write to create one.", file=sys.stderr)
        return 2

    differences = baseline.compare(stored, snapshot)
    if not differences:
        print("\nNo change against the recorded baseline.")
        return 0
    print(f"\n{len(differences)} difference(s) against the recorded baseline:\n")
    for line in differences:
        print(f"  {line}")
    print("\nIf these changes are intended, re-record with --write.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
