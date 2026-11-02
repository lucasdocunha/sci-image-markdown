#!/usr/bin/env python3
"""Rank diagnostic eval reports saved by run_diag_eval.sh.

Usage:
    python rank_diag.py [EVAL_RESULTS_DIR]

Reads every `diag_*/report.json` under the results directory (default:
outputs/eval_results), prints icdar_score (%) sorted best-to-worst and the
delta vs the `baseline` set when present.
"""

import json
import sys
from pathlib import Path


def main() -> None:
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/eval_results")
    rows = []
    for report in sorted(results_dir.glob("diag_*/report.json")):
        tag = report.parent.name.replace("diag_", "")
        try:
            metrics = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: skipping {report}: {exc}", file=sys.stderr)
            continue
        score = metrics.get("icdar_score")
        if not isinstance(score, (int, float)):
            continue
        rows.append((tag, score))

    if not rows:
        print(f"No diag_*/report.json found under {results_dir}")
        return

    base = dict(rows).get("baseline")

    header = f"{'set':20} {'icdar':>7}" + (f" {'delta_pp':>9}" if base is not None else "")
    print(header)
    print("-" * len(header))
    for tag, score in sorted(rows, key=lambda item: -item[1]):
        delta = ""
        if base is not None:
            delta = f"{100 * (score - base):+.2f}"
        print(f"{tag:20} {100 * score:6.2f} {delta:>9}")


if __name__ == "__main__":
    main()
