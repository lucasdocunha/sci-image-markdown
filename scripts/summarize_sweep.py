"""
Collects the per-budget reports written by run_b0_sweep.sh into one table.

Reads outputs/b0_sweep/res_<budget>/report_*.json and prints metrics side by
side, so the effect of the resolution budget is readable in one place.
"""

import json
import sys
from pathlib import Path

# Ordered by how much they matter for Task 2: structural validity first, then
# numerical fidelity, then the lexical proxies.
HEADLINE_METRICS = [
    "valid_table",
    "cell_recall",
    "cell_f1",
    "cell_rmse",
    "edit_similarity",
    "rouge_2",
]
LOWER_IS_BETTER = {"cell_rmse", "cell_rne"}


def approx_px(budget: int) -> int:
    """Visual-token budget back to an approximate square edge in pixels."""
    return int((budget * 28 * 28) ** 0.5)


def load_runs(root: Path, kind: str):
    runs = []
    for run_dir in sorted(root.glob("res_*"), key=lambda p: int(p.name.split("_")[1])):
        report = run_dir / f"report_{kind}.json"
        if not report.exists():
            continue
        budget = int(run_dir.name.split("_")[1])
        runs.append((budget, json.loads(report.read_text(encoding="utf-8"))))
    return runs


def print_table(runs, kind: str) -> None:
    print(f"\n=== {kind} ===")
    header = f"{'budget':>8} {'~px':>7}" + "".join(f"{m:>17}" for m in HEADLINE_METRICS)
    print(header)
    print("-" * len(header))

    best = {}
    for metric in HEADLINE_METRICS:
        vals = [(b, r.get(metric)) for b, r in runs if isinstance(r.get(metric), (int, float))]
        if not vals:
            continue
        pick = min if metric in LOWER_IS_BETTER else max
        best[metric] = pick(vals, key=lambda kv: kv[1])[0]

    for budget, report in runs:
        row = f"{budget:>8} {approx_px(budget):>6}px"
        for metric in HEADLINE_METRICS:
            value = report.get(metric)
            if not isinstance(value, (int, float)):
                row += f"{'-':>17}"
                continue
            marker = " *" if best.get(metric) == budget else "  "
            row += f"{value:>15.4f}{marker}"
        print(row)
    print("\n(* = best for that metric; cell_rmse is lower-is-better)")


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/b0_sweep")
    if not root.exists():
        print(f"No sweep directory at {root}")
        return 1

    found = False
    for kind in ("finetuned", "base"):
        runs = load_runs(root, kind)
        if runs:
            print_table(runs, kind)
            found = True

    if not found:
        print(f"No reports found under {root}. Did the sweep get past training?")
        return 1

    print(
        "\nReminder: all arms train for a fixed number of epochs with no checkpoint\n"
        "selection, so the per-budget numbers stay comparable to each other."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
