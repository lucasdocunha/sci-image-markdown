"""
CLI tool for per-panel error analysis with the official Task 2 score.

Works with predictions from any model: either the repo's predictions.jsonl
(written by evaluate.py) or a JSONL already in the official submission format.
"""

import json
import os
from typing import Optional, Tuple

import click
import pandas as pd

from src.analysis.convert import to_official_predictions
from src.analysis.reference import load_panels
from src.analysis.scoring import score_panel
from src.analysis.taxonomy import classify_panel, flag_suspicious_gt
from src.utils.io import read_jsonl, write_jsonl
from src.utils.logging import setup_logger

logger = setup_logger("analyze_errors")


def table_shape(table: Optional[str]) -> Tuple[int, int]:
    """Returns (data_rows, columns) of a markdown table, or (0, 0) without one."""
    lines = [ln.strip() for ln in (table or "").splitlines() if ln.strip()]
    if not lines:
        return 0, 0
    return max(0, len(lines) - 2), len(lines[0].strip("|").split("|"))


@click.command()
@click.option("--metadata", default="data/raw/validation_metadata.jsonl", help="HF metadata.jsonl of the split to analyze.")
@click.option("--repo-predictions", default=None, help="predictions.jsonl written by evaluate.py.")
@click.option("--official-predictions", default=None, help="Predictions JSONL in the official submission format.")
@click.option("--out-dir", required=True, help="Directory for panels.csv, summary.json and official_predictions.jsonl.")
def main(metadata: str, repo_predictions: str, official_predictions: str, out_dir: str):
    """Scores every panel with the official metric and writes a per-panel table."""
    if bool(repo_predictions) == bool(official_predictions):
        raise click.UsageError("Pass exactly one of --repo-predictions or --official-predictions.")

    panels = load_panels(metadata)
    logger.info(f"Loaded {len(panels)} panels with ground truth from {metadata}")

    summary = {}
    if repo_predictions:
        preds, info = to_official_predictions(read_jsonl(repo_predictions), panels)
        summary["extra_tables"] = sum(i["extra_tables"] for i in info)
    else:
        preds = read_jsonl(official_predictions)

    os.makedirs(out_dir, exist_ok=True)
    write_jsonl(preds, os.path.join(out_dir, "official_predictions.jsonl"))
    by_sample = {p["sample_id"]: p.get("data_extraction", {}) for p in preds}

    rows = []
    for p in panels:
        pred_table = by_sample.get(p.sample_id, {}).get(p.panel_id)
        score = score_panel(p.gt_table, pred_table)
        gt_rows, gt_cols = table_shape(p.gt_table)
        pred_rows, pred_cols = table_shape(pred_table)
        rows.append({
            "sample_id": p.sample_id,
            "panel_id": p.panel_id,
            "figure_type": p.figure_type,
            "n_panels": p.n_panels,
            "gt_rows": gt_rows,
            "gt_cols": gt_cols,
            "pred_rows": pred_rows,
            "pred_cols": pred_cols,
            "pred_valid": score.pred_valid,
            "rms": score.rms,
            "teds": score.teds,
            "weighted": score.weighted,
            "categorias": ";".join(classify_panel(p.gt_table, pred_table)),
            "gt_suspeito": flag_suspicious_gt(p.gt_table),
            "gt_table": p.gt_table,
            "pred_table": pred_table or "",
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "panels.csv"), index=False)
    summary.update({
        "n_panels": len(df),
        "rms": float(df["rms"].mean()),
        "teds": float(df["teds"].mean()),
        "weighted": float(df["weighted"].mean()),
    })
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Official score over {len(df)} panels: {summary['weighted']:.2f} -> {out_dir}")


if __name__ == "__main__":
    main()
