"""
Converts the repo's predictions.jsonl into the official per-panel submission format.
"""

import os
import warnings
from typing import Dict, List, Tuple

from .reference import PanelRef


def split_tables(text: str) -> List[str]:
    """Returns every pipe-delimited table block in the text, in order.

    A block is a maximal run of lines starting with "|" (code fences and prose
    end a run), and only runs of 2+ lines count as a table.
    """
    if not text:
        return []
    lines = [ln.strip() for ln in text.splitlines()]
    tables, run = [], []
    for ln in lines + [""]:
        if ln.startswith("|"):
            run.append(ln)
            continue
        if len(run) >= 2:
            tables.append("\n".join(run))
        run = []
    return tables


def to_official_predictions(
    pred_records: List[dict], panels: List[PanelRef]
) -> Tuple[List[dict], List[dict]]:
    """Maps each record's tables onto its figure's panels by order.

    Records are matched to figures by "id", falling back to the image basename.
    Returns (official_predictions, info) where info has one row per figure with
    the number of predicted tables and how many were left over.
    """
    fig_panels: Dict[str, List[PanelRef]] = {}
    by_image: Dict[str, str] = {}
    for p in panels:
        fig_panels.setdefault(p.figure_id, []).append(p)
        by_image[p.image_file] = p.figure_id

    missing_raw = 0
    official, info = [], []
    for rec in pred_records:
        fid = rec.get("id")
        if fid not in fig_panels:
            fid = by_image.get(os.path.basename(rec.get("image", "")))
        if fid is None:
            continue

        if "raw_output" in rec:
            text = rec["raw_output"]
        else:
            missing_raw += 1
            text = rec.get("markdown_table", "")

        targets = fig_panels[fid]
        tables = split_tables(text)
        official.append({
            "sample_id": targets[0].sample_id,
            "data_extraction": {p.panel_id: t for p, t in zip(targets, tables)},
        })
        info.append({
            "sample_id": targets[0].sample_id,
            "n_pred_tables": len(tables),
            "extra_tables": max(0, len(tables) - len(targets)),
        })

    if missing_raw:
        warnings.warn(
            f"{missing_raw} records have no raw_output; using markdown_table, which keeps "
            "only the first table, so multi-panel figures are under-scored.",
            UserWarning,
        )
    return official, info
