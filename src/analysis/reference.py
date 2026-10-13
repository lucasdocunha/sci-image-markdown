"""
Per-panel ground truth loaded from the Sci-ImageMiner HF metadata.jsonl.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..utils.io import read_jsonl


@dataclass
class PanelRef:
    sample_id: str
    figure_id: str
    image_file: str
    panel_id: str
    gt_table: str
    figure_type: str
    n_panels: int
    bbox: Optional[dict]


def _by_panel(items) -> Dict[str, dict]:
    return {d.get("panel_id"): d for d in (items or []) if isinstance(d, dict)}


def load_panels(metadata_path: str) -> List[PanelRef]:
    """Returns one PanelRef per panel with a non-empty table, in data_extraction order."""
    panels: List[PanelRef] = []
    for rec in read_jsonl(metadata_path):
        tables = [
            d for d in (rec.get("data_extraction") or [])
            if isinstance(d, dict) and (d.get("data") or "").strip()
        ]
        classes = _by_panel(rec.get("classification"))
        boxes = _by_panel(rec.get("bbox"))
        for d in tables:
            pid = d["panel_id"]
            box = boxes.get(pid)
            panels.append(PanelRef(
                sample_id=rec["sample_id"],
                figure_id=rec.get("id", ""),
                image_file=os.path.basename(rec.get("file_name", "")),
                panel_id=pid,
                gt_table=d["data"],
                figure_type=classes.get(pid, {}).get("label", "unknown"),
                n_panels=len(tables),
                bbox={k: box[k] for k in ("x", "y", "width", "height")} if box else None,
            ))
    return panels


def to_official_reference(panels: List[PanelRef]) -> List[dict]:
    """Builds the reference JSON consumed by the official evaluation script."""
    by_sample: Dict[str, dict] = {}
    for p in panels:
        by_sample.setdefault(p.sample_id, {"sample_id": p.sample_id, "data_extraction": {}})
        by_sample[p.sample_id]["data_extraction"][p.panel_id] = p.gt_table
    return list(by_sample.values())
