"""
Per-panel scoring must reproduce the official evaluator exactly.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.analysis.reference import load_panels, to_official_reference
from src.analysis.scoring import EVAL_SCRIPT, score_panel

METADATA = str(Path(__file__).parent / "fixtures" / "analysis" / "metadata.jsonl")

# fig_one: exact copy; fig_two a: wrong values; fig_two b: no spaces around "|";
# fig_three: missing from predictions on purpose.
PREDICTIONS = [
    {"sample_id": "ald/exp/1/fig_one", "data_extraction": {
        "a": "| x | y |\n|---|---|\n| 1 | 10 |\n| 2 | 20 |\n| 3 | 30 |"}},
    {"sample_id": "ald/exp/1/fig_two", "data_extraction": {
        "a": "| T (C) | k |\n|---|---|\n| 30 | 9.9 |\n| 60 | 5.0 |\n| 90 | 1.0 |",
        "b": "|E (eV)|I|\n|---|---|\n|1.5|400|\n|2.0|900|"}},
]


def test_mean_matches_official_script(tmp_path):
    panels = load_panels(METADATA)
    ref_path = tmp_path / "reference.json"
    ref_path.write_text(json.dumps(to_official_reference(panels)))
    pred_path = tmp_path / "predictions.jsonl"
    pred_path.write_text("\n".join(json.dumps(p) for p in PREDICTIONS) + "\n")
    out_path = tmp_path / "scores.json"

    subprocess.run([
        sys.executable, str(EVAL_SCRIPT), "--task", "data_extraction",
        "--reference-path", str(ref_path), "--prediction-path", str(pred_path),
        "--output-scores-path", str(out_path),
    ], check=True, cwd=EVAL_SCRIPT.parent)
    official = json.loads(out_path.read_text())

    preds = {p["sample_id"]: p["data_extraction"] for p in PREDICTIONS}
    scores = [score_panel(p.gt_table, preds.get(p.sample_id, {}).get(p.panel_id)) for p in panels]
    n = len(scores)
    assert sum(s.rms for s in scores) / n == pytest.approx(official["rms"], abs=1e-9)
    assert sum(s.teds for s in scores) / n == pytest.approx(official["teds"], abs=1e-9)
    assert sum(s.weighted for s in scores) / n == pytest.approx(official["weighted_score"], abs=1e-9)


def test_table_without_spaces_gets_zero_rms():
    gt = "| x | y |\n|---|---|\n| 33 | 7.5 |"
    assert score_panel(gt, "|x|y|\n|---|---|\n|33|7.5|").rms == 0


def test_invalid_prediction_scores_zero():
    gt = "| x | y |\n|---|---|\n| 33 | 7.5 |"
    s = score_panel(gt, "x, y\n33, 7.5")
    assert (s.pred_valid, s.weighted) == (False, 0.0)
