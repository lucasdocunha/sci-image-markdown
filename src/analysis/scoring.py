"""
Official Task 2 score (0.5*RMS + 0.5*TEDS) computed for a single panel.

Calls the unmodified evaluator in third_party/sci_imageminer_eval so that the mean
over panels equals what the official script reports for the same files.
"""

import importlib.util
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

EVAL_DIR = Path(__file__).resolve().parents[2] / "third_party" / "sci_imageminer_eval"
EVAL_SCRIPT = EVAL_DIR / "sci-imageminer_evaluation.py"


@dataclass
class PanelScore:
    pred_valid: bool
    rms: float
    teds: float
    weighted: float


@lru_cache(maxsize=1)
def _official():
    # The official modules import each other as top-level names (metrics, deplot_metric, ...).
    if str(EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(EVAL_DIR))
    spec = importlib.util.spec_from_file_location("sci_imageminer_evaluation", EVAL_SCRIPT)
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)

    import metrics as official_metrics
    from deplot_metric import metrics as deplot_metrics
    from teds_pubtables import teds_pubtables_official

    return evaluation, official_metrics, deplot_metrics, teds_pubtables_official


def is_valid_table(text: Optional[str]) -> bool:
    """Official submission check: a non-empty, strictly formatted markdown table."""
    evaluation = _official()[0]
    return not evaluation.is_empty(text) and evaluation.is_valid_markdown_table_strict(text)


def score_panel(gt_table: str, pred_table: Optional[str]) -> PanelScore:
    """Scores one panel exactly as the official data_extraction task does."""
    evaluation, official_metrics, deplot_metrics, teds = _official()

    valid = is_valid_table(pred_table)
    truth = evaluation.decode_unicode(gt_table)
    pred = evaluation.decode_unicode(pred_table if valid else "")
    if not pred:
        return PanelScore(pred_valid=False, rms=0.0, teds=0.0, weighted=0.0)

    flatten = official_metrics.flatten_markdown_table
    rms = float(deplot_metrics.table_datapoints_precision_recall(
        [[flatten(truth)]], [flatten(pred)]
    )["table_datapoints_f1"])
    teds_score = teds(truth, pred) * 100
    return PanelScore(pred_valid=True, rms=rms, teds=teds_score, weighted=0.5 * rms + 0.5 * teds_score)
