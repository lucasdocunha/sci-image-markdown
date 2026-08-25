"""
Domain metrics for Table Extraction: RNE, RMSE, Normalized Edit Distance, Precision, Recall, F1.
"""

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import Levenshtein


def compute_normalized_edit_distance(pred_str: str, target_str: str) -> float:
    """Computes normalized Levenshtein edit distance between two strings [0, 1].
    
    1.0 means identical, 0.0 means completely different.
    """
    if not pred_str and not target_str:
        return 1.0
    if not pred_str or not target_str:
        return 0.0

    dist = Levenshtein.distance(pred_str, target_str)
    max_len = max(len(pred_str), len(target_str))
    if max_len == 0:
        return 1.0
    return max(0.0, 1.0 - (dist / max_len))


def extract_numerical_cells(df: pd.DataFrame) -> List[float]:
    """Extracts all numeric float values present in a DataFrame."""
    values = []
    for col in df.columns:
        for val in df[col]:
            try:
                num = float(val)
                if not (math.isnan(num) or math.isinf(num)):
                    values.append(num)
            except (ValueError, TypeError):
                continue
    return values


def compute_numerical_cell_metrics(
    pred_df: Optional[pd.DataFrame],
    target_df: Optional[pd.DataFrame],
    rel_tol: float = 0.05
) -> Dict[str, float]:
    """Computes numerical cell precision, recall, F1, RMSE, and RNE.
    
    Matches extracted values against ground-truth values within `rel_tol` (default 5%).
    """
    if pred_df is None or target_df is None:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "rmse": float("nan"),
            "mean_rne": float("nan"),
        }

    pred_vals = extract_numerical_cells(pred_df)
    target_vals = extract_numerical_cells(target_df)

    if not target_vals:
        return {
            "precision": 1.0 if not pred_vals else 0.0,
            "recall": 1.0,
            "f1": 1.0 if not pred_vals else 0.0,
            "rmse": 0.0 if not pred_vals else float("nan"),
            "mean_rne": 0.0 if not pred_vals else float("nan"),
        }

    if not pred_vals:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "rmse": float("nan"),
            "mean_rne": float("nan"),
        }

    matched_targets = set()
    matched_preds = set()
    paired_errors = []
    paired_rnes = []

    # Greedy bipartite matching based on minimum relative difference
    candidates = []
    for p_idx, p in enumerate(pred_vals):
        for t_idx, t in enumerate(target_vals):
            denom = abs(t) if abs(t) > 1e-7 else 1e-7
            rne = abs(p - t) / denom
            if rne <= rel_tol:
                candidates.append((rne, abs(p - t), p_idx, t_idx))

    candidates.sort(key=lambda x: x[0])

    for rne, abs_err, p_idx, t_idx in candidates:
        if p_idx not in matched_preds and t_idx not in matched_targets:
            matched_preds.add(p_idx)
            matched_targets.add(t_idx)
            paired_errors.append(abs_err ** 2)
            paired_rnes.append(rne)

    tp = len(matched_preds)
    precision = tp / len(pred_vals) if pred_vals else 0.0
    recall = tp / len(target_vals) if target_vals else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    rmse = math.sqrt(np.mean(paired_errors)) if paired_errors else float("nan")
    mean_rne = float(np.mean(paired_rnes)) if paired_rnes else float("nan")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "rmse": rmse,
        "mean_rne": mean_rne,
    }
