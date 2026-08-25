"""
End-to-end evaluation pipeline for scientific image to markdown tables.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from .table_parser import parse_markdown_to_dataframe
from .table_metrics import compute_normalized_edit_distance, compute_numerical_cell_metrics


class TableExtractionEvaluator:
    """Evaluates predicted markdown tables against ground truth tables."""

    def __init__(self, rel_tol: float = 0.05):
        self.rel_tol = rel_tol
        self.rouge_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        self.smooth_fn = SmoothingFunction().method1

    def evaluate_sample(self, pred_text: str, target_text: str) -> Dict[str, Any]:
        """Evaluates a single prediction against a single ground truth target."""
        pred_df, is_pred_valid = parse_markdown_to_dataframe(pred_text)
        target_df, is_target_valid = parse_markdown_to_dataframe(target_text)

        # Edit distance
        edit_sim = compute_normalized_edit_distance(pred_text.strip(), target_text.strip())

        # Lexical metrics
        rouge_res = self.rouge_scorer.score(target_text, pred_text)
        rouge_l = rouge_res["rougeL"].fmeasure
        rouge_1 = rouge_res["rouge1"].fmeasure
        rouge_2 = rouge_res["rouge2"].fmeasure

        target_tokens = target_text.split()
        pred_tokens = pred_text.split()
        bleu_4 = sentence_bleu([target_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smooth_fn)

        # Exact match
        exact_match = 1.0 if pred_text.strip() == target_text.strip() else 0.0

        # Numerical metrics
        num_metrics = compute_numerical_cell_metrics(pred_df, target_df, rel_tol=self.rel_tol)

        return {
            "valid_table": 1.0 if is_pred_valid else 0.0,
            "exact_match": exact_match,
            "edit_similarity": edit_sim,
            "rouge_l": rouge_l,
            "rouge_1": rouge_1,
            "rouge_2": rouge_2,
            "bleu_4": bleu_4,
            "cell_precision": num_metrics["precision"],
            "cell_recall": num_metrics["recall"],
            "cell_f1": num_metrics["f1"],
            "cell_rmse": num_metrics["rmse"],
            "cell_rne": num_metrics["mean_rne"],
        }

    def evaluate_batch(self, predictions: List[str], targets: List[str]) -> Dict[str, float]:
        """Evaluates a batch or full dataset of predictions."""
        if len(predictions) != len(targets):
            raise ValueError(f"Predictions count ({len(predictions)}) != Targets count ({len(targets)})")

        sample_metrics = [self.evaluate_sample(p, t) for p, t in zip(predictions, targets)]
        
        aggregated: Dict[str, float] = {}
        for key in sample_metrics[0].keys():
            vals = [m[key] for m in sample_metrics if not np.isnan(m[key])]
            aggregated[key] = float(np.mean(vals)) if vals else 0.0

        return aggregated
