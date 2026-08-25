"""
Unit tests for evaluation metrics calculation.
"""

import pandas as pd
import pytest
from src.metrics.table_metrics import (
    compute_normalized_edit_distance,
    compute_numerical_cell_metrics,
    extract_numerical_cells,
)
from src.metrics.evaluator import TableExtractionEvaluator


def test_normalized_edit_distance():
    assert compute_normalized_edit_distance("hello", "hello") == 1.0
    assert compute_normalized_edit_distance("", "") == 1.0
    sim = compute_normalized_edit_distance("abc", "abd")
    assert 0.6 < sim < 0.7


def test_extract_numerical_cells():
    df = pd.DataFrame({
        "A": [1.0, 2.5, "text"],
        "B": ["3.14", 4.0, None]
    })
    cells = extract_numerical_cells(df)
    assert sorted(cells) == [1.0, 2.5, 3.14, 4.0]


def test_compute_numerical_cell_metrics_perfect_match():
    df = pd.DataFrame({"X": [10.0, 20.0], "Y": [1.0, 2.0]})
    metrics = compute_numerical_cell_metrics(df, df, rel_tol=0.05)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["rmse"] == 0.0
    assert metrics["mean_rne"] == 0.0


def test_compute_numerical_cell_metrics_partial_match():
    pred_df = pd.DataFrame({"X": [10.0, 22.0], "Y": [1.0, 5.0]})
    target_df = pd.DataFrame({"X": [10.0, 20.0], "Y": [1.0, 2.0]})
    # 10.0 matches 10.0, 1.0 matches 1.0, 22.0 vs 20.0 has 10% diff > 5% tol, 5.0 vs 2.0 has 150% diff
    metrics = compute_numerical_cell_metrics(pred_df, target_df, rel_tol=0.05)
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_table_extraction_evaluator():
    evaluator = TableExtractionEvaluator()
    table_gt = (
        "| Temp | Rate |\n"
        "|---|---|\n"
        "| 100 | 1.0 |\n"
        "| 200 | 2.0 |\n"
    )
    table_pred = (
        "| Temp | Rate |\n"
        "|---|---|\n"
        "| 100 | 1.0 |\n"
        "| 200 | 2.0 |\n"
    )
    res = evaluator.evaluate_sample(table_pred, table_gt)
    assert res["valid_table"] == 1.0
    assert res["exact_match"] == 1.0
    assert res["cell_f1"] == 1.0
    assert res["rouge_l"] == 1.0
