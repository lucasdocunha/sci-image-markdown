"""
Unit tests for comparing rule categories against the manual review.
"""

import pandas as pd
import pytest

from src.analysis.calibration import rule_precision


def test_rule_precision_against_manual_review(tmp_path):
    panels = tmp_path / "panels.csv"
    pd.DataFrame([
        {"sample_id": "s1", "panel_id": "a", "categorias": "escala_magnitude", "gt_suspeito": False},
        {"sample_id": "s2", "panel_id": "a", "categorias": "escala_magnitude;amostragem_linhas", "gt_suspeito": True},
        {"sample_id": "s3", "panel_id": "a", "categorias": "", "gt_suspeito": False},
        {"sample_id": "s4", "panel_id": "a", "categorias": "escala_magnitude", "gt_suspeito": False},
    ]).to_csv(panels, index=False)
    review = tmp_path / "review.csv"
    # Same columns the review notebook writes
    pd.DataFrame([
        {"sample_id": "s1", "panel_id": "a", "categorias": "escala_magnitude", "gt_suspeito": "não"},
        {"sample_id": "s2", "panel_id": "a", "categorias": "amostragem_linhas", "gt_suspeito": "sim"},
        {"sample_id": "s3", "panel_id": "a", "categorias": "escala_magnitude", "gt_suspeito": "sim"},
        {"sample_id": "s4", "panel_id": "a", "categorias": "", "gt_suspeito": ""},  # not reviewed yet
    ]).assign(origem="pior", weighted=10.0, categorias_regra="x", observacao="").to_csv(review, index=False)

    df = rule_precision(str(panels), str(review)).set_index("categoria")

    # escala_magnitude: rule hit s1 (right) and s2 (wrong), missed s3; s4 is not reviewed
    assert df.loc["escala_magnitude", ["tp", "fp", "fn"]].tolist() == [1, 1, 1]
    assert df.loc["escala_magnitude", "precision"] == pytest.approx(0.5)
    assert df.loc["amostragem_linhas", ["tp", "fp", "fn"]].tolist() == [1, 0, 0]
    assert df.loc["gt_suspeito", ["tp", "fp", "fn"]].tolist() == [1, 0, 1]


def test_nothing_reviewed_yet_gives_zero_counts(tmp_path):
    panels = tmp_path / "panels.csv"
    pd.DataFrame([{"sample_id": "s1", "panel_id": "a", "categorias": "escala_magnitude", "gt_suspeito": False}]).to_csv(panels, index=False)
    review = tmp_path / "review.csv"
    pd.DataFrame([{"sample_id": "s1", "panel_id": "a", "categorias": "", "gt_suspeito": ""}]).to_csv(review, index=False)

    df = rule_precision(str(panels), str(review))

    assert (df[["tp", "fp", "fn"]] == 0).all().all()
