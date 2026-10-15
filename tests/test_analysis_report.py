"""
Unit tests for category costs and score breakdowns.
"""

import pandas as pd
import pytest

from src.analysis.report import category_costs, slice_scores


def panels(rows):
    return pd.DataFrame(rows, columns=["categorias", "weighted", "figure_type"])


def test_cost_is_points_lost_over_all_panels():
    df = panels([
        ("escala_magnitude", 50.0, "line chart"),
        ("escala_magnitude;amostragem_linhas", 50.0, "line chart"),
        ("", 100.0, "bar chart"),
        ("", 100.0, "bar chart"),
    ])
    costs = category_costs(df).set_index("categoria")
    assert costs.loc["escala_magnitude", "n"] == 2
    assert costs.loc["escala_magnitude", "custo_total"] == pytest.approx(25.0)
    # only the first panel has escala_magnitude alone
    assert costs.loc["escala_magnitude", "custo_exclusivo"] == pytest.approx(12.5)
    assert costs.loc["amostragem_linhas", "custo_exclusivo"] == pytest.approx(0.0)


def test_costs_sorted_by_total_cost():
    df = panels([("painel_faltando", 0.0, "x"), ("escala_magnitude", 90.0, "x")])
    assert category_costs(df)["categoria"].tolist()[:2] == ["painel_faltando", "escala_magnitude"]


def test_slice_scores_groups_with_counts():
    df = panels([("", 10.0, "line chart"), ("", 30.0, "line chart"), ("", 80.0, "bar chart")])
    out = slice_scores(df, "figure_type").set_index("figure_type")
    assert out.loc["line chart", "n"] == 2
    assert out.loc["line chart", "weighted"] == pytest.approx(20.0)
    assert out.index.tolist()[0] == "line chart"  # most panels first
