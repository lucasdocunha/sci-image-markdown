"""
Tests for official ICDAR 2026 metrics (TEDS, RMS, Composite Score).
"""

import pytest
from src.metrics.table_metrics import (
    compute_teds,
    compute_icdar_rms,
    compute_icdar_score,
)
from src.metrics.table_parser import parse_markdown_to_dataframe


def test_teds_identical_tables():
    md1 = "| Cycle | Thickness (Å) |\n|---|---|\n| 10 | 25.0 |\n| 20 | 50.0 |"
    md2 = "| Cycle | Thickness (Å) |\n|---|---|\n| 10 | 25.0 |\n| 20 | 50.0 |"
    score = compute_teds(md1, md2)
    assert score == pytest.approx(1.0, abs=1e-4)


def test_teds_different_headers():
    md1 = "| Cycle | Thickness (Å) |\n|---|---|\n| 10 | 25.0 |"
    md2 = "| Steps | Growth (nm) |\n|---|---|\n| 10 | 25.0 |"
    score = compute_teds(md1, md2)
    assert 0.0 < score < 1.0


def test_teds_empty_tables():
    assert compute_teds("", "") == 1.0
    assert compute_teds("| A | B |\n|---|---|\n| 1 | 2 |", "") == 0.0
    assert compute_teds("not a table", "| A | B |\n|---|---|\n| 1 | 2 |") == 0.0


def test_icdar_rms_perfect_numerical_match():
    md1 = "| A | B |\n|---|---|\n| 10 | 20.0 |\n| 30 | 40.0 |"
    df1, _ = parse_markdown_to_dataframe(md1)
    df2, _ = parse_markdown_to_dataframe(md1)
    rms = compute_icdar_rms(df1, df2, rel_tol=0.05)
    assert rms == pytest.approx(1.0, abs=1e-4)


def test_icdar_rms_within_tolerance():
    md1 = "| A | B |\n|---|---|\n| 100.0 | 200.0 |"
    md2 = "| A | B |\n|---|---|\n| 102.0 | 196.0 |"  # within 5%
    df1, _ = parse_markdown_to_dataframe(md1)
    df2, _ = parse_markdown_to_dataframe(md2)
    rms = compute_icdar_rms(df1, df2, rel_tol=0.05)
    assert rms > 0.95


def test_icdar_rms_out_of_tolerance():
    md1 = "| A | B |\n|---|---|\n| 100.0 | 200.0 |"
    md2 = "| A | B |\n|---|---|\n| 150.0 | 300.0 |"  # >5% error
    df1, _ = parse_markdown_to_dataframe(md1)
    df2, _ = parse_markdown_to_dataframe(md2)
    rms = compute_icdar_rms(df1, df2, rel_tol=0.05)
    assert rms == 0.0


def test_compute_icdar_composite_score():
    md1 = "| Cycle | Thickness (Å) |\n|---|---|\n| 10 | 25.0 |\n| 20 | 50.0 |"
    md2 = "| Cycle | Thickness (Å) |\n|---|---|\n| 10 | 25.0 |\n| 20 | 50.0 |"
    res = compute_icdar_score(md1, md2, rel_tol=0.05)
    assert res["teds"] == pytest.approx(1.0, abs=1e-4)
    assert res["rms"] == pytest.approx(1.0, abs=1e-4)
    assert res["icdar_score"] == pytest.approx(1.0, abs=1e-4)
