"""
CLI tests for analyze_errors.py.
"""

import json
from pathlib import Path

import pandas as pd
from click.testing import CliRunner

from analyze_errors import main

METADATA = str(Path(__file__).parent / "fixtures" / "analysis" / "metadata.jsonl")


def test_missing_figure_scores_zero(tmp_path):
    preds = tmp_path / "official.jsonl"
    preds.write_text(json.dumps({
        "sample_id": "ald/exp/1/fig_one",
        "data_extraction": {"a": "| x | y |\n|---|---|\n| 1 | 10 |\n| 2 | 20 |\n| 3 | 30 |"},
    }) + "\n")
    out = tmp_path / "out"

    result = CliRunner().invoke(main, [
        "--metadata", METADATA, "--official-predictions", str(preds), "--out-dir", str(out),
    ])

    assert result.exit_code == 0, result.output
    df = pd.read_csv(out / "panels.csv")
    summary = json.loads((out / "summary.json").read_text())
    assert summary["n_panels"] == 4
    assert (df[df["sample_id"] != "ald/exp/1/fig_one"]["weighted"] == 0).all()
    assert df.loc[df["sample_id"] == "ald/exp/1/fig_one", "pred_rows"].item() == 3


def test_panels_csv_has_categories_and_gt_flag(tmp_path):
    preds = tmp_path / "official.jsonl"
    preds.write_text(json.dumps({
        "sample_id": "ald/exp/1/fig_one",
        "data_extraction": {"a": "| x | y |\n|---|---|\n| 1 | 10 |\n| 2 | 20 |\n| 3 | 30 |"},
    }) + "\n")
    out = tmp_path / "out"

    CliRunner().invoke(main, ["--metadata", METADATA, "--official-predictions", str(preds), "--out-dir", str(out)])

    df = pd.read_csv(out / "panels.csv", keep_default_na=False).set_index(["sample_id", "panel_id"])
    assert df.loc[("ald/exp/1/fig_one", "a"), "categorias"] == ""
    assert df.loc[("ald/exp/1/fig_two", "b"), "categorias"] == "painel_faltando"
    assert df["gt_suspeito"].astype(str).eq("False").all()


def test_requires_exactly_one_prediction_source(tmp_path):
    result = CliRunner().invoke(main, ["--metadata", METADATA, "--out-dir", str(tmp_path)])
    assert result.exit_code != 0
    assert "exactly one of --repo-predictions or --official-predictions" in result.output
