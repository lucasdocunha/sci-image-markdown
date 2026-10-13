"""
evaluate.py must save the full model output: markdown_table keeps only the first
table, so multi-panel figures can only be scored per panel from raw_output.
"""

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from click.testing import CliRunner

import evaluate

RAW = "| a |\n|---|\n| 1 |\n\n| b |\n|---|\n| 2 |"


class FakePredictor:
    def __init__(self, cfg, adapter_path=None):
        pass

    def predict_image(self, path):
        return {"raw_output": RAW, "markdown_table": "| a |\n|---|\n| 1 |", "is_valid_table": True}


def test_saved_predictions_have_id_and_raw_output(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "TablePredictor", FakePredictor)
    val = tmp_path / "val.jsonl"
    val.write_text(json.dumps({"id": "fig_x", "image": "images/fig_x.jpg", "table": "| a |\n|---|\n| 1 |"}) + "\n")
    preds = tmp_path / "preds.jsonl"

    result = CliRunner().invoke(evaluate.main, [
        "--test-file", str(val),
        "--output-report", str(tmp_path / "report.json"),
        "--save-predictions", str(preds),
    ])

    assert result.exit_code == 0, result.output
    item = json.loads(preds.read_text().splitlines()[0])
    assert item["id"] == "fig_x"
    assert "| b |" in item["raw_output"]
