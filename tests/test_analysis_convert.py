"""
Unit tests for converting repo predictions into the official per-panel format.
"""

from pathlib import Path

import pytest

from src.analysis.convert import split_tables, to_official_predictions
from src.analysis.reference import load_panels

METADATA = str(Path(__file__).parent / "fixtures" / "analysis" / "metadata.jsonl")
T1 = "| x | y |\n|---|---|\n| 1 | 10 |"
T2 = "| e | i |\n|---|---|\n| 1.5 | 400 |"
T3 = "| q | r |\n|---|---|\n| 9 | 9 |"


def test_split_tables_ignores_fences_and_text():
    text = f"Here is the table:\n```markdown\n{T1}\n```\nAnd panel b:\n\n{T2}\nDone."
    assert split_tables(text) == [T1, T2]


def test_maps_tables_by_order():
    recs = [{"id": "fig_two", "raw_output": f"{T1}\n\n{T2}"}]
    official, _ = to_official_predictions(recs, load_panels(METADATA))
    assert official[0]["data_extraction"] == {"a": T1, "b": T2}


def test_extra_tables_are_counted():
    recs = [{"id": "fig_two", "raw_output": f"{T1}\n\n{T2}\n\n{T3}"}]
    official, info = to_official_predictions(recs, load_panels(METADATA))
    assert set(official[0]["data_extraction"]) == {"a", "b"}
    assert info[0]["extra_tables"] == 1


def test_maps_to_panel_with_ground_truth():
    # fig_three: panel a has no table, so the only predicted table goes to b
    recs = [{"id": "fig_three", "raw_output": T1}]
    official, _ = to_official_predictions(recs, load_panels(METADATA))
    assert official[0]["data_extraction"] == {"b": T1}


def test_matches_by_basename_without_id():
    recs = [{"image": "images/fig_one.jpg", "raw_output": T1}]
    official, _ = to_official_predictions(recs, load_panels(METADATA))
    assert official[0]["sample_id"] == "ald/exp/1/fig_one"


def test_missing_raw_output_warns():
    recs = [{"id": "fig_one", "markdown_table": T1}]
    with pytest.warns(UserWarning, match="raw_output"):
        official, _ = to_official_predictions(recs, load_panels(METADATA))
    assert official[0]["data_extraction"] == {"a": T1}
