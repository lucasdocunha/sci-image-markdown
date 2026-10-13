"""
Unit tests for per-panel ground truth loading.
"""

from pathlib import Path

from src.analysis.reference import load_panels, to_official_reference

METADATA = str(Path(__file__).parent / "fixtures" / "analysis" / "metadata.jsonl")


def test_only_panels_with_table_are_loaded():
    panels = load_panels(METADATA)
    assert [(p.figure_id, p.panel_id) for p in panels] == [
        ("fig_one", "a"), ("fig_two", "a"), ("fig_two", "b"), ("fig_three", "b"),
    ]


def test_n_panels_counts_panels_with_table():
    third = [p for p in load_panels(METADATA) if p.figure_id == "fig_three"]
    assert len(third) == 1
    assert third[0].n_panels == 1


def test_panel_fields_come_from_metadata():
    b = [p for p in load_panels(METADATA) if p.figure_id == "fig_two" and p.panel_id == "b"][0]
    assert b.figure_type == "spectra chart"
    assert b.image_file == "fig_two.jpg"
    assert b.bbox == {"x": 50, "y": 0, "width": 50, "height": 80}


def test_official_reference_groups_by_sample():
    ref = to_official_reference(load_panels(METADATA))
    two = [r for r in ref if r["sample_id"] == "ald/exp/1/fig_two"][0]
    assert set(two["data_extraction"]) == {"a", "b"}
