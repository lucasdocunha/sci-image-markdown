"""
Tests for the decomposed (experiment B) table representation: skeleton / values / fill.
"""

import json
import os

from src.data.decomposed import (
    PLACEHOLDER,
    FusionTextDataset,
    build_fusion_user_text,
    fill_skeleton,
    make_skeleton,
    make_values,
    parse_values,
    split_blocks,
)
from src.data.preprocessor import format_qwen_vl_conversation
from src.metrics.table_parser import parse_markdown_to_dataframe


SINGLE = (
    "| Temp (°C) | k | Field (MV/cm) |\n"
    "|---|---|---|\n"
    "| 33 | 7.5 | 3.7 |\n"
    "| 58 | 8.2 | 4.0 |\n"
    "| 125 |  8.4 ; 7.5 | 4.2 |"
)

MULTI = (
    "| x | y |\n"
    "|---|---|\n"
    "| 1 | 2 |\n"
    "| 3 | 4 |\n\n"
    "| a | b | c |\n"
    "|---|---|---|\n"
    "| 5 | 6 | 7 |"
)


def _cells(md: str):
    """Normalizes a Markdown table(s) to a list of tables -> rows -> cells, ignoring separators."""
    out = []
    for lines in split_blocks(md):
        rows = []
        for ln in lines:
            if "-" in ln and set(ln.replace("|", "").replace(" ", "")) <= {"-", ":"}:
                continue  # separator row only (an all-empty data row keeps its cells)
            rows.append([c.strip() for c in ln.strip().strip("|").split("|")])
        out.append(rows)
    return out


def test_skeleton_keeps_headers_and_masks_body():
    skel = make_skeleton(SINGLE)
    rows = _cells(skel)[0]
    assert rows[0] == ["Temp (°C)", "k", "Field (MV/cm)"]
    assert all(c == PLACEHOLDER for r in rows[1:] for c in r)
    assert len(rows) == 4  # header + 3 body rows


def test_values_has_no_header():
    vals = make_values(SINGLE)
    rows = _cells(vals)[0]
    assert rows[0] == ["33", "7.5", "3.7"]
    assert len(rows) == 3


def test_round_trip_single_table():
    rebuilt = fill_skeleton(make_skeleton(SINGLE), make_values(SINGLE))
    assert _cells(rebuilt) == _cells(SINGLE)


def test_round_trip_multi_table():
    rebuilt = fill_skeleton(make_skeleton(MULTI), make_values(MULTI))
    assert _cells(rebuilt) == _cells(MULTI)
    assert len(split_blocks(rebuilt)) == 2


def test_round_trip_is_parseable_by_repo_parser():
    rebuilt = fill_skeleton(make_skeleton(SINGLE), make_values(SINGLE))
    df, ok = parse_markdown_to_dataframe(rebuilt)
    assert ok and df is not None and df.shape[0] == 3


def test_fill_tolerates_missing_values():
    skel = make_skeleton(SINGLE)
    partial = "| 33 | 7.5 | 3.7 |"  # only first row predicted
    rows = _cells(fill_skeleton(skel, partial))[0]
    assert rows[1] == ["33", "7.5", "3.7"]
    assert rows[2] == ["", "", ""]  # missing rows become empty, no crash


def test_fill_ignores_extra_values():
    skel = make_skeleton(SINGLE)
    extra = make_values(SINGLE) + "\n| 9 | 9 | 9 |"
    assert _cells(fill_skeleton(skel, extra))[0][1] == ["33", "7.5", "3.7"]


def test_parse_values_drops_separators():
    vals = "|---|---|\n| 1 | 2 |"
    assert parse_values(vals) == [[["1", "2"]]]


def test_fusion_user_text_contains_both_parts():
    text = build_fusion_user_text("| a |\n|---|\n| ? |", "| 1 |")
    assert "| ? |" in text and "| 1 |" in text


def test_text_only_conversation_has_no_image_part():
    msgs = format_qwen_vl_conversation(image=None, target_table="| a |", user_text="hello")
    user = msgs[1]["content"]
    assert all(part["type"] == "text" for part in user)
    assert user[0]["text"] == "hello"


def test_fusion_dataset_reads_records(tmp_path):
    path = os.path.join(tmp_path, "fusion.jsonl")
    rec = {"id": "x", "skeleton": make_skeleton(SINGLE), "values": make_values(SINGLE), "table": SINGLE}
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    ds = FusionTextDataset(path)
    item = ds[0]
    assert len(ds) == 1 and item["image"] is None and item["table"] == SINGLE
    assert "?" in item["user_text"]


def test_round_trip_on_real_ground_truth():
    """Uses the real train split when available (skipped on machines without the data)."""
    path = "data/processed/train.jsonl"
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f][:200]
    for rec in records:
        gt = rec["table"]
        rebuilt = fill_skeleton(make_skeleton(gt), make_values(gt))
        assert _cells(rebuilt) == _cells(gt), rec["id"]
