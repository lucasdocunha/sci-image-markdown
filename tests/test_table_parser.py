"""
Unit tests for table parsing and markdown extraction.
"""

import pandas as pd
import pytest
from src.metrics.table_parser import (
    extract_markdown_table_block,
    parse_markdown_to_dataframe,
    dataframe_to_markdown,
)


def test_extract_markdown_table_block_raw():
    text = (
        "Here is the table:\n\n"
        "| Temp | Rate |\n"
        "|---|---|\n"
        "| 150 | 1.25 |\n"
        "| 200 | 1.30 |\n"
        "\nHope this helps!"
    )
    table_block = extract_markdown_table_block(text)
    assert table_block is not None
    assert "| Temp | Rate |" in table_block
    assert "| 200 | 1.30 |" in table_block


def test_extract_markdown_table_block_fenced():
    text = (
        "Sure, here is the result:\n"
        "```markdown\n"
        "| A | B |\n"
        "|---|---|\n"
        "| 1 | 2 |\n"
        "```\n"
    )
    table_block = extract_markdown_table_block(text)
    assert table_block is not None
    assert "| A | B |" in table_block
    assert "| 1 | 2 |" in table_block


def test_parse_markdown_to_dataframe_valid():
    md = (
        "| Dose (s) | Thickness (nm) |\n"
        "|:---|---:|\n"
        "| 0.5 | 10.2 |\n"
        "| 1.0 | 15.6 |\n"
        "| 2.0 | 20.1 |\n"
    )
    df, is_valid = parse_markdown_to_dataframe(md)
    assert is_valid is True
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == ["Dose (s)", "Thickness (nm)"]
    assert df["Dose (s)"].tolist() == [0.5, 1.0, 2.0]
    assert df["Thickness (nm)"].tolist() == [10.2, 15.6, 20.1]


def test_parse_markdown_to_dataframe_invalid():
    invalid_text = "This is just a sentence with no table."
    df, is_valid = parse_markdown_to_dataframe(invalid_text)
    assert is_valid is False
    assert df is None


def test_dataframe_to_markdown_roundtrip():
    df = pd.DataFrame({
        "Cycles": [50, 100, 150],
        "Growth": [2.1, 4.2, 6.3]
    })
    md = dataframe_to_markdown(df)
    assert "Cycles" in md and "Growth" in md
    parsed_df, is_valid = parse_markdown_to_dataframe(md)
    assert is_valid is True
    assert parsed_df["Cycles"].tolist() == [50, 100, 150]
    assert parsed_df["Growth"].tolist() == [2.1, 4.2, 6.3]
