"""Tests for the Sci-ImageMiner-like synthetic dataset generator."""

import json
import random

from PIL import Image

from generate_synthetic_dataset import TEMPLATES, _multi_series, generate_dataset, split_records
from src.metrics.table_parser import parse_markdown_to_dataframe


def _table_rows(table):
    return [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in table.splitlines()]


def test_generator_creates_parseable_easy_medium_and_hard_examples(tmp_path):
    generate_dataset(tmp_path, num_samples=9, difficulty="all", seed=7, multi_series_share=0.0)

    records = []
    for split in ("train", "val", "test"):
        lines = (tmp_path / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        records.extend(json.loads(line) for line in lines)

    assert len(records) == 9
    assert len({record["table"] for record in records}) == 9
    column_counts = {record["table"].splitlines()[0].count("|") - 1 for record in records}
    assert min(column_counts) == 2
    assert max(column_counts) >= 3
    for record in records:
        assert set(record) == {"id", "image", "table"}
        assert (tmp_path / record["image"]).is_file()
        assert record["image"].endswith(".jpg")
        with Image.open(tmp_path / record["image"]) as image:
            assert image.format == "JPEG"
            assert image.mode == "RGB"
            assert image.width >= 600
        frame, valid = parse_markdown_to_dataframe(record["table"])
        assert valid
        assert frame is not None

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["difficulty_counts"] == {"easy": 4, "medium": 4, "hard": 1}
    assert manifest["unique_tables"] == 9


def test_generator_exposes_broad_coherent_template_catalog():
    assert len(TEMPLATES) >= 20
    assert len({template.name for template in TEMPLATES}) == len(TEMPLATES)
    assert {template.subdomain for template in TEMPLATES} == {
        "atomic_layer_deposition",
        "atomic_layer_etching",
    }
    assert all(template.difficulty in {"easy", "medium", "hard"} for template in TEMPLATES)


def test_multi_series_builds_one_wide_table_with_varied_series_and_shared_x():
    series_counts = set()
    headers = set()
    irregular_x = 0
    for seed in range(60):
        image, table = _multi_series(random.Random(seed))
        assert image.width >= 600
        assert "\n\n" not in table
        header, separator, *body = _table_rows(table)
        assert all(cell == "---" for cell in separator)
        series_counts.add(len(header) - 1)
        headers.update(header)
        x_values = [float(row[0]) for row in body]
        assert x_values == sorted(set(x_values))
        assert all(len(row) == len(header) and all(row) for row in body)
        steps = {round(b - a, 9) for a, b in zip(x_values, x_values[1:])}
        irregular_x += len(steps) > 1

    assert series_counts == {2, 3, 4, 5, 6}
    assert len(headers) >= 60
    assert 15 <= irregular_x <= 45


def test_generate_dataset_mixes_in_multi_series_share(tmp_path):
    generate_dataset(tmp_path, num_samples=20, seed=3, multi_series_share=0.5)

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["template_counts"]["multi_series"] == 10
    assert manifest["multi_series_share"] == 0.5


def test_split_records_uses_official_dataset_proportions():
    records = [{"id": str(index)} for index in range(1200)]

    splits = split_records(records)

    assert {name: len(items) for name, items in splits.items()} == {
        "train": 723,
        "val": 104,
        "test": 373,
    }
