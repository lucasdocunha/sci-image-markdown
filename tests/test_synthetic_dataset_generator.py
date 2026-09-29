"""Tests for the Sci-ImageMiner-like synthetic dataset generator."""

import json

from PIL import Image

from generate_synthetic_dataset import TEMPLATES, generate_dataset, split_records
from src.metrics.table_parser import parse_markdown_to_dataframe


def test_generator_creates_parseable_easy_medium_and_hard_examples(tmp_path):
    generate_dataset(tmp_path, num_samples=9, difficulty="all", seed=7)

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


def test_split_records_uses_official_dataset_proportions():
    records = [{"id": str(index)} for index in range(1200)]

    splits = split_records(records)

    assert {name: len(items) for name, items in splits.items()} == {
        "train": 723,
        "val": 104,
        "test": 373,
    }
