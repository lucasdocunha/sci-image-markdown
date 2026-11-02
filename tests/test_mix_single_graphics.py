"""Unit tests for mix_single_graphics.py."""

import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from mix_single_graphics import (
    count_markdown_tables,
    is_single_graphic_record,
    composite_images,
    combine_markdown_tables,
    extract_study_key,
    generate_pairs,
    create_mixed_dataset,
    build_candidate_image_dirs,
)


def test_build_candidate_image_dirs(tmp_path):
    input_file = tmp_path / "dataset" / "processed" / "train.jsonl"
    candidates = build_candidate_image_dirs(input_file, explicit_dirs=[tmp_path / "custom"])
    assert (tmp_path / "custom") in candidates
    assert (tmp_path / "dataset" / "processed") in candidates
    assert (tmp_path / "dataset" / "raw") in candidates
    assert (tmp_path / "dataset" / "raw" / "images") in candidates


def test_count_markdown_tables():
    single_tbl = (
        "| Col1 | Col2 |\n"
        "|---|---|\n"
        "| 1 | 2 |\n"
        "| 3 | 4 |"
    )
    assert count_markdown_tables(single_tbl) == 1

    two_tbls = (
        "| Col1 | Col2 |\n"
        "|:---|---:|\n"
        "| 1 | 2 |\n\n"
        "| ColA | ColB |\n"
        "|---|---|\n"
        "| a | b |"
    )
    assert count_markdown_tables(two_tbls) == 2

    assert count_markdown_tables("") == 0
    assert count_markdown_tables("Just plain text with no tables") == 0


def test_is_single_graphic_record():
    rec_single = {"id": "1", "table": "| A | B |\n|---|---|\n| 1 | 2 |"}
    rec_multi = {"id": "2", "table": "| A | B |\n|---|---|\n| 1 | 2 |\n\n| C | D |\n|---|---|\n| 3 | 4 |"}
    assert is_single_graphic_record(rec_single) is True
    assert is_single_graphic_record(rec_multi) is False


def test_composite_images_horizontal_pad():
    im1 = Image.new("RGB", (100, 80), (255, 0, 0))
    im2 = Image.new("RGB", (120, 90), (0, 255, 0))
    spacing = 15

    res = composite_images(im1, im2, layout="horizontal", alignment="pad", spacing=spacing)
    assert res.mode == "RGB"
    assert res.size == (100 + 120 + 15, max(80, 90))


def test_composite_images_horizontal_scale():
    im1 = Image.new("RGB", (100, 80), (255, 0, 0))
    im2 = Image.new("RGB", (120, 100), (0, 255, 0))
    spacing = 10

    res = composite_images(im1, im2, layout="horizontal", alignment="scale", spacing=spacing)
    assert res.mode == "RGB"
    target_h = min(80, 100)
    assert res.height == target_h


def test_composite_images_vertical_pad():
    im1 = Image.new("RGB", (100, 80), (255, 0, 0))
    im2 = Image.new("RGB", (120, 90), (0, 255, 0))
    spacing = 10

    res = composite_images(im1, im2, layout="vertical", alignment="pad", spacing=spacing)
    assert res.mode == "RGB"
    assert res.size == (max(100, 120), 80 + 90 + 10)


def test_composite_images_vertical_scale():
    im1 = Image.new("RGB", (100, 80), (255, 0, 0))
    im2 = Image.new("RGB", (120, 90), (0, 255, 0))
    spacing = 10

    res = composite_images(im1, im2, layout="vertical", alignment="scale", spacing=spacing)
    assert res.mode == "RGB"
    target_w = min(100, 120)
    assert res.width == target_w


def test_combine_markdown_tables():
    t1 = "| A | B |\n|---|---|\n| 1 | 2 |"
    t2 = "| C | D |\n|---|---|\n| 3 | 4 |"
    combined = combine_markdown_tables(t1, t2)
    assert combined == f"{t1}\n\n{t2}"
    assert count_markdown_tables(combined) == 2


def test_extract_study_key():
    sid = "sci_imageminer__atomic_layer_deposition__simulation_usecase__32__fig_12"
    assert extract_study_key(sid) == "sci_imageminer__atomic_layer_deposition__simulation_usecase__32"


def test_generate_pairs_without_replacement():
    records = [{"id": f"rec_{i}"} for i in range(6)]
    pairs = generate_pairs(records, num_samples=None)
    assert len(pairs) == 3
    used_ids = {r["id"] for pair in pairs for r in pair}
    assert len(used_ids) == 6


def test_generate_pairs_with_custom_num_samples():
    records = [{"id": f"rec_{i}"} for i in range(4)]
    pairs = generate_pairs(records, num_samples=10)
    assert len(pairs) == 10
    for a, b in pairs:
        assert a["id"] != b["id"]


def test_create_mixed_dataset_end_to_end(tmp_path):
    # Setup mock images
    images_dir = tmp_path / "raw_images"
    images_dir.mkdir()

    records = []
    for i in range(4):
        img_name = f"fig_{i}.jpg"
        img = Image.new("RGB", (100 + i * 10, 80 + i * 5), color=(i * 40, i * 40, i * 40))
        img.save(images_dir / img_name)
        records.append({
            "id": f"study__1__fig_{i}",
            "image": img_name,
            "table": f"| Header_{i} |\n|---|\n| val_{i} |",
        })

    # Add one multi-table record to verify it gets filtered out
    records.append({
        "id": "multi_fig",
        "image": "fig_0.jpg",
        "table": "| H1 |\n|---|\n| 1 |\n\n| H2 |\n|---|\n| 2 |",
    })

    input_jsonl = tmp_path / "train.jsonl"
    with open(input_jsonl, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    output_dir = tmp_path / "output_mixed"

    manifest = create_mixed_dataset(
        input_file=input_jsonl,
        output_dir=output_dir,
        image_dirs=[images_dir],
        num_samples=2,
        layout="horizontal",
        alignment="pad",
        spacing=15,
        pairing_strategy="random",
        seed=42,
    )

    assert manifest["single_graphic_source_records"] == 4
    assert manifest["generated_samples"] == 2
    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "manifest.json").exists()

    # Read output jsonl
    with open(output_dir / "train.jsonl") as f:
        out_recs = [json.loads(l) for l in f]

    assert len(out_recs) == 2
    for r in out_recs:
        assert count_markdown_tables(r["table"]) == 2
        assert (output_dir / r["image"]).exists()
        assert r["metadata"]["layout"] == "horizontal"
        assert r["metadata"]["panel_count"] == 2
