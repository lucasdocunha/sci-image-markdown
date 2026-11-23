"""
Tests for synthetic two-table sample generation.
"""

import json
import os

from PIL import Image

from src.data.synthetic import (
    CANVAS_SIZE,
    compose_images,
    compose_table_markdown,
    generate_pairs,
)
from src.data.dataset import SciImageTableDataset
from src.utils.io import write_jsonl


def _make_source(tmp_path):
    src = tmp_path / "processed"
    (src / "images").mkdir(parents=True)
    records = []
    for i, (w, h) in enumerate([(300, 200), (400, 100), (250, 250)]):
        name = f"fig_{i}.jpg"
        Image.new("RGB", (w, h), (i * 60, 0, 0)).save(src / "images" / name)
        records.append({
            "id": f"rec_{i}",
            "image": f"images/{name}",
            "table": f"| X | Y{i} |\n|---|---|\n| 1 | {i} |",
        })
    write_jsonl(records, str(src / "train.jsonl"))
    return src, records


def test_compose_output_size_is_fixed_for_every_layout_and_input():
    inputs = [
        (Image.new("RGB", (300, 200), "red"), Image.new("RGB", (400, 100), "blue")),
        (Image.new("RGB", (2000, 80), "red"), Image.new("RGB", (60, 900), "blue")),
    ]
    for a, b in inputs:
        for layout in ("side_by_side", "stacked"):
            assert compose_images(a, b, layout).size == (CANVAS_SIZE, CANVAS_SIZE)


def test_compose_never_crops_figures():
    # Solid-colour figures: every colour must appear in the output, i.e. nothing is cut off.
    a = Image.new("RGB", (2000, 80), (255, 0, 0))   # very wide
    b = Image.new("RGB", (60, 900), (0, 0, 255))    # very tall
    for layout in ("side_by_side", "stacked"):
        out = compose_images(a, b, layout)
        colors = {c for _, c in out.getcolors(maxcolors=CANVAS_SIZE * CANVAS_SIZE)}
        assert (255, 0, 0) in colors and (0, 0, 255) in colors


def test_table_markdown_has_blank_line_between_tables():
    text = compose_table_markdown("| A |\n|---|\n| 1 |", "| B |\n|---|\n| 2 |")
    assert text == "| A |\n|---|\n| 1 |\n\n| B |\n|---|\n| 2 |"


def test_generate_pairs_and_dataset_loading(tmp_path):
    src, records = _make_source(tmp_path)
    out_dir = tmp_path / "synthetic"
    pairs = generate_pairs(records, str(src), str(out_dir), "stacked", num_pairs=5, seed=1)

    assert len(pairs) == 5
    assert len({p["id"] for p in pairs}) == 5
    for p in pairs:
        a, b = p["source_ids"]
        assert a != b
        assert os.path.exists(out_dir / p["image"])

    jsonl = out_dir / "pairs_stacked.jsonl"
    write_jsonl(pairs, str(jsonl))
    ds = SciImageTableDataset(data_path=str(jsonl))
    assert len(ds) == 5
    sample = ds[0]
    assert isinstance(sample["image"], Image.Image)
    assert sample["table"].count("\n\n") == 1


def test_generate_pairs_is_deterministic(tmp_path):
    src, records = _make_source(tmp_path)
    p1 = generate_pairs(records, str(src), str(tmp_path / "o1"), "side_by_side", 4, seed=7)
    p2 = generate_pairs(records, str(src), str(tmp_path / "o2"), "side_by_side", 4, seed=7)
    assert [p["source_ids"] for p in p1] == [p["source_ids"] for p in p2]


def test_mix_into_train_ratio_and_paths(tmp_path):
    from src.data.synthetic import mix_into_train

    base_dir = tmp_path / "processed"
    syn_a_dir = tmp_path / "synA"
    syn_b_dir = tmp_path / "synB"
    for d in (base_dir, syn_a_dir, syn_b_dir):
        (d / "images").mkdir(parents=True)
        Image.new("RGB", (20, 20)).save(d / "images" / "x.jpg")

    base = [{"id": f"b{i}", "image": "images/x.jpg", "table": "t"} for i in range(100)]
    syn_a = [{"id": f"a{i}", "image": "images/x.jpg", "table": "t"} for i in range(40)]
    syn_b = [{"id": f"s{i}", "image": "images/x.jpg", "table": "t"} for i in range(40)]
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    mixed = mix_into_train(
        base, str(base_dir),
        [(syn_a, str(syn_a_dir)), (syn_b, str(syn_b_dir))],
        ratio=0.15, output_dir=str(out_dir), seed=3,
    )
    assert len(mixed) == 115  # 15 synthetic records = round(0.15 * 100), split across two sets
    extra = [r for r in mixed if r["id"][0] in "as"]
    assert len(extra) == 15

    # Paths resolve from the output directory.
    for r in mixed:
        assert os.path.exists(os.path.join(str(out_dir), r["image"])), r["image"]
