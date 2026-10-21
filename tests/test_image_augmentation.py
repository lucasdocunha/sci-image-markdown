"""Verify augmentation provenance, source preservation, and dataset boundaries."""

import hashlib
import json

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from augment_images import augment_dataset, main


@pytest.fixture
def originals(tmp_path):
    root = tmp_path / "original"
    (root / "images").mkdir(parents=True)
    pixels = np.random.default_rng(7).integers(0, 256, (64, 96, 3), dtype=np.uint8)
    for split in ("train", "val", "test"):
        records = []
        for index in range(3):
            sample_id = f"sci_imageminer__{split}_{index}"
            image = f"images/{sample_id}.png"
            Image.fromarray(pixels).save(root / image)
            table = "| x | y |\n|---|---|\n" + "| 1 | 2 |\n" * (index + 1)
            records.append({"id": sample_id, "image": image, "table": table})
        (root / f"{split}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
        )
    return root


def read_records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_variants_keep_targets_and_sources_and_repeat_exactly(originals, tmp_path):
    before = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in originals.rglob("*") if p.is_file()
    }
    first, second = tmp_path / "first", tmp_path / "second"
    manifest = augment_dataset(originals, first, split="train", variants=3, seed=9)
    augment_dataset(originals, second, split="train", variants=3, seed=9)

    sources = {r["id"]: r for r in read_records(originals / "train.jsonl")}
    records = read_records(first / "train.jsonl")
    assert len(records) == len({r["id"] for r in records}) == 9
    assert manifest["splits"] == {"train": {"sources": 3, "generated": 9}}
    assert not (first / "val.jsonl").exists()
    assert not (first / "test.jsonl").exists()
    hashes = set()
    for record in records:
        assert set(record) == {"id", "image", "table"}
        assert record["table"] == sources[record["id"].rsplit("__aug_", 1)[0]]["table"]
        assert (first / record["image"]).read_bytes() == (second / record["image"]).read_bytes()
        with Image.open(first / record["image"]) as image:
            assert image.mode == "RGB"
            assert image.width >= 96 and image.height >= 64
        hashes.add(hashlib.sha256((first / record["image"]).read_bytes()).hexdigest())
    assert len(hashes) == 9
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}
    assert {entry["source_id"] for entry in manifest["images"]} == set(sources)


def test_default_processes_every_image_and_preserves_splits(originals, tmp_path):
    output = tmp_path / "augmented"
    manifest = augment_dataset(originals, output, variants=1)
    assert manifest["selection"] == "all_images"
    for split in ("train", "val", "test"):
        sources = read_records(originals / f"{split}.jsonl")
        records = read_records(output / f"{split}.jsonl")
        assert len(records) == len(sources) == 3
        assert manifest["splits"][split] == {"sources": 3, "generated": 3}
        for record, source in zip(records, sources):
            assert record["id"] == f"{source['id']}__aug_001"
            assert record["table"] == source["table"]


@pytest.mark.parametrize("severity", ["mild", "moderate", "strong"])
def test_all_severities_degrade_even_without_rotation(originals, tmp_path, severity):
    output = tmp_path / severity
    manifest = augment_dataset(originals, output, variants=1, rotation_limit=0, severity=severity)
    record = read_records(output / "train.jsonl")[0]
    with Image.open(output / record["image"]) as augmented:
        with Image.open(originals / manifest["images"][0]["source_image"]) as source:
            assert not np.array_equal(np.array(augmented), np.array(source))


@pytest.mark.parametrize("destination", ["same", "child", "parent", "existing"])
def test_refuses_unsafe_output_locations(originals, tmp_path, destination):
    paths = {
        "same": originals,
        "child": originals / "augmented",
        "parent": tmp_path,
        "existing": tmp_path / "existing",
    }
    output = paths[destination]
    if destination == "existing":
        output.mkdir()
        (output / "keep.txt").write_text("keep")
    with pytest.raises(ValueError, match="separate|not empty"):
        augment_dataset(originals, output)
    if destination == "existing":
        assert (output / "keep.txt").read_text() == "keep"


@pytest.mark.parametrize("bad_id", ["synthetic__plot", "synth_001", "original__aug_001"])
def test_rejects_generated_sources(originals, tmp_path, bad_id):
    records = read_records(originals / "train.jsonl")
    records[0]["id"] = bad_id
    (originals / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="Expected original"):
        augment_dataset(originals, output)
    assert not output.exists()


def test_cli_processes_all_splits_by_default(originals, tmp_path):
    output = tmp_path / "output"
    result = CliRunner().invoke(
        main,
        ["--source-dir", str(originals), "--output-dir", str(output), "--variants", "1"],
    )
    assert result.exit_code == 0, result.output
    for split in ("train", "val", "test"):
        assert f"{split}: 3 originals -> 3 new images" in result.output
        assert len(read_records(output / f"{split}.jsonl")) == 3


def test_source_path_cannot_escape_to_other_datasets(originals, tmp_path):
    records = read_records(originals / "train.jsonl")
    records[-1]["image"] = "../synthetic/images/figure.png"
    (originals / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    with pytest.raises(ValueError, match="inside source_dir"):
        augment_dataset(originals, tmp_path / "output")
