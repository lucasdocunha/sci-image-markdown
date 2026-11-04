"""
Composition of synthetic two-table samples from existing Sci-Image records.

Two layouts are supported:
- "side_by_side": figure A on the left, figure B on the right.
- "stacked":      figure A on top, figure B below.

The target markdown is always the two tables stacked, separated by a blank
line (valid GFM: each table keeps its own header and separator row).
"""

import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

LAYOUTS = ("side_by_side", "stacked")
_SEPARATOR_ROW = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
CANVAS_SIZE = 1024  # every composed image is CANVAS_SIZE x CANVAS_SIZE, whatever the layout
BACKGROUND = (255, 255, 255)


def fit_within(img: Image.Image, box_w: int, box_h: int) -> Image.Image:
    """Scale img to fit inside box_w x box_h, keeping aspect ratio. Never crops."""
    scale = min(box_w / img.width, box_h / img.height)
    size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    return img if size == img.size else img.resize(size, Image.LANCZOS)


def compose_images(
    img_a: Image.Image, img_b: Image.Image, layout: str, canvas: int = CANVAS_SIZE
) -> Image.Image:
    """
    Place two figures on a fixed square canvas (white background).

    side_by_side: A in the left half, B in the right half.
    stacked:      A in the top half, B in the bottom half.
    Each figure is fitted into its half and centred, so the output size is the same for every sample.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"Unknown layout: {layout!r}. Expected one of {LAYOUTS}.")

    canvas_img = Image.new("RGB", (canvas, canvas), BACKGROUND)
    half = canvas // 2
    if layout == "side_by_side":
        cells = [(0, 0, half, canvas), (half, 0, half, canvas)]
    else:
        cells = [(0, 0, canvas, half), (0, half, canvas, half)]

    for img, (x0, y0, w, h) in zip((img_a, img_b), cells):
        fitted = fit_within(img.convert("RGB"), w, h)
        x = x0 + (w - fitted.width) // 2
        y = y0 + (h - fitted.height) // 2
        canvas_img.paste(fitted, (x, y))
    return canvas_img


def count_markdown_tables(table: str) -> int:
    """Count GFM tables by their separator rows (one per table)."""
    return sum(1 for line in table.splitlines() if _SEPARATOR_ROW.match(line))


def compose_table_markdown(table_a: str, table_b: str) -> str:
    """Join two markdown tables with a blank line (table A first)."""
    return table_a.strip() + "\n\n" + table_b.strip()


def sample_pairs(
    indices: List[int], num_pairs: int, seed: int
) -> List[Tuple[int, int]]:
    """Sample distinct ordered pairs of record indices (never pairs a record with itself)."""
    if len(indices) < 2:
        raise ValueError("Need at least two records to build pairs.")
    rng = random.Random(seed)
    pairs: List[Tuple[int, int]] = []
    seen = set()
    max_unique = len(indices) * (len(indices) - 1)
    target = min(num_pairs, max_unique)
    while len(pairs) < target:
        a, b = rng.sample(indices, 2)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        pairs.append((a, b))
    return pairs


def resolve_image(record_image: str, source_dir: str) -> str:
    """Resolve a record's image path relative to the processed-dataset directory."""
    if os.path.isabs(record_image):
        return record_image
    candidate = os.path.join(source_dir, record_image)
    if os.path.exists(candidate):
        return candidate
    fallback = os.path.join(source_dir, "images", os.path.basename(record_image))
    return fallback


def generate_pairs(
    records: List[Dict[str, Any]],
    source_dir: str,
    output_dir: str,
    layout: str,
    num_pairs: int,
    seed: int,
    image_dir_name: str = "images",
) -> List[Dict[str, Any]]:
    """Create composed images on disk and return the JSONL records."""
    out_img_dir = os.path.join(output_dir, image_dir_name)
    os.makedirs(out_img_dir, exist_ok=True)

    # Only single-table sources, so each composed sample holds exactly two tables.
    usable = [
        i for i, r in enumerate(records)
        if r.get("image") and count_markdown_tables(r.get("table", "")) == 1
    ]
    out: List[Dict[str, Any]] = []
    for n, (idx_a, idx_b) in enumerate(sample_pairs(usable, num_pairs, seed)):
        rec_a, rec_b = records[idx_a], records[idx_b]
        img_a = Image.open(resolve_image(rec_a["image"], source_dir))
        img_b = Image.open(resolve_image(rec_b["image"], source_dir))
        composed = compose_images(img_a, img_b, layout)

        pair_id = f"synth_{layout}_{seed}_{n:05d}"
        rel_path = f"{image_dir_name}/{pair_id}.jpg"
        composed.save(os.path.join(output_dir, rel_path), quality=95)
        out.append({
            "id": pair_id,
            "image": rel_path,
            "table": compose_table_markdown(rec_a["table"], rec_b["table"]),
            "source_ids": [rec_a["id"], rec_b["id"]],
        })
    return out


def mix_into_train(
    base_records: List[Dict[str, Any]],
    base_dir: str,
    synthetic_sets: List[Tuple[List[Dict[str, Any]], str]],
    ratio: float,
    output_dir: str,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Append ~ratio * len(base) synthetic records to the base training records.

    The extra records are split evenly across the synthetic sets (e.g. one per layout)
    and sampled with a fixed seed. Image paths are rewritten relative to output_dir,
    so the mixed JSONL resolves with SciImageTableDataset from its own directory.
    """
    if not synthetic_sets:
        raise ValueError("At least one synthetic set is required.")

    n_extra = round(ratio * len(base_records))
    per_set = n_extra // len(synthetic_sets)
    remainder = n_extra - per_set * len(synthetic_sets)

    mixed: List[Dict[str, Any]] = []
    for rec in base_records:
        mixed.append({**rec, "image": os.path.relpath(
            os.path.join(base_dir, rec["image"]), output_dir)})

    for k, (records, src_dir) in enumerate(synthetic_sets):
        take = per_set + (1 if k < remainder else 0)
        if take > len(records):
            raise ValueError(f"Synthetic set {k} has {len(records)} records, need {take}.")
        for rec in random.Random(seed + k).sample(records, take):
            mixed.append({**rec, "image": os.path.relpath(
                os.path.join(src_dir, rec["image"]), output_dir)})

    random.Random(seed).shuffle(mixed)
    return mixed
