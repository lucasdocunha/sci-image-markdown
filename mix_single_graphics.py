"""Generate a synthetic dataset of two-graphic figures by compositing single-graphic figures.

Takes records with only one chart/table, pairs them together (either horizontally,
vertically, or mixed), composites the images with clean spacing/padding, and
concatenates their Markdown tables with a double newline (preserving the exact
source-of-truth table content).
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import click
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mix_single_graphics")


def count_markdown_tables(table_text: str) -> int:
    """Count the number of distinct Markdown tables in a text block.
    
    A table is recognized by a header separator row (e.g., '|---|---|' or '|:---|---:|').
    """
    if not table_text or not isinstance(table_text, str):
        return 0
    lines = [line.strip() for line in table_text.strip().split("\n") if line.strip()]
    count = 0
    for line in lines:
        if line.startswith("|") and line.endswith("|"):
            inner = line[1:-1].strip()
            parts = inner.split("|")
            if parts and all(re.match(r"^\s*:?-+:?\s*$", p) for p in parts if p.strip()):
                count += 1
    return count


def is_single_graphic_record(record: dict[str, Any]) -> bool:
    """Return True if the record has exactly one table."""
    table_text = record.get("table", record.get("markdown", record.get("target", "")))
    return count_markdown_tables(table_text) == 1


def build_candidate_image_dirs(
    input_file: Path,
    explicit_dirs: Optional[Sequence[Path | str]] = None,
) -> List[Path]:
    """Auto-discover candidate directories where source images might be located."""
    seen: set[Path] = set()
    candidates: List[Path] = []

    def _add(p: Optional[Path | str]) -> None:
        if not p:
            return
        resolved = Path(p).resolve()
        if resolved not in seen:
            seen.add(resolved)
            candidates.append(resolved)

    # 1. User-supplied explicit directories take highest priority
    if explicit_dirs:
        for d in explicit_dirs:
            _add(d)

    # 2. Derive paths relative to input_file (crucial for external storage / clusters)
    input_dir = input_file.resolve().parent
    _add(input_dir)
    _add(input_dir / "images")

    dataset_root = input_dir.parent
    _add(dataset_root)
    _add(dataset_root / "images")
    _add(dataset_root / "raw")
    _add(dataset_root / "raw" / "images")
    _add(dataset_root / "processed")
    _add(dataset_root / "processed" / "images")

    # 3. Local workspace fallbacks
    cwd = Path.cwd().resolve()
    _add(cwd)
    _add(cwd / "data")
    _add(cwd / "data" / "raw")
    _add(cwd / "data" / "raw" / "images")
    _add(cwd / "data" / "processed")
    _add(cwd / "data" / "processed" / "images")

    return candidates


def resolve_image_path(image_rel_or_abs: str, base_dirs: Sequence[Path]) -> Optional[Path]:
    """Resolve an image path against a list of candidate directories."""
    path = Path(image_rel_or_abs)
    if path.is_absolute() and path.exists():
        return path

    for base in base_dirs:
        # Direct join
        candidate = base / path
        if candidate.exists():
            return candidate
        # Subdirectory 'images' with basename
        basename_cand = base / "images" / path.name
        if basename_cand.exists():
            return basename_cand
        # Direct basename under base
        direct_cand = base / path.name
        if direct_cand.exists():
            return direct_cand
        # Sibling 'raw' directory checks
        raw_cand = base / "raw" / path
        if raw_cand.exists():
            return raw_cand
        raw_basename_cand = base / "raw" / "images" / path.name
        if raw_basename_cand.exists():
            return raw_basename_cand

    return None


def composite_images(
    im1: Image.Image,
    im2: Image.Image,
    layout: str = "horizontal",
    alignment: str = "pad",
    spacing: int = 20,
    background_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Composite two images side-by-side (horizontal) or stacked (vertical).
    
    Args:
        im1: First (left or top) image.
        im2: Second (right or bottom) image.
        layout: 'horizontal' or 'vertical'.
        alignment: 'pad' (canvas padded with background color, preserving exact pixels)
                   or 'scale' (resizes images to matching dimension preserving aspect ratio).
        spacing: Gap in pixels between the two images.
        background_color: RGB background fill (default white).
    """
    im1 = im1.convert("RGB")
    im2 = im2.convert("RGB")

    if layout == "horizontal":
        if alignment == "scale":
            target_h = min(im1.height, im2.height)
            w1 = max(1, round(im1.width * target_h / im1.height))
            w2 = max(1, round(im2.width * target_h / im2.height))
            p1 = im1.resize((w1, target_h), Image.Resampling.LANCZOS)
            p2 = im2.resize((w2, target_h), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (w1 + w2 + spacing, target_h), background_color)
            canvas.paste(p1, (0, 0))
            canvas.paste(p2, (w1 + spacing, 0))
            return canvas
        else:  # pad
            h = max(im1.height, im2.height)
            w = im1.width + im2.width + spacing
            canvas = Image.new("RGB", (w, h), background_color)
            canvas.paste(im1, (0, (h - im1.height) // 2))
            canvas.paste(im2, (im1.width + spacing, (h - im2.height) // 2))
            return canvas

    elif layout == "vertical":
        if alignment == "scale":
            target_w = min(im1.width, im2.width)
            h1 = max(1, round(im1.height * target_w / im1.width))
            h2 = max(1, round(im2.height * target_w / im2.width))
            p1 = im1.resize((target_w, h1), Image.Resampling.LANCZOS)
            p2 = im2.resize((target_w, h2), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (target_w, h1 + h2 + spacing), background_color)
            canvas.paste(p1, (0, 0))
            canvas.paste(p2, (0, h1 + spacing))
            return canvas
        else:  # pad
            w = max(im1.width, im2.width)
            h = im1.height + im2.height + spacing
            canvas = Image.new("RGB", (w, h), background_color)
            canvas.paste(im1, ((w - im1.width) // 2, 0))
            canvas.paste(im2, ((w - im2.width) // 2, im1.height + spacing))
            return canvas

    else:
        raise ValueError(f"Unknown layout: {layout}. Expected 'horizontal' or 'vertical'.")


def combine_markdown_tables(table1: str, table2: str) -> str:
    """Concatenate two Markdown tables separated by double newlines."""
    return f"{table1.strip()}\n\n{table2.strip()}"


def extract_study_key(sample_id: str) -> str:
    """Extract paper/study grouping key from sample ID if available.
    
    Example: sci_imageminer__atomic_layer_deposition__simulation_usecase__32__fig_12
    returns: sci_imageminer__atomic_layer_deposition__simulation_usecase__32
    """
    parts = sample_id.split("__")
    if len(parts) >= 4:
        return "__".join(parts[:-1])
    return sample_id


def generate_pairs(
    records: List[dict[str, Any]],
    num_samples: Optional[int] = None,
    pairing_strategy: str = "random",
    rng: Optional[random.Random] = None,
) -> List[Tuple[dict[str, Any], dict[str, Any]]]:
    """Pair single-graphic records based on the chosen strategy."""
    if rng is None:
        rng = random.Random(42)

    if len(records) < 2:
        raise ValueError(f"Need at least 2 single-graphic records to pair, found {len(records)}.")

    pairs: List[Tuple[dict[str, Any], dict[str, Any]]] = []

    if pairing_strategy == "same-paper":
        # Group by study/paper
        by_study: dict[str, List[dict[str, Any]]] = {}
        for r in records:
            key = extract_study_key(r["id"])
            by_study.setdefault(key, []).append(r)

        intra_pairs: List[Tuple[dict[str, Any], dict[str, Any]]] = []
        for key, group in by_study.items():
            if len(group) >= 2:
                shuffled_group = list(group)
                rng.shuffle(shuffled_group)
                for i in range(0, len(shuffled_group) - 1, 2):
                    intra_pairs.append((shuffled_group[i], shuffled_group[i + 1]))

        rng.shuffle(intra_pairs)
        pairs.extend(intra_pairs)

        # If more needed or not enough intra-paper pairs, fall back to random for remainder
        if num_samples is not None and len(pairs) < num_samples:
            logger.info(
                f"Same-paper strategy found {len(pairs)} pairs; filling remaining "
                f"{num_samples - len(pairs)} pairs randomly."
            )

    # Random pairing (or filling remaining)
    if num_samples is None:
        # Default: pair all available records without replacement
        if pairing_strategy != "same-paper":
            shuffled = list(records)
            rng.shuffle(shuffled)
            for i in range(0, len(shuffled) - 1, 2):
                pairs.append((shuffled[i], shuffled[i + 1]))
    else:
        # User requested a specific number of samples
        while len(pairs) < num_samples:
            a, b = rng.sample(records, 2)
            if a["id"] != b["id"]:
                pairs.append((a, b))

    if num_samples is not None and len(pairs) > num_samples:
        pairs = pairs[:num_samples]

    return pairs


def create_mixed_dataset(
    input_file: Path | str,
    output_dir: Path | str,
    image_dirs: Optional[Sequence[Path | str]] = None,
    num_samples: Optional[int] = None,
    layout: str = "random",
    alignment: str = "pad",
    spacing: int = 20,
    pairing_strategy: str = "random",
    seed: int = 42,
    overwrite: bool = False,
    image_format: str = "JPEG",
    image_quality: int = 95,
) -> dict[str, Any]:
    """Filter single-graphic images, create 2-graphic composites, and write JSONL + images."""
    input_file = Path(input_file).resolve()
    output_dir = Path(output_dir).resolve()

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Build all candidate image directories (auto-discovering relative to input_file)
    base_dirs = build_candidate_image_dirs(input_file, image_dirs)
    existing_dirs = [d for d in base_dirs if d.exists()]
    logger.info(f"Discovered {len(existing_dirs)} available image search directories:")
    for d in existing_dirs:
        logger.info(f"  -> {d}")

    if output_dir.exists() and not overwrite and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory exists and is not empty: {output_dir}. Use --overwrite to replace."
        )

    out_images_dir = output_dir / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    # 1. Read input JSONL and filter single-graphic records
    logger.info(f"Reading records from {input_file}...")
    all_records: List[dict[str, Any]] = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))

    single_records = [r for r in all_records if is_single_graphic_record(r)]
    logger.info(
        f"Found {len(single_records)} single-graphic records out of {len(all_records)} total."
    )

    if len(single_records) < 2:
        raise ValueError(
            f"Found fewer than 2 single-graphic records in {input_file}. Cannot create pairs."
        )

    # 2. Generate pairs
    pairs = generate_pairs(
        single_records,
        num_samples=num_samples,
        pairing_strategy=pairing_strategy,
        rng=rng,
    )
    logger.info(f"Generated {len(pairs)} pairs for compositing.")

    # 3. Composite pairs and build output records
    generated_records: List[dict[str, Any]] = []
    skipped_count = 0

    ext = "jpg" if image_format.upper() in ("JPEG", "JPG") else "png"

    for idx, (rec_a, rec_b) in enumerate(pairs, start=1):
        path_a = resolve_image_path(rec_a["image"], base_dirs)
        path_b = resolve_image_path(rec_b["image"], base_dirs)

        if not path_a or not path_b:
            logger.warning(
                f"Skipping pair ({rec_a['id']}, {rec_b['id']}): could not resolve image paths "
                f"(a={path_a}, b={path_b})."
            )
            skipped_count += 1
            continue

        try:
            with Image.open(path_a) as im_a, Image.open(path_b) as im_b:
                # Determine layout
                current_layout = (
                    rng.choice(["horizontal", "vertical"]) if layout == "random" else layout
                )

                composite = composite_images(
                    im_a,
                    im_b,
                    layout=current_layout,
                    alignment=alignment,
                    spacing=spacing,
                )

                composite_id = f"composite__{rec_a['id']}__x__{rec_b['id']}"
                filename = f"{composite_id}.{ext}"
                out_path = out_images_dir / filename

                if ext == "jpg":
                    composite.save(out_path, format="JPEG", quality=image_quality, subsampling=0)
                else:
                    composite.save(out_path, format="PNG")

                combined_table = combine_markdown_tables(rec_a["table"], rec_b["table"])

                out_record = {
                    "id": composite_id,
                    "image": f"images/{filename}",
                    "table": combined_table,
                    "metadata": {
                        "source_a": rec_a["id"],
                        "source_b": rec_b["id"],
                        "layout": current_layout,
                        "alignment": alignment,
                        "spacing": spacing,
                        "panel_count": 2,
                    },
                }
                generated_records.append(out_record)
        except Exception as e:
            logger.warning(f"Error processing pair ({rec_a['id']}, {rec_b['id']}): {e}")
            skipped_count += 1
            continue

    if len(generated_records) == 0 and len(pairs) > 0:
        dir_list = "\n  ".join(str(d) for d in existing_dirs) if existing_dirs else "  (none found)"
        raise RuntimeError(
            f"Could not locate image files for any of the {len(pairs)} pairs! "
            f"Searched candidate directories:\n  {dir_list}\n\n"
            f"Hint: Provide the path where your source images reside using --image-dir (or -I), "
            f"for example:\n  python mix_single_graphics.py --image-dir /path/to/images ..."
        )

    # 4. Save train.jsonl (or output split name)
    stem = input_file.stem  # e.g. 'train'
    out_jsonl = output_dir / f"{stem}.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in generated_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 5. Save manifest
    layout_counts: dict[str, int] = {}
    for r in generated_records:
        l = r["metadata"]["layout"]
        layout_counts[l] = layout_counts.get(l, 0) + 1

    manifest = {
        "source_input_file": str(input_file),
        "total_source_records": len(all_records),
        "single_graphic_source_records": len(single_records),
        "generated_samples": len(generated_records),
        "skipped_pairs": skipped_count,
        "layout_mode": layout,
        "layout_counts": layout_counts,
        "alignment": alignment,
        "spacing": spacing,
        "pairing_strategy": pairing_strategy,
        "seed": seed,
        "image_format": image_format.upper(),
        "output_jsonl": str(out_jsonl.name),
    }

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    logger.info(
        f"Successfully generated {len(generated_records)} composite figures in {output_dir} "
        f"({layout_counts.get('horizontal', 0)} horizontal, {layout_counts.get('vertical', 0)} vertical)."
    )
    return manifest


@click.command()
@click.option(
    "--input-file",
    default="data/processed/train.jsonl",
    show_default=True,
    help="Path to source JSONL dataset.",
)
@click.option(
    "--output-dir",
    default="data/synthetic_two_graphics",
    show_default=True,
    help="Directory to save the composite figures and JSONL.",
)
@click.option(
    "--image-dir",
    "-I",
    multiple=True,
    default=(),
    help="Optional directory to search for source image files (auto-discovers relative to --input-file if omitted).",
)
@click.option(
    "--num-samples",
    default=None,
    type=int,
    help="Number of 2-graphic samples to generate (defaults to pairing all available singles).",
)
@click.option(
    "--layout",
    default="random",
    show_default=True,
    type=click.Choice(["random", "horizontal", "vertical"]),
    help="Arrangement of the two panels in the composite image.",
)
@click.option(
    "--alignment",
    default="pad",
    show_default=True,
    type=click.Choice(["pad", "scale"]),
    help="Pad with white canvas (preserves original pixels) or scale to matching dimension.",
)
@click.option(
    "--spacing",
    default=20,
    show_default=True,
    type=int,
    help="Pixel margin/gap between the two panels.",
)
@click.option(
    "--pairing",
    default="random",
    show_default=True,
    type=click.Choice(["random", "same-paper", "different-paper"]),
    help="Pairing strategy for single-graphic figures.",
)
@click.option(
    "--seed",
    default=42,
    show_default=True,
    type=int,
    help="Random seed for reproducible dataset generation.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite existing output directory if not empty.",
)
@click.option(
    "--image-format",
    default="JPEG",
    show_default=True,
    type=click.Choice(["JPEG", "PNG"], case_sensitive=False),
    help="Image output format.",
)
def main(
    input_file: str,
    output_dir: str,
    image_dir: tuple[str, ...],
    num_samples: Optional[int],
    layout: str,
    alignment: str,
    spacing: int,
    pairing: str,
    seed: int,
    overwrite: bool,
    image_format: str,
) -> None:
    """Generate a synthetic dataset of two-graphic figures from single-graphic figures."""
    try:
        manifest = create_mixed_dataset(
            input_file=input_file,
            output_dir=output_dir,
            image_dirs=list(image_dir),
            num_samples=num_samples,
            layout=layout,
            alignment=alignment,
            spacing=spacing,
            pairing_strategy=pairing,
            seed=seed,
            overwrite=overwrite,
            image_format=image_format,
        )
        click.echo(
            f"Successfully created dataset with {manifest['generated_samples']} composite samples "
            f"at '{output_dir}'."
        )
    except Exception as e:
        logger.error(str(e))
        raise click.ClickException(str(e)) from e


if __name__ == "__main__":
    main()
