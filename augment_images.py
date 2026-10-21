"""Create degraded copies of original Sci-ImageMiner figures with Albumentations."""

import hashlib
import json
import os
from pathlib import Path

# Avoid a network version check when running this offline data preparation tool.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import click
import cv2
import numpy as np
from PIL import Image, ImageOps


def build_pipeline(rotation_limit: float, severity: str) -> A.Compose:
    """Preserve the entire figure while simulating lower quality scans/photos."""
    settings = {
        "mild": ((0.8, 0.95), (0.005, 0.015), (75, 90), 0.08, (3, 3)),
        "moderate": ((0.6, 0.85), (0.01, 0.035), (45, 75), 0.15, (3, 5)),
        "strong": ((0.45, 0.7), (0.02, 0.05), (30, 55), 0.2, (3, 7)),
    }
    scale, noise, quality, brightness, blur = settings[severity]
    return A.Compose(
        [
            A.Affine(
                rotate=(-rotation_limit, rotation_limit),
                fit_output=True,
                border_mode=cv2.BORDER_CONSTANT,
                fill=(255, 255, 255),
                p=1.0,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=brightness, contrast_limit=brightness, p=0.7
            ),
            # Every copy receives at least one degradation, even with rotation disabled.
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=blur, p=1.0),
                    A.GaussNoise(std_range=noise, per_channel=False, p=1.0),
                    A.Downscale(
                        scale_range=scale,
                        interpolation_pair={
                            "downscale": cv2.INTER_AREA,
                            "upscale": cv2.INTER_LINEAR,
                        },
                        p=1.0,
                    ),
                ],
                p=1.0,
            ),
            A.ImageCompression(quality_range=quality, p=1.0),
        ],
        strict=True,
        save_applied_params=True,
    )


def augment_dataset(
    source_dir: str | Path = "data/processed",
    output_dir: str | Path = "data/augmented",
    split: str = "all",
    variants: int = 3,
    seed: int = 42,
    rotation_limit: float = 10.0,
    severity: str = "moderate",
) -> dict:
    """Write new images and annotations without changing or resplitting originals."""
    if variants < 1:
        raise ValueError("variants must be positive")
    if not 0 <= rotation_limit <= 180:
        raise ValueError("rotation_limit must be between 0 and 180 degrees")
    if severity not in {"mild", "moderate", "strong"}:
        raise ValueError("severity must be mild, moderate, or strong")
    if split not in {"train", "val", "test", "all"}:
        raise ValueError("split must be train, val, test, or all")

    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if source_dir.is_relative_to(output_dir) or output_dir.is_relative_to(source_dir):
        raise ValueError("Source and output directories must be separate and not nested")
    # Deliberately no overwrite option: a new run needs a new or empty directory.
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(f"Output directory is not empty: {output_dir}")

    splits = ("train", "val", "test") if split == "all" else (split,)
    selected = {}
    seen_ids = set()
    for split_name in splits:
        records = []
        annotation_path = source_dir / f"{split_name}.jsonl"
        for line_number, line in enumerate(
            annotation_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or any(
                not isinstance(record.get(key), str) or not record[key].strip()
                for key in ("id", "image", "table")
            ):
                raise ValueError(f"Invalid id/image/table in {annotation_path}:{line_number}")
            sample_id = record["id"]
            if sample_id.lower().startswith(("synthetic", "synth_")) or "__aug_" in sample_id:
                raise ValueError(f"Expected original records, found generated sample: {sample_id}")
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate source ID across selected splits: {sample_id}")
            seen_ids.add(sample_id)
            records.append(record)

        selected[split_name] = records
    if not any(selected.values()):
        raise ValueError("No source images selected")

    # Validate every source path before creating output. Resolve symlinks as well.
    for records in selected.values():
        for record in records:
            image_path = (source_dir / record["image"]).resolve()
            if not image_path.is_relative_to(source_dir):
                raise ValueError(f"Source image must be inside source_dir: {image_path}")
            if not image_path.is_file():
                raise FileNotFoundError(f"Source image not found: {image_path}")

    pipeline = build_pipeline(rotation_limit, severity)
    manifest = {
        "source_dir": str(source_dir),
        "selection": "all_images",
        "variants_per_image": variants,
        "seed": seed,
        "rotation_limit": rotation_limit,
        "severity": severity,
        "albumentations_version": A.__version__,
        "pipeline": A.to_dict(pipeline),
        "splits": {},
        "images": [],
    }
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    for split_name, records in selected.items():
        generated = []
        for record in records:
            with Image.open(source_dir / record["image"]) as original:
                image = np.array(ImageOps.exif_transpose(original).convert("RGB"))
            for variant in range(1, variants + 1):
                identity = f"{seed}:{split_name}:{record['id']}:{variant}"
                digest = hashlib.sha256(identity.encode()).hexdigest()
                variant_seed = int(digest[:16], 16)
                pipeline.set_random_seed(variant_seed)
                result = pipeline(image=image)
                relative_path = f"images/{digest[:24]}__aug_{variant:03d}.png"
                # Exclusive creation also protects against accidental filename collisions.
                with (output_dir / relative_path).open("xb") as destination:
                    Image.fromarray(result["image"]).save(destination, format="PNG")
                sample_id = f"{record['id']}__aug_{variant:03d}"
                generated.append(
                    {"id": sample_id, "image": relative_path, "table": record["table"]}
                )
                manifest["images"].append(
                    {
                        "id": sample_id,
                        "source_id": record["id"],
                        "source_image": record["image"],
                        "split": split_name,
                        "seed": variant_seed,
                        "transforms": [name for name, _ in result["applied_transforms"]],
                    }
                )
        with (output_dir / f"{split_name}.jsonl").open("x", encoding="utf-8") as annotations:
            for record in generated:
                annotations.write(json.dumps(record, ensure_ascii=False) + "\n")
        manifest["splits"][split_name] = {"sources": len(records), "generated": len(generated)}
    with (output_dir / "manifest.json").open("x", encoding="utf-8") as destination:
        json.dump(manifest, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    return manifest


@click.command()
@click.option(
    "--source-dir", default="data/processed",
    type=click.Path(exists=True, file_okay=False, path_type=Path), show_default=True,
)
@click.option(
    "--output-dir", default="data/augmented",
    type=click.Path(path_type=Path), show_default=True,
)
@click.option(
    "--split", default="all",
    type=click.Choice(["train", "val", "test", "all"]), show_default=True,
)
@click.option(
    "--variants", default=3, type=click.IntRange(min=1), show_default=True,
    help="New images per source image.",
)
@click.option("--seed", default=42, type=int, show_default=True)
@click.option(
    "--rotation-limit", default=10.0, type=click.FloatRange(min=0, max=180),
    show_default=True, help="Maximum absolute rotation angle in degrees.",
)
@click.option(
    "--severity", default="moderate",
    type=click.Choice(["mild", "moderate", "strong"]), show_default=True,
)
def main(**kwargs) -> None:
    """Augment all ORIGINAL images, preserving labels and source split membership."""
    try:
        manifest = augment_dataset(**kwargs)
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    for split_name, counts in manifest["splits"].items():
        click.echo(
            f"{split_name}: {counts['sources']} originals -> {counts['generated']} new images"
        )
    click.echo(f"Saved images, JSONL annotations and manifest to {kwargs['output_dir']}")


if __name__ == "__main__":
    main()
