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

# Each factor gets harder from mild -> strong. Rotation comes from the CLI flag.
_SEVERITY_SETTINGS = {
    "mild": {
        "stretch": (0.9, 1.05),
        "gray_p": 0.0,
        "hsv_limit": (5, 10, 10),
        "gamma": (75, 125),
        "gamma_p": 0.3,
        "brightness": 0.08,
        "perspective": None,
        "shadow_p": 0.0,
        "blur": (3, 3),
        "gauss_std": (0.02, 0.05),
        "iso": ((0.01, 0.03), (0.2, 0.4)),
        "downscale": (0.8, 0.95),
        "sharpen_p": 0.0,
        "quality": (75, 90),
    },
    "moderate": {
        "stretch": (0.85, 1.15),
        "gray_p": 0.15,
        "hsv_limit": (10, 20, 20),
        "gamma": (60, 140),
        "gamma_p": 0.5,
        "brightness": 0.15,
        "perspective": (0.03, 0.07),
        "perspective_p": 0.15,
        "shadow_p": 0.15,
        "blur": (3, 5),
        "gauss_std": (0.04, 0.09),
        "iso": ((0.01, 0.05), (0.3, 0.7)),
        "downscale": (0.65, 0.9),
        "sharpen_p": 0.15,
        "quality": (45, 75),
    },
    "strong": {
        "stretch": (0.8, 1.25),
        "gray_p": 0.3,
        "hsv_limit": (15, 30, 30),
        "gamma": (50, 150),
        "gamma_p": 0.7,
        "brightness": 0.2,
        "perspective": (0.05, 0.1),
        "perspective_p": 0.3,
        "shadow_p": 0.25,
        "blur": (3, 7),
        "gauss_std": (0.06, 0.14),
        "iso": ((0.02, 0.08), (0.5, 1.0)),
        "downscale": (0.55, 0.85),
        "sharpen_p": 0.25,
        "quality": (30, 55),
    },
}

_WHITE = (255, 255, 255)


def _affine(settings: dict, rotation_limit: float) -> A.Affine:
    """Rotation plus independent per-axis stretch, always with a white border fit."""
    stretch = settings["stretch"]
    return A.Affine(
        rotate=(-rotation_limit, rotation_limit),
        scale={"x": stretch, "y": stretch},
        fit_output=True,
        border_mode=cv2.BORDER_CONSTANT,
        fill=_WHITE,
        p=1.0,
    )


def build_pipeline(rotation_limit: float, severity: str) -> A.Compose:
    """Simulate a realistic photo/scan of a printed figure, accumulated in one pass."""
    s = _SEVERITY_SETTINGS[severity]
    transforms = [_affine(s, rotation_limit)]
    if s["perspective"]:
        transforms.append(
            A.Perspective(
                scale=s["perspective"],
                keep_size=True,
                fit_output=True,
                border_mode=cv2.BORDER_CONSTANT,
                fill=_WHITE,
                p=s["perspective_p"],
            )
        )
    if s["gray_p"] > 0:
        transforms.append(A.ToGray(p=s["gray_p"]))
    transforms.append(A.HueSaturationValue(*s["hsv_limit"], p=0.5))
    transforms.append(A.RandomGamma(*s["gamma"], p=s["gamma_p"]))
    transforms.append(
        A.RandomBrightnessContrast(
            brightness_limit=s["brightness"], contrast_limit=s["brightness"], p=0.7
        )
    )
    if s["shadow_p"] > 0:
        transforms.append(
            A.RandomShadow(
                shadow_roi=(0.0, 0.0, 1.0, 1.0),
                num_shadows_limit=(1, 2),
                shadow_intensity_range=(0.3, 0.6),
                p=s["shadow_p"],
            )
        )
    transforms.append(
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=s["blur"], p=1.0),
                A.GaussNoise(std_range=s["gauss_std"], per_channel=False, p=1.0),
                A.ISONoise(color_shift=s["iso"][0], intensity=s["iso"][1], p=1.0),
                A.Downscale(
                    scale_range=s["downscale"],
                    interpolation_pair={
                        "downscale": cv2.INTER_AREA,
                        "upscale": cv2.INTER_LINEAR,
                    },
                    p=1.0,
                ),
            ],
            # Every copy receives at least one local degradation.
            p=1.0,
        )
    )
    if s["sharpen_p"] > 0:
        transforms.append(A.Sharpen(p=s["sharpen_p"]))
    transforms.append(A.ImageCompression(quality_range=s["quality"], p=1.0))
    return A.Compose(transforms, strict=True, save_applied_params=True)


# Diagnostic presets: one isolated transformation (or none) per output set.
DIAGNOSTIC_SETS = (
    "baseline",
    "rotate",
    "stretch",
    "grayscale",
    "hue_gamma",
    "perspective",
    "shadow",
    "blur",
    "gauss_noise",
    "iso_noise",
    "downscale",
    "sharpen",
    "jpeg",
)


def build_diagnostic_pipeline(name: str, rotation_limit: float, severity: str):
    """Return a pipeline applying only the requested transform (p=1), or None for baseline."""
    s = _SEVERITY_SETTINGS[severity]

    def compose(transform: A.BasicTransform) -> A.Compose:
        return A.Compose([transform], strict=True, save_applied_params=True)

    if name == "baseline":
        return None
    if name == "rotate":
        return compose(_affine(s, rotation_limit))
    if name == "stretch":
        stretch = s["stretch"]
        return compose(
            A.Affine(
                rotate=0,
                scale={"x": stretch, "y": stretch},
                fit_output=True,
                border_mode=cv2.BORDER_CONSTANT,
                fill=_WHITE,
                p=1.0,
            )
        )
    if name == "grayscale":
        return compose(A.ToGray(p=1.0))
    if name == "hue_gamma":
        return compose(
            A.Compose(
                [A.HueSaturationValue(*s["hsv_limit"], p=1.0), A.RandomGamma(*s["gamma"], p=1.0)],
                strict=True,
            )
        )
    if name == "perspective":
        return compose(
            A.Perspective(
                scale=s["perspective"] or (0.05, 0.1),
                keep_size=True,
                fit_output=True,
                border_mode=cv2.BORDER_CONSTANT,
                fill=_WHITE,
                p=1.0,
            )
        )
    if name == "shadow":
        return compose(
            A.RandomShadow(
                shadow_roi=(0.0, 0.0, 1.0, 1.0),
                num_shadows_limit=(1, 2),
                shadow_intensity_range=(0.3, 0.6),
                p=1.0,
            )
        )
    if name == "blur":
        return compose(A.GaussianBlur(blur_limit=s["blur"], p=1.0))
    if name == "gauss_noise":
        return compose(A.GaussNoise(std_range=s["gauss_std"], per_channel=False, p=1.0))
    if name == "iso_noise":
        return compose(A.ISONoise(color_shift=s["iso"][0], intensity=s["iso"][1], p=1.0))
    if name == "downscale":
        return compose(
            A.Downscale(
                scale_range=s["downscale"],
                interpolation_pair={
                    "downscale": cv2.INTER_AREA,
                    "upscale": cv2.INTER_LINEAR,
                },
                p=1.0,
            )
        )
    if name == "sharpen":
        return compose(A.Sharpen(p=1.0))
    if name == "jpeg":
        return compose(A.ImageCompression(quality_range=s["quality"], p=1.0))
    raise ValueError(f"Unknown diagnostic set: {name}")


class _NameRegistry:
    """Readable, collision-free PNG names tied to the original figure stem."""

    def __init__(self, selected: dict[str, list[dict]]):
        stem_counts: dict[str, int] = {}
        stem_to_splits: dict[str, set[str]] = {}
        for split_name, records in selected.items():
            for record in records:
                stem = Path(record["image"]).stem
                stem_counts[stem] = stem_counts.get(stem, 0) + 1
                stem_to_splits.setdefault(stem, set()).add(split_name)
        self._duplicated = {
            stem for stem, count in stem_counts.items() if count > 1
        }
        self._splits = stem_to_splits

    def filename(self, record: dict, split_name: str, variant: int) -> str:
        stem = Path(record["image"]).stem
        if stem in self._duplicated:
            identity = f"{split_name}:{record['id']}"
            tag = hashlib.sha256(identity.encode()).hexdigest()[:6]
            stem = f"{stem}_d{tag}"
        return f"{stem}__aug_{variant:03d}.png"


def _synthesize_set(
    source_dir: Path,
    output_dir: Path,
    selected: dict[str, list[dict]],
    pipeline_builder,
    variants: int,
    seed: int,
    rotation_limit: float,
    severity: str,
    set_name: str,
    name_registry: _NameRegistry,
) -> dict:
    """Generate copies for one augmentation set and write images, JSONLs and manifest."""
    pipeline = pipeline_builder()
    manifest = {
        "set": set_name,
        "source_dir": str(source_dir),
        "selection": "all_images",
        "variants_per_image": variants,
        "seed": seed,
        "rotation_limit": rotation_limit,
        "severity": severity,
        "albumentations_version": A.__version__,
        "pipeline": A.to_dict(pipeline) if pipeline else None,
        "splits": {},
        "images": [],
    }
    (output_dir / "images").mkdir(parents=True)
    for split_name, records in selected.items():
        generated = []
        for record in records:
            with Image.open(source_dir / record["image"]) as original:
                image = np.array(ImageOps.exif_transpose(original).convert("RGB"))
            for variant in range(1, variants + 1):
                identity = f"{set_name}:{seed}:{split_name}:{record['id']}:{variant}"
                digest = hashlib.sha256(identity.encode()).hexdigest()
                variant_seed = int(digest[:16], 16)
                if pipeline:
                    pipeline.set_random_seed(variant_seed)
                    result = pipeline(image=image)
                    transformed = result["image"]
                    applied = [name for name, _ in result["applied_transforms"]]
                else:
                    transformed = image
                    applied = []
                relative_path = f"images/{name_registry.filename(record, split_name, variant)}"
                # Exclusive creation also protects against accidental filename collisions.
                try:
                    with (output_dir / relative_path).open("xb") as destination:
                        Image.fromarray(transformed).save(destination, format="PNG")
                except FileExistsError as exc:
                    raise ValueError(f"Duplicate generated file: {relative_path}") from exc
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
                        "transforms": applied,
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


def augment_dataset(
    source_dir: str | Path = "data/processed",
    output_dir: str | Path = "data/augmented",
    split: str = "all",
    variants: int = 3,
    seed: int = 42,
    rotation_limit: float = 10.0,
    severity: str = "moderate",
    diagnostic: bool = False,
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

    name_registry = _NameRegistry(selected)
    if diagnostic:
        shared = {
            "source_dir": source_dir,
            "selected": selected,
            "seed": seed,
            "rotation_limit": rotation_limit,
            "severity": severity,
            "name_registry": name_registry,
        }
        top_manifest = {
            "mode": "diagnostic",
            "source_dir": str(source_dir),
            "split": split,
            "seed": seed,
            "rotation_limit": rotation_limit,
            "severity": severity,
            "albumentations_version": A.__version__,
            "sets": {},
        }
        for set_name in DIAGNOSTIC_SETS:
            top_manifest["sets"][set_name] = _synthesize_set(
                output_dir=output_dir / set_name,
                pipeline_builder=lambda name=set_name: build_diagnostic_pipeline(
                    name, rotation_limit, severity
                ),
                variants=1,
                set_name=set_name,
                **shared,
            )
        with (output_dir / "manifest.json").open("x", encoding="utf-8") as destination:
            json.dump(top_manifest, destination, ensure_ascii=False, indent=2)
            destination.write("\n")
        return top_manifest
    return _synthesize_set(
        source_dir=source_dir,
        output_dir=output_dir,
        selected=selected,
        pipeline_builder=lambda: build_pipeline(rotation_limit, severity),
        variants=variants,
        seed=seed,
        rotation_limit=rotation_limit,
        severity=severity,
        set_name="standard",
        name_registry=name_registry,
    )


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
@click.option(
    "--diagnostic", is_flag=True, default=False,
    help="Generate one isolated-transform copy per image instead of the full pipeline.",
)
def main(**kwargs) -> None:
    """Augment all ORIGINAL images, preserving labels and source split membership."""
    try:
        manifest = augment_dataset(**kwargs)
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    if manifest.get("mode") == "diagnostic":
        for set_name, sub in manifest["sets"].items():
            for split_name, counts in sub["splits"].items():
                click.echo(
                    f"{set_name}/{split_name}: {counts['sources']} originals -> "
                    f"{counts['generated']} new images"
                )
    else:
        for split_name, counts in manifest["splits"].items():
            click.echo(
                f"{split_name}: {counts['sources']} originals -> {counts['generated']} new images"
            )
    click.echo(f"Saved images, JSONL annotations and manifest to {kwargs['output_dir']}")


if __name__ == "__main__":
    main()
