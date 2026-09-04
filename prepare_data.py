"""
CLI tool for preparing and structuring Sci-Image dataset splits.
Supports downloading from Hugging Face (SciKnowOrg/Sci-ImageMiner),
processing raw local annotations, or generating synthetic demo data.
"""

import os
import shutil
import json
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Any
import click
from PIL import Image, ImageDraw
from tqdm import tqdm

from src.utils.logging import setup_logger
from src.utils.io import write_jsonl

logger = setup_logger("prepare_data")

HF_DATASET_ID = "SciKnowOrg/Sci-ImageMiner"
HF_BASE_URL = f"https://huggingface.co/datasets/{HF_DATASET_ID}/resolve/main"


def create_synthetic_demo_data(output_dir: str, num_samples: int = 10):
    """Generates a small synthetic toy dataset for verification and baseline testing."""
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    samples = []

    for i in range(num_samples):
        img_name = f"synth_plot_{i:03d}.png"
        img_path = os.path.join(output_dir, "images", img_name)

        img = Image.new("RGB", (600, 400), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Draw axes
        draw.line([(80, 50), (80, 320)], fill=(0, 0, 0), width=2)
        draw.line([(80, 320), (550, 320)], fill=(0, 0, 0), width=2)

        # Generate synthetic data points
        x_vals = [round(100.0 + j * 50.0, 1) for j in range(5)]
        y_vals = [round(0.5 + random.random() * 2.5, 2) for _ in range(5)]

        # Draw plot line & points
        points = []
        for x, y in zip(x_vals, y_vals):
            px = 80 + int((x - 100) * 1.5)
            py = 320 - int(y * 80)
            points.append((px, py))
            draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=(220, 50, 50))

        if len(points) > 1:
            draw.line(points, fill=(50, 50, 200), width=2)

        img.save(img_path)

        markdown_table = (
            "| Temperature (°C) | Growth Rate (Å/cycle) |\n"
            "|---|---|\n"
            + "\n".join([f"| {x:.1f} | {y:.2f} |" for x, y in zip(x_vals, y_vals)])
        )

        samples.append({
            "id": f"synth_{i:03d}",
            "image": os.path.join("images", img_name),
            "table": markdown_table,
        })

    random.seed(42)
    random.shuffle(samples)
    train_split = samples[:int(num_samples * 0.7)]
    val_split = samples[int(num_samples * 0.7):int(num_samples * 0.9)]
    test_split = samples[int(num_samples * 0.9):]

    write_jsonl(train_split, os.path.join(output_dir, "train.jsonl"))
    write_jsonl(val_split, os.path.join(output_dir, "val.jsonl"))
    write_jsonl(test_split, os.path.join(output_dir, "test.jsonl"))

    logger.info(f"Created demo dataset in {output_dir}:")
    logger.info(f"  Train: {len(train_split)} samples")
    logger.info(f"  Val:   {len(val_split)} samples")
    logger.info(f"  Test:  {len(test_split)} samples")


def _download_file(url: str, dest_path: str, timeout: int = 30) -> bool:
    """Downloads a single file from URL with retries."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return True
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_path = dest_path + ".tmp"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(temp_path, "wb") as f:
                f.write(resp.read())
            os.rename(temp_path, dest_path)
            return True
        except Exception as e:
            if attempt == 2:
                logger.warning(f"Failed downloading {url}: {e}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return False
    return False


def download_and_process_hf_dataset(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    max_samples_per_split: Optional[int] = None,
    num_workers: int = 8,
):
    """Downloads and processes the official Sci-ImageMiner Task 2 dataset from Hugging Face."""
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    image_dir = os.path.join(raw_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    splits_map = {
        "train": "train.jsonl",
        "validation": "val.jsonl",
        "test": "test.jsonl",
    }

    for hf_split, out_filename in splits_map.items():
        logger.info(f"--- Processing Split: {hf_split} ---")
        meta_url = f"{HF_BASE_URL}/{hf_split}/metadata.jsonl"
        meta_path = os.path.join(raw_dir, f"{hf_split}_metadata.jsonl")

        logger.info(f"Fetching metadata from: {meta_url}")
        if not _download_file(meta_url, meta_path):
            logger.error(f"Failed to fetch metadata for {hf_split}")
            continue

        raw_records = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw_records.append(json.loads(line))

        # Filter for samples with data table extraction
        table_samples = []
        for rec in raw_records:
            data_ext = rec.get("data_extraction")
            if not data_ext or not isinstance(data_ext, list):
                continue
            tables = [d.get("data", "").strip() for d in data_ext if isinstance(d, dict) and d.get("data")]
            if not tables:
                continue

            combined_table = "\n\n".join(tables)
            file_name = rec.get("file_name", "")
            img_basename = os.path.basename(file_name)
            img_dest = os.path.join(image_dir, img_basename)
            img_url = f"{HF_BASE_URL}/{hf_split}/{file_name}"

            table_samples.append({
                "id": rec.get("id", img_basename),
                "image": os.path.join("images", img_basename),
                "table": combined_table,
                "img_url": img_url,
                "img_dest": img_dest,
            })

        if max_samples_per_split:
            table_samples = table_samples[:max_samples_per_split]

        logger.info(f"Split {hf_split}: found {len(table_samples)} data table extraction samples. Downloading images...")

        # Download images concurrently
        download_tasks = [(s["img_url"], s["img_dest"]) for s in table_samples]
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_download_file, url, dest): dest for url, dest in download_tasks}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Downloading {hf_split} images"):
                fut.result()

        # Build clean processed jsonl records
        processed_records = []
        for s in table_samples:
            if os.path.exists(s["img_dest"]):
                proc_img_dest = os.path.join(processed_dir, "images", os.path.basename(s["img_dest"]))
                os.makedirs(os.path.dirname(proc_img_dest), exist_ok=True)
                if not os.path.exists(proc_img_dest):
                    try:
                        os.link(s["img_dest"], proc_img_dest)
                    except OSError:
                        shutil.copyfile(s["img_dest"], proc_img_dest)
                processed_records.append({
                    "id": s["id"],
                    "image": s["image"],
                    "table": s["table"],
                })

        out_path = os.path.join(processed_dir, out_filename)
        write_jsonl(processed_records, out_path)
        logger.info(f"Saved {len(processed_records)} samples to {out_path}")

    logger.info("Dataset preparation complete!")


@click.command()
@click.option("--raw-dir", default="data/raw", help="Path to raw dataset directory.")
@click.option("--processed-dir", default="data/processed", help="Path to processed output directory.")
@click.option("--create-synthetic", is_flag=True, help="Create a toy synthetic dataset for testing pipelines.")
@click.option("--num-samples", default=10, help="Number of synthetic samples to generate if --create-synthetic is set.")
@click.option("--download-hf", is_flag=True, default=False, help="Download official Sci-ImageMiner dataset from Hugging Face.")
@click.option("--max-samples", default=None, type=int, help="Optional max samples per split when downloading from Hugging Face.")
@click.option("--num-workers", default=16, type=int, help="Number of concurrent download threads.")
def main(
    raw_dir: str,
    processed_dir: str,
    create_synthetic: bool,
    num_samples: int,
    download_hf: bool,
    max_samples: Optional[int],
    num_workers: int,
):
    """Prepares Sci-Image dataset splits for training and evaluation."""
    os.makedirs(processed_dir, exist_ok=True)

    if create_synthetic:
        logger.info(f"Generating {num_samples} synthetic samples...")
        create_synthetic_demo_data(processed_dir, num_samples=num_samples)
        return

    if download_hf or not os.path.exists(os.path.join(processed_dir, "train.jsonl")):
        logger.info("Downloading and processing official Sci-ImageMiner dataset from Hugging Face...")
        download_and_process_hf_dataset(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            max_samples_per_split=max_samples,
            num_workers=num_workers,
        )
        return

    logger.info(f"Processed dataset already exists in {processed_dir}.")


if __name__ == "__main__":
    main()
