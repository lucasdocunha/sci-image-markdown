"""
CLI tool for preparing and structuring Sci-Image dataset splits.
"""

import os
import json
import random
from typing import Optional
import click
from PIL import Image, ImageDraw, ImageFont

from src.utils.logging import setup_logger
from src.utils.io import write_jsonl

logger = setup_logger("prepare_data")


def create_synthetic_demo_data(output_dir: str, num_samples: int = 10):
    """Generates a small synthetic toy dataset for verification and baseline testing."""
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    samples = []

    for i in range(num_samples):
        # Create a simple synthetic figure image
        img_name = f"synth_plot_{i:03d}.png"
        img_path = os.path.join(output_dir, "images", img_name)

        img = Image.new("RGB", (600, 400), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Draw axes
        draw.line([(80, 50), (80, 320)], fill=(0, 0, 0), width=2)
        draw.line([(80, 320), (550, 320)], fill=(0, 0, 0), width=2)

        # Generate some synthetic data points
        x_vals = [round(100.0 + j * 50.0, 1) for j in range(5)]
        y_vals = [round(0.5 + random.random() * 2.5, 2) for _ in range(5)]

        # Draw plot line
        points = []
        for x, y in zip(x_vals, y_vals):
            px = 80 + int((x - 100) * 1.5)
            py = 320 - int(y * 80)
            points.append((px, py))
            draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=(220, 50, 50))

        if len(points) > 1:
            draw.line(points, fill=(50, 50, 200), width=2)

        img.save(img_path)

        # Create corresponding ground truth markdown table
        markdown_table = (
            "| Temperature (°C) | Growth Rate (Å/cycle) |\n"
            "|---|---|\n"
            + "\n".join([f"| {x:.1f} | {y:.2f} |" for x, y in zip(x_vals, y_vals)])
        )

        samples.append({
            "id": f"synth_{i:03d}",
            "image": os.path.join("images", img_name),
            "table": markdown_table,
            "metadata": {
                "chart_type": "line_plot",
                "domain": "ALD_temperature_window"
            }
        })

    # Split into train/val/test
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


@click.command()
@click.option("--raw-dir", default="data/raw", help="Path to raw dataset directory.")
@click.option("--processed-dir", default="data/processed", help="Path to processed output directory.")
@click.option("--create-synthetic", is_flag=True, help="Create a toy synthetic dataset for testing pipelines.")
@click.option("--num-samples", default=10, help="Number of synthetic samples to generate if --create-synthetic is set.")
def main(raw_dir: str, processed_dir: str, create_synthetic: bool, num_samples: int):
    """Prepares Sci-Image dataset splits for training and evaluation."""
    os.makedirs(processed_dir, exist_ok=True)

    if create_synthetic:
        logger.info(f"Generating {num_samples} synthetic samples...")
        create_synthetic_demo_data(processed_dir, num_samples=num_samples)
        return

    logger.info(f"Processing raw data from: {raw_dir}")
    # Process standard Sci-ImageMiner JSON/JSONL format if present
    raw_train = os.path.join(raw_dir, "train.jsonl")
    if os.path.exists(raw_train):
        logger.info(f"Found raw train annotations in {raw_train}")
    else:
        logger.warning(f"Raw data not found in {raw_dir}. Use --create-synthetic to generate dummy data for verification.")


if __name__ == "__main__":
    main()
