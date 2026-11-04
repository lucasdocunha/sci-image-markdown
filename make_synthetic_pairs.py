"""
CLI for generating synthetic two-table samples from the processed Sci-Image dataset.

Layouts:
  side_by_side  figure A left, figure B right   (version 1)
  stacked       figure A top,  figure B bottom  (version 2)

Example:
  .venv/bin/python make_synthetic_pairs.py --layout side_by_side --num-pairs 200 --seed 0
"""

import os
import click

from src.data.synthetic import LAYOUTS, generate_pairs
from src.utils.io import read_jsonl, write_jsonl
from src.utils.logging import setup_logger

logger = setup_logger("make_synthetic_pairs")


@click.command()
@click.option("--input-jsonl", default="data/processed/train.jsonl", show_default=True,
              help="Source split (only training data by default, to avoid leaking val/test).")
@click.option("--output-dir", default="data/synthetic", show_default=True)
@click.option("--layout", type=click.Choice(LAYOUTS), default="side_by_side", show_default=True)
@click.option("--num-pairs", default=200, show_default=True)
@click.option("--seed", default=0, show_default=True)
def main(input_jsonl: str, output_dir: str, layout: str, num_pairs: int, seed: int) -> None:
    source_dir = os.path.dirname(os.path.abspath(input_jsonl))
    records = read_jsonl(input_jsonl)
    logger.info(f"Loaded {len(records)} records from {input_jsonl}")

    os.makedirs(output_dir, exist_ok=True)
    pairs = generate_pairs(records, source_dir, output_dir, layout, num_pairs, seed)

    out_path = os.path.join(output_dir, f"pairs_{layout}.jsonl")
    write_jsonl(pairs, out_path)
    logger.info(f"Wrote {len(pairs)} {layout} pairs to {out_path}")


if __name__ == "__main__":
    main()
