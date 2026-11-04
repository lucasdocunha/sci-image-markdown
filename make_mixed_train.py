"""
Build a training JSONL with ~RATIO extra synthetic two-table samples on top of the real train split.

Example (15% more examples than data/processed/train.jsonl):
  .venv/bin/python make_mixed_train.py --ratio 0.15 \
      --synthetic-file data/synthetic/pairs_side_by_side.jsonl \
      --synthetic-file data/synthetic/pairs_stacked.jsonl
"""

import os
import click

from src.data.synthetic import mix_into_train
from src.utils.io import read_jsonl, write_jsonl
from src.utils.logging import setup_logger

logger = setup_logger("make_mixed_train")


@click.command()
@click.option("--base-file", default="data/processed/train.jsonl", show_default=True)
@click.option("--synthetic-file", "synthetic_files", multiple=True, required=True,
              help="Synthetic pairs JSONL. Images are read relative to its directory. Repeat per layout.")
@click.option("--ratio", default=0.15, show_default=True, help="Extra synthetic samples as a fraction of the base train set.")
@click.option("--output-file", default="data/processed/train_synth15.jsonl", show_default=True)
@click.option("--seed", default=42, show_default=True)
def main(base_file: str, synthetic_files, ratio: float, output_file: str, seed: int) -> None:
    base = read_jsonl(base_file)
    synthetic_sets = [(read_jsonl(f), os.path.dirname(os.path.abspath(f))) for f in synthetic_files]
    out_dir = os.path.dirname(os.path.abspath(output_file))

    mixed = mix_into_train(
        base, os.path.dirname(os.path.abspath(base_file)), synthetic_sets, ratio, out_dir, seed
    )
    write_jsonl(mixed, output_file)
    n_extra = len(mixed) - len(base)
    logger.info(f"Base: {len(base)} | synthetic added: {n_extra} ({n_extra / len(base):.1%}) | total: {len(mixed)}")
    logger.info(f"Wrote {output_file}")


if __name__ == "__main__":
    main()
