"""
CLI tool for training/fine-tuning Vision-Language Models on Sci-Image dataset.
"""

import os
import click
from src.utils.config import load_config, merge_configs
from src.utils.logging import setup_logger
from src.models.loader import load_model_and_processor
from src.data.dataset import SciImageTableDataset
from src.training.trainer import SciImageTableTrainer

logger = setup_logger("train")


@click.command()
@click.option("--config", default="configs/default.yaml", help="Path to base configuration YAML.")
@click.option("--model-config", default=None, help="Optional model override configuration YAML.")
@click.option("--training-config", default=None, help="Optional training override configuration YAML.")
def main(config: str, model_config: str, training_config: str):
    """Fine-tunes a Vision-Language Model on Sci-Image Task 2."""
    cfg = load_config(config)

    if model_config:
        cfg = merge_configs(cfg, load_config(model_config))
    if training_config:
        cfg = merge_configs(cfg, load_config(training_config))

    logger.info(f"Loaded configuration for experiment: {cfg.get('experiment_name')}")

    # Load datasets (Real + Synthetic + 10% Double-Table Augmentation)
    data_cfg = cfg.get("data", {})
    max_res = data_cfg.get("max_image_resolution", 280)
    extra_train_paths = []

    synth_train_file = data_cfg.get("synthetic_train_file")
    if synth_train_file:
        if not os.path.exists(synth_train_file):
            synth_dir = os.path.dirname(synth_train_file) or "data/synthetic"
            synth_num = int(data_cfg.get("synthetic_num_samples", 400))
            logger.info(f"Synthetic train file not found at {synth_train_file}. Generating {synth_num} synthetic samples in {synth_dir}...")
            from generate_synthetic_dataset import generate_dataset
            generate_dataset(synth_dir, num_samples=synth_num, seed=int(cfg.get("seed", 42)), overwrite=True)
        if os.path.exists(synth_train_file):
            extra_train_paths.append(synth_train_file)
            logger.info(f"Including synthetic training dataset: {synth_train_file}")

    double_aug_prob = float(data_cfg.get("double_table_aug_prob", 0.02))
    synth_ratio_raw = data_cfg.get("synthetic_sample_ratio", 0.02)
    synth_sample_ratio = float(synth_ratio_raw) if synth_ratio_raw is not None else None
    train_dataset = SciImageTableDataset(
        data_path=data_cfg.get("train_file", "data/processed/train.jsonl"),
        image_dir=data_cfg.get("image_folder"),
        system_prompt=data_cfg.get("system_prompt"),
        max_image_resolution=max_res,
        extra_data_paths=extra_train_paths,
        extra_sample_ratio=synth_sample_ratio,
        double_table_aug_prob=double_aug_prob,
        seed=int(cfg.get("seed", 42)),
    )
    logger.info(
        f"Loaded {len(train_dataset)} total training samples "
        f"(extra_sources={len(extra_train_paths)}, synth_ratio={synth_sample_ratio}, double_table_aug_prob={double_aug_prob:.0%})."
    )

    eval_dataset = None
    val_file = data_cfg.get("val_file", "data/processed/val.jsonl")
    if val_file:
        try:
            eval_dataset = SciImageTableDataset(
                data_path=val_file,
                image_dir=data_cfg.get("image_folder"),
                system_prompt=data_cfg.get("system_prompt"),
                max_image_resolution=max_res,
                double_table_aug_prob=0.0,
            )
            logger.info(f"Loaded {len(eval_dataset)} validation samples.")
        except FileNotFoundError:
            logger.warning(f"Validation file {val_file} not found. Skipping eval dataset.")

    # Load model and processor
    model, processor = load_model_and_processor(cfg, is_training=True)

    # Initialize and run trainer
    trainer = SciImageTableTrainer(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        cfg=cfg
    )

    trainer.train()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
