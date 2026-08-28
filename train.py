"""
CLI tool for training/fine-tuning Vision-Language Models on Sci-Image dataset.
"""

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

    # Load datasets
    data_cfg = cfg.get("data", {})
    max_res = data_cfg.get("max_image_resolution", 280)
    train_dataset = SciImageTableDataset(
        data_path=data_cfg.get("train_file", "data/processed/train.jsonl"),
        image_dir=data_cfg.get("image_folder"),
        system_prompt=data_cfg.get("system_prompt"),
        max_image_resolution=max_res,
    )
    logger.info(f"Loaded {len(train_dataset)} training samples.")

    eval_dataset = None
    val_file = data_cfg.get("val_file", "data/processed/val.jsonl")
    if val_file:
        try:
            eval_dataset = SciImageTableDataset(
                data_path=val_file,
                image_dir=data_cfg.get("image_folder"),
                system_prompt=data_cfg.get("system_prompt"),
                max_image_resolution=max_res,
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
