"""
Training pipeline and orchestration for multimodal table extraction.
"""

import os
from typing import Any, Dict, Optional
import torch
from transformers import TrainingArguments, Trainer
from trl import SFTTrainer

from ..utils.logging import setup_logger
from ..data.collator import QwenVLDataCollator
from ..data.preprocessor import format_qwen_vl_conversation

logger = setup_logger(__name__)


def build_training_arguments(cfg: Dict[str, Any]) -> TrainingArguments:
    """Builds HuggingFace TrainingArguments from configuration dictionary."""
    t_cfg = cfg.get("training", {})
    output_dir = t_cfg.get("output_dir", "outputs/checkpoints")
    os.makedirs(output_dir, exist_ok=True)

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=t_cfg.get("num_train_epochs", 5),
        per_device_train_batch_size=t_cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=t_cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=t_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=float(t_cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=t_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=t_cfg.get("warmup_ratio", 0.05),
        weight_decay=t_cfg.get("weight_decay", 0.01),
        logging_steps=t_cfg.get("logging_steps", 10),
        eval_strategy=t_cfg.get("eval_strategy", "epoch"),
        save_strategy=t_cfg.get("save_strategy", "epoch"),
        save_total_limit=t_cfg.get("save_total_limit", 3),
        load_best_model_at_end=t_cfg.get("load_best_model_at_end", False),
        bf16=t_cfg.get("bf16", True),
        fp16=t_cfg.get("fp16", False),
        dataloader_num_workers=t_cfg.get("dataloader_num_workers", 2),
        gradient_checkpointing=t_cfg.get("gradient_checkpointing", True),
        report_to=t_cfg.get("report_to", "none"),
        remove_unused_columns=False,
    )


class SciImageTableTrainer:
    """Orchestrator for fine-tuning VLM on scientific table extraction."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        train_dataset: Any,
        eval_dataset: Optional[Any],
        cfg: Dict[str, Any],
    ):
        self.model = model
        self.processor = processor
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.cfg = cfg
        self.training_args = build_training_arguments(cfg)
        self.data_collator = QwenVLDataCollator(processor=self.processor)

    def _prepare_formatted_dataset(self, dataset: Any):
        """Converts raw dataset items into conversation message dictionaries."""
        formatted = []
        for idx in range(len(dataset)):
            item = dataset[idx]
            msgs = format_qwen_vl_conversation(
                image=item["image"],
                target_table=item["table"],
                system_prompt=item["system_prompt"]
            )
            formatted.append({"messages": msgs})
        return formatted

    def train(self):
        """Executes model training."""
        logger.info("Preparing training dataset formatting...")
        train_formatted = self._prepare_formatted_dataset(self.train_dataset)

        eval_formatted = None
        if self.eval_dataset is not None:
            logger.info("Preparing validation dataset formatting...")
            eval_formatted = self._prepare_formatted_dataset(self.eval_dataset)

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_formatted,
            eval_dataset=eval_formatted,
            data_collator=self.data_collator,
        )

        logger.info("Starting training loop...")
        train_result = trainer.train()

        logger.info("Saving best model adapter and tokenizer/processor...")
        output_dir = self.training_args.output_dir
        trainer.save_model(os.path.join(output_dir, "final_adapter"))
        self.processor.save_pretrained(os.path.join(output_dir, "final_adapter"))

        return train_result
