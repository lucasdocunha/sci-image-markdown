"""
Training pipeline and orchestration for multimodal table extraction.
"""

import os
from typing import Any, Dict, Optional
import torch
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer

from ..utils.logging import setup_logger
from ..data.collator import QwenVLDataCollator
from ..data.preprocessor import format_qwen_vl_conversation

logger = setup_logger(__name__)


class LazyMultimodalDataset(Dataset):
    """Dataset wrapper that formats multimodal messages on demand."""

    def __init__(self, dataset: Any, max_table_chars: Optional[int] = None):
        self.dataset = dataset
        self.max_table_chars = max_table_chars

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        table = item["table"]
        if self.max_table_chars and len(table) > self.max_table_chars:
            table = table[:self.max_table_chars]
        msgs = format_qwen_vl_conversation(
            image=item["image"],
            target_table=table,
            system_prompt=item["system_prompt"]
        )
        return {"messages": msgs}


def build_training_arguments(cfg: Dict[str, Any]) -> TrainingArguments:
    """Builds HuggingFace TrainingArguments from configuration dictionary."""
    t_cfg = cfg.get("training", {})
    output_dir = t_cfg.get("output_dir", "outputs/checkpoints")
    os.makedirs(output_dir, exist_ok=True)

    bf16 = t_cfg.get("bf16", False)
    fp16 = t_cfg.get("fp16", True)

    # Check CUDA device capability: sm_80+ supports native bf16, older GPUs (e.g. GTX 1660 / Turing) should use fp16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
        bf16 = False
        fp16 = True

    warmup_steps = t_cfg.get("warmup_steps", 20)
    optim = t_cfg.get("optim", "paged_adamw_8bit")

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=t_cfg.get("num_train_epochs", 3),
        max_steps=t_cfg.get("max_steps", -1),
        per_device_train_batch_size=t_cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=t_cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=t_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=float(t_cfg.get("learning_rate", 2e-4)),
        lr_scheduler_type=t_cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=warmup_steps,
        weight_decay=t_cfg.get("weight_decay", 0.01),
        logging_steps=t_cfg.get("logging_steps", 10),
        eval_strategy=t_cfg.get("eval_strategy", "epoch"),
        save_strategy=t_cfg.get("save_strategy", "epoch"),
        save_total_limit=t_cfg.get("save_total_limit", 2),
        load_best_model_at_end=t_cfg.get("load_best_model_at_end", False),
        bf16=bf16,
        fp16=fp16,
        optim=optim,
        dataloader_num_workers=t_cfg.get("dataloader_num_workers", 2),
        dataloader_pin_memory=False,
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

    def train(self):
        """Executes model training."""
        logger.info("Initializing lazy multimodal datasets...")
        max_table_chars = self.cfg.get("data", {}).get("max_table_chars", 1500)
        train_formatted = LazyMultimodalDataset(self.train_dataset, max_table_chars=max_table_chars)
        eval_formatted = (
            LazyMultimodalDataset(self.eval_dataset, max_table_chars=max_table_chars)
            if self.eval_dataset is not None
            else None
        )

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
        final_adapter_dir = os.path.join(output_dir, "final_adapter")
        os.makedirs(final_adapter_dir, exist_ok=True)
        trainer.save_model(final_adapter_dir)
        self.processor.save_pretrained(final_adapter_dir)

        return train_result

