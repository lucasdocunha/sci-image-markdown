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


from .icdar_loss import ICDARMetricLoss


class ICDARTrainer(Trainer):
    """Custom Hugging Face Trainer that integrates ICDAR Metric Loss (SCST / Token-Weighted)."""

    def __init__(self, *args, icdar_loss_module: Optional[ICDARMetricLoss] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.icdar_loss_module = icdar_loss_module

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.icdar_loss_module is None:
            # Handle newer transformers versions supporting num_items_in_batch
            kwargs = {}
            if num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        outputs = model(**inputs)
        loss, loss_stats = self.icdar_loss_module(model, inputs, sft_loss=outputs.loss)

        if self.state.global_step % max(1, self.args.logging_steps) == 0:
            for k, v in loss_stats.items():
                if isinstance(v, (int, float)):
                    self.log({f"train/{k}": v})

        return (loss, outputs) if return_outputs else loss


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

        # Build ICDAR metric loss module if configured
        t_cfg = cfg.get("training", {})
        loss_type = t_cfg.get("loss_type", "ce")
        if loss_type in ("scst", "token_weighted", "hybrid", "icdar_metric"):
            m_cfg = t_cfg.get("metric_loss", {})
            actual_type = "scst" if loss_type == "icdar_metric" else loss_type
            self.icdar_loss_module = ICDARMetricLoss(
                processor=self.processor,
                loss_type=actual_type,
                lambda_metric=m_cfg.get("lambda_metric", 0.3),
                rel_tol=m_cfg.get("rel_tol", 0.05),
                temperature=m_cfg.get("temperature", 0.7),
                max_gen_tokens=m_cfg.get("max_gen_tokens", 512),
                numeric_weight=m_cfg.get("numeric_weight", 3.0),
                structural_weight=m_cfg.get("structural_weight", 2.0),
            )
            logger.info(f"Initialized ICDARMetricLoss (type={actual_type}, lambda={m_cfg.get('lambda_metric', 0.3)})")
        else:
            self.icdar_loss_module = None

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

        trainer_cls = ICDARTrainer if self.icdar_loss_module is not None else Trainer
        trainer_kwargs = {
            "model": self.model,
            "args": self.training_args,
            "train_dataset": train_formatted,
            "eval_dataset": eval_formatted,
            "data_collator": self.data_collator,
        }
        if self.icdar_loss_module is not None:
            trainer_kwargs["icdar_loss_module"] = self.icdar_loss_module

        trainer = trainer_cls(**trainer_kwargs)

        logger.info("Starting training loop...")
        train_result = trainer.train()

        logger.info("Saving best model adapter and tokenizer/processor...")
        output_dir = self.training_args.output_dir
        final_adapter_dir = os.path.join(output_dir, "final_adapter")
        os.makedirs(final_adapter_dir, exist_ok=True)
        trainer.save_model(final_adapter_dir)
        self.processor.save_pretrained(final_adapter_dir)

        return train_result

