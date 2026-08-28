"""
CLI tool for exporting and merging fine-tuned LoRA adapters with base VLM.
Produces standalone Hugging Face model weights ready for production inference.
"""

import os
import click
import torch
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from peft import PeftModel

from src.utils.config import load_config
from src.utils.logging import setup_logger

logger = setup_logger("export")


@click.command()
@click.option("--config", default="configs/default.yaml", help="Path to base configuration YAML.")
@click.option("--adapter-path", default="outputs/checkpoints/final_adapter", help="Path to trained PEFT LoRA adapter.")
@click.option("--output-dir", default="outputs/merged_model", help="Directory to save merged standalone model.")
@click.option("--torch-dtype", default="float16", help="Torch dtype for merged model (float16 or bfloat16).")
@click.option("--device", default="auto", help="Device to perform merge ('auto', 'cuda', or 'cpu').")
def main(config: str, adapter_path: str, output_dir: str, torch_dtype: str, device: str):
    """Merges a trained LoRA adapter into the base vision-language model."""
    cfg = load_config(config) if os.path.exists(config) else {}
    model_cfg = cfg.get("model", {})
    base_model_name = model_cfg.get("name_or_path", "Qwen/Qwen2-VL-2B-Instruct")
    model_type = model_cfg.get("model_type", "qwen2_vl")
    trust_remote_code = model_cfg.get("trust_remote_code", True)

    dtype = torch.bfloat16 if torch_dtype == "bfloat16" else torch.float16

    logger.info(f"Loading processor for: {base_model_name}")
    processor = AutoProcessor.from_pretrained(
        adapter_path if os.path.exists(os.path.join(adapter_path, "preprocessor_config.json")) else base_model_name,
        trust_remote_code=trust_remote_code,
    )

    logger.info(f"Loading unquantized base model: {base_model_name} (dtype={dtype})")
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": device,
        "trust_remote_code": trust_remote_code,
    }

    if "qwen2_5" in model_type or "qwen2.5" in model_type or "Qwen2.5" in base_model_name:
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base_model_name, **model_kwargs)
    elif "qwen2" in model_type or "qwen" in model_type or "Qwen2" in base_model_name:
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(base_model_name, **model_kwargs)
    elif model_type == "florence2":
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    else:
        base_model = AutoModelForImageTextToText.from_pretrained(base_model_name, **model_kwargs)

    logger.info(f"Loading and applying LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    logger.info("Merging LoRA weights into base model weights...")
    merged_model = model.merge_and_unload()

    logger.info(f"Saving merged standalone model to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    logger.info(f"Successfully exported fine-tuned merged model to: {output_dir}")


if __name__ == "__main__":
    main()
