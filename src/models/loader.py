"""
Model loader and PEFT / Quantization configuration.
"""

from typing import Any, Dict, Optional, Tuple
import torch
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)

from ..utils.logging import setup_logger

logger = setup_logger(__name__)


def build_quantization_config(cfg: Dict[str, Any]) -> Optional[BitsAndBytesConfig]:
    """Constructs BitsAndBytesConfig for 4-bit / 8-bit QLoRA."""
    q_cfg = cfg.get("quantization", {})
    if not q_cfg.get("load_in_4bit", False):
        return None

    compute_dtype = torch.bfloat16 if q_cfg.get("bnb_4bit_compute_dtype") == "bfloat16" else torch.float16

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=q_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=q_cfg.get("bnb_4bit_use_double_quant", True),
    )


def build_peft_config(cfg: Dict[str, Any]) -> LoraConfig:
    """Constructs LoRA configuration."""
    p_cfg = cfg.get("peft", {})
    return LoraConfig(
        r=p_cfg.get("r", 16),
        lora_alpha=p_cfg.get("lora_alpha", 32),
        lora_dropout=p_cfg.get("lora_dropout", 0.05),
        bias=p_cfg.get("bias", "none"),
        task_type=p_cfg.get("task_type", "CAUSAL_LM"),
        target_modules=p_cfg.get("target_modules", ["q_proj", "v_proj"]),
    )


def load_model_and_processor(
    cfg: Dict[str, Any],
    is_training: bool = True,
    adapter_path: Optional[str] = None
) -> Tuple[Any, Any]:
    """Loads model and processor with optional LoRA / QLoRA wrapping."""
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name_or_path", "Qwen/Qwen2.5-VL-3B-Instruct")
    model_type = model_cfg.get("model_type", "qwen2_5_vl")
    trust_remote_code = model_cfg.get("trust_remote_code", True)

    logger.info(f"Loading processor for: {model_name}")
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )

    bnb_config = build_quantization_config(cfg) if (is_training and cfg.get("training", {}).get("method") == "qlora") else None
    torch_dtype = torch.bfloat16 if model_cfg.get("torch_dtype") == "bfloat16" else torch.float16

    logger.info(f"Loading base model: {model_name} (dtype={torch_dtype}, 4bit={bnb_config is not None})")

    if model_type == "qwen2_5_vl":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
    elif model_type == "florence2":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )

    if adapter_path:
        logger.info(f"Loading trained LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    elif is_training and cfg.get("training", {}).get("method") in ["qlora", "lora"]:
        if bnb_config is not None:
            model = prepare_model_for_kbit_training(model)
        peft_config = build_peft_config(cfg)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model, processor
