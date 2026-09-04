"""
Model loader and PEFT / Quantization configuration.
Supports Qwen2.5-VL and Qwen2-VL vision-language architectures with 4-bit QLoRA.
"""

from typing import Any, Dict, Optional, Tuple
import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast
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

    load_4bit = cfg.get("quantization", {}).get("load_in_4bit", False) or (is_training and cfg.get("training", {}).get("method") == "qlora")
    bnb_config = build_quantization_config(cfg) if load_4bit else None
    torch_dtype = torch.bfloat16 if model_cfg.get("torch_dtype") == "bfloat16" else torch.float16

    attn_impl = model_cfg.get("attn_implementation")
    if attn_impl == "flash_attention_2" and not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8):
        logger.info("FlashAttention-2 requires compute capability >= 8.0. Falling back to 'sdpa'.")
        attn_impl = "sdpa"

    logger.info(f"Loading base model: {model_name} (dtype={torch_dtype}, 4bit={bnb_config is not None}, attn={attn_impl})")

    kwargs = {
        "quantization_config": bnb_config,
        "torch_dtype": torch_dtype,
        "device_map": "auto",
        "trust_remote_code": trust_remote_code,
    }
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl

    if "qwen2_5" in model_type or "qwen2.5" in model_type or "Qwen2.5" in model_name:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
    elif "qwen2" in model_type or "qwen" in model_type or "Qwen2" in model_name:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
    else:
        model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)

    if adapter_path:
        logger.info(f"Loading trained LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    elif is_training and cfg.get("training", {}).get("method") in ["qlora", "lora"]:
        if bnb_config is not None:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=cfg.get("training", {}).get("gradient_checkpointing", True)
            )
        peft_config = build_peft_config(cfg)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Apply selective memory-efficient forward pass to prevent OOM on large vocabulary logits
    if is_training:
        target_base = getattr(model, "base_model", None)
        if target_base is not None and hasattr(target_base, "model"):
            base_inner = target_base.model
            if isinstance(base_inner, (Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration)):
                base_inner.forward = _memory_efficient_qwen2_vl_forward.__get__(base_inner, type(base_inner))
                logger.info("Applied selective memory-efficient forward pass to base model.")
        elif isinstance(model, (Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration)):
            model.forward = _memory_efficient_qwen2_vl_forward.__get__(model, type(model))
            logger.info("Applied selective memory-efficient forward pass to model.")

        if cfg.get("training", {}).get("gradient_checkpointing", True) and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    return model, processor


def _memory_efficient_qwen2_vl_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    mm_token_type_ids=None,
    logits_to_keep=0,
    **kwargs,
):
    """Memory-efficient forward pass that only computes lm_head and cross-entropy on non-masked tokens."""
    import torch.nn as nn

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state

    loss = None
    logits = None

    if labels is not None:
        shift_labels = nn.functional.pad(labels, (0, 1), value=-100)
        shift_labels = shift_labels[..., 1:].contiguous()

        shift_labels_flat = shift_labels.view(-1)
        valid_mask = shift_labels_flat != -100

        if valid_mask.any():
            shift_hidden = hidden_states.view(-1, hidden_states.shape[-1])
            valid_hidden = shift_hidden[valid_mask]
            valid_labels = shift_labels_flat[valid_mask].to(valid_hidden.device)

            valid_logits = self.lm_head(valid_hidden).float()

            num_items_in_batch = kwargs.get("num_items_in_batch", None)
            reduction = "sum" if num_items_in_batch is not None else "mean"
            loss = nn.functional.cross_entropy(valid_logits, valid_labels, reduction=reduction)
            if reduction == "sum":
                if torch.is_tensor(num_items_in_batch):
                    num_items_in_batch = num_items_in_batch.to(loss.device)
                loss = loss / num_items_in_batch
        else:
            loss = torch.tensor(0.0, device=hidden_states.device, requires_grad=True)
    else:
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

    return Qwen2VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=outputs.rope_deltas,
    )

