"""
Custom Data Collator for Vision-Language Models (Qwen2.5-VL and Qwen2-VL).
"""

from typing import Any, Dict, List
import torch
from qwen_vl_utils import process_vision_info


class QwenVLDataCollator:
    """Collates and tokenizes multimodal batches for Qwen2.5-VL fine-tuning."""

    def __init__(self, processor: Any):
        self.processor = processor

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = []
        image_inputs = []

        for item in batch:
            messages = item if isinstance(item, list) else item.get("messages")
            if messages is None:
                raise ValueError("Item in batch does not contain 'messages' or is not a list of messages.")

            # Process text with chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)

            # Extract vision inputs
            image_input, _ = process_vision_info(messages)
            image_inputs.append(image_input)

        # Batch process using Qwen processor
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        )

        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels

        return inputs
