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

        # Mask prompt tokens prior to assistant table response
        im_start_id = getattr(self.processor.tokenizer, "convert_tokens_to_ids", lambda x: None)("<|im_start|>")
        assistant_ids = self.processor.tokenizer.encode("assistant", add_special_tokens=False) if hasattr(self.processor, "tokenizer") else []
        if im_start_id is not None and assistant_ids:
            header_ids = [im_start_id] + assistant_ids
            h_len = len(header_ids)
            for i in range(labels.shape[0]):
                seq = inputs["input_ids"][i].tolist()
                mask_end = 0
                for j in range(len(seq) - h_len):
                    if seq[j : j + h_len] == header_ids:
                        offset = h_len
                        if j + offset < len(seq) and seq[j + offset] == 198:  # newline '\n'
                            offset += 1
                        mask_end = j + offset
                        break
                if mask_end > 0:
                    labels[i, :mask_end] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        inputs["labels"] = labels

        return inputs
