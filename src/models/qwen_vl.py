"""
Inference and Generation wrapper for Qwen2.5-VL models.
"""

from typing import Any, Dict, List, Optional
from PIL import Image
import torch
from qwen_vl_utils import process_vision_info

from ..data.preprocessor import format_qwen_vl_conversation


class QwenVLTableExtractor:
    """Wrapper for running inference on scientific figures with Qwen2.5-VL."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        system_prompt: str = "You are an expert scientific figure analyzer. Extract the plotted quantitative data into a clean Markdown table.",
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        self.model = model
        self.processor = processor
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    @torch.no_grad()
    def predict(self, image: Image.Image) -> str:
        """Extracts markdown table from a single PIL image."""
        messages = format_qwen_vl_conversation(
            image=image,
            target_table=None,
            system_prompt=self.system_prompt
        )

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0.0,
        }
        if self.temperature > 0.0:
            gen_kwargs["temperature"] = self.temperature

        generated_ids = self.model.generate(**inputs, **gen_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return output_text.strip()
