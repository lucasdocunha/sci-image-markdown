"""
Inference and fine-tuning wrapper for Florence-2 models.
"""

from typing import Any, Optional
from PIL import Image
import torch


class Florence2TableExtractor:
    """Wrapper for Florence-2 scientific table extraction."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        task_prompt: str = "<TABLE>",
        max_new_tokens: int = 1024,
    ):
        self.model = model
        self.processor = processor
        self.task_prompt = task_prompt
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def predict(self, image: Image.Image) -> str:
        """Extracts table from an image using Florence-2."""
        inputs = self.processor(
            text=self.task_prompt,
            images=image,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs.get("pixel_values"),
            max_new_tokens=self.max_new_tokens,
            num_beams=3,
            do_sample=False,
        )

        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False
        )[0]

        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=self.task_prompt,
            image_size=(image.width, image.height)
        )

        if isinstance(parsed_answer, dict):
            return str(parsed_answer.get(self.task_prompt, generated_text))
        return str(parsed_answer)
