"""
Data preprocessing and conversation template building.
"""

from typing import Any, Dict, List, Optional
from PIL import Image


def format_qwen_vl_conversation(
    image: Image.Image,
    target_table: Optional[str] = None,
    system_prompt: str = "You are an expert scientific figure analyzer. Extract the plotted quantitative data into a clean Markdown table.",
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Formats an image and optional target into Qwen2.5-VL chat template format."""
    image_dict: Dict[str, Any] = {"type": "image", "image": image}
    if min_pixels is not None:
        image_dict["min_pixels"] = min_pixels
    if max_pixels is not None:
        image_dict["max_pixels"] = max_pixels

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                image_dict,
                {"type": "text", "text": "Extract all numerical data points from this scientific figure panel into a Markdown table with clear column headers."}
            ]
        }
    ]

    if target_table is not None:
        messages.append({
            "role": "assistant",
            "content": target_table.strip()
        })

    return messages
