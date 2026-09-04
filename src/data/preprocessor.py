"""
Data preprocessing and conversation template building.
"""

from typing import Any, Dict, List, Optional
from PIL import Image


def format_qwen_vl_conversation(
    image: Image.Image,
    target_table: Optional[str] = None,
    system_prompt: str = "You are an expert scientific figure analyzer. Extract the plotted quantitative data into a clean Markdown table."
) -> List[Dict[str, Any]]:
    """Formats an image and optional target into Qwen2.5-VL chat template format."""
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
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
