"""
Data preprocessing and conversation template building.
"""

from typing import Any, Dict, List, Optional
from PIL import Image


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert scientific figure analyzer. Your task is to extract all numerical data points and "
    "values plotted in the given scientific figure panel into a structured Markdown table with accurate column headers.\n"
    "Follow these strict guidelines:\n"
    "1. Peak-Picking: For continuous spectroscopy plots (e.g. RBS, Raman, XRD, XPS, FTIR), do not sample arbitrary "
    "regular grid intervals. Locate and extract the precise coordinates and labels of physical peaks and extrema.\n"
    "2. Multi-Panel Figures: If the figure contains multiple subplots (panels a, b, c), extract a separate Markdown "
    "table for EACH panel. Do not stop early.\n"
    "3. Pure Numeric Values: Output clean numerical values without approximation symbols (never use '~', 'approx'). "
    "Use standard scientific notation (e.g. 1e7) when applicable.\n"
    "4. Avoid Dummy Sequences: Never output artificial linear progression sequences (such as 0, 100, 200, 300...)."
)


def format_qwen_vl_conversation(
    image: Image.Image,
    target_table: Optional[str] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
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
                {
                    "type": "text",
                    "text": (
                        "Extract all numerical data points from this scientific figure panel into a Markdown table "
                        "with clear column headers. If there are multiple panels or subplots, extract a separate "
                        "table for each panel."
                    ),
                }
            ]
        }
    ]


    if target_table is not None:
        messages.append({
            "role": "assistant",
            "content": target_table.strip()
        })

    return messages
