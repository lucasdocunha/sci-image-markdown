"""
Per-panel inference: each panel of a figure is cropped by its bbox and predicted alone.
"""

from typing import Callable, Dict, List

from PIL import Image

from ..analysis.convert import split_tables


def crop_panel(image: Image.Image, bbox: dict, margin: float = 0.0) -> Image.Image:
    """Crops a panel by its bbox, grown by `margin` (fraction of its size) and clamped to the image."""
    pad_x = bbox["width"] * margin
    pad_y = bbox["height"] * margin
    left = max(0, round(bbox["x"] - pad_x))
    top = max(0, round(bbox["y"] - pad_y))
    right = min(image.width, round(bbox["x"] + bbox["width"] + pad_x))
    bottom = min(image.height, round(bbox["y"] + bbox["height"] + pad_y))
    return image.crop((left, top, right, bottom))


def predict_panels(
    predict: Callable[[Image.Image], str],
    image: Image.Image,
    bboxes: List[dict],
    margin: float = 0.0,
) -> Dict[str, dict]:
    """Runs `predict` on every panel crop; keeps the first table of each output."""
    results = {}
    for bbox in bboxes:
        raw_output = predict(crop_panel(image, bbox, margin))
        tables = split_tables(raw_output)
        results[bbox["panel_id"]] = {
            "table": tables[0] if tables else None,
            "raw_output": raw_output,
            "extra_tables": max(0, len(tables) - 1),
        }
    return results
