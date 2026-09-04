"""
Dataset class for Sci-Image scientific figure to markdown table task.
"""

import os
from typing import Any, Dict, List, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset

from ..utils.io import read_jsonl


class SciImageTableDataset(Dataset):
    """PyTorch Dataset for scientific figure panels and markdown table targets."""

    def __init__(
        self,
        data_path: str,
        image_dir: Optional[str] = None,
        system_prompt: str = "Extract the plotted data from this figure into a clean Markdown table."
    ):
        self.data_path = data_path
        self.image_dir = image_dir
        self.system_prompt = system_prompt

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

        self.records: List[Dict[str, Any]] = read_jsonl(data_path)

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_image_path(self, img_path: str) -> str:
        if os.path.isabs(img_path) and os.path.exists(img_path):
            return img_path
        if self.image_dir:
            candidate = os.path.join(self.image_dir, img_path)
            if os.path.exists(candidate):
                return candidate
        # Check relative to dataset file directory
        parent_dir = os.path.dirname(os.path.abspath(self.data_path))
        candidate = os.path.join(parent_dir, img_path)
        if os.path.exists(candidate):
            return candidate
        # Check raw sibling directory or basename fallbacks
        raw_candidate = os.path.join(os.path.dirname(parent_dir), "raw", img_path)
        if os.path.exists(raw_candidate):
            return raw_candidate
        basename_candidate = os.path.join(parent_dir, "images", os.path.basename(img_path))
        if os.path.exists(basename_candidate):
            return basename_candidate
        raw_basename_candidate = os.path.join(os.path.dirname(parent_dir), "raw", "images", os.path.basename(img_path))
        if os.path.exists(raw_basename_candidate):
            return raw_basename_candidate
        return img_path

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.records[idx]

        image_path = self._resolve_image_path(item["image"])
        image = Image.open(image_path).convert("RGB")

        markdown_table = item.get("table", item.get("markdown", item.get("target", "")))
        sample_id = item.get("id", str(idx))
        metadata = item.get("metadata", {})

        return {
            "id": sample_id,
            "image": image,
            "image_path": image_path,
            "table": markdown_table,
            "system_prompt": self.system_prompt,
            "metadata": metadata,
        }
