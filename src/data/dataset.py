"""
Dataset class for Sci-Image scientific figure to markdown table task.
"""

import os
import random
from typing import Any, Dict, List, Optional
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.io import read_jsonl


class SciImageTableDataset(Dataset):
    """PyTorch Dataset for scientific figure panels and markdown table targets.

    When ``augment`` is enabled, the image degradation pipeline (rotation,
    blur, noise, JPEG compression, ...) is applied on the fly at every access,
    so the training loop sees a fresh degradation of the same original each
    epoch. ``augment_probability`` gates the whole pipeline at image level:
    with 0.75, for example, 75% of the accesses are degraded and 25% stay
    clean. Validation/test datasets must pass ``augment=False``.
    """

    def __init__(
        self,
        data_path: str,
        image_dir: Optional[str] = None,
        system_prompt: str = "Extract the plotted data from this figure into a clean Markdown table.",
        max_image_resolution: Optional[int] = None,
        augment: bool = False,
        rotation_limit: float = 10.0,
        severity: str = "moderate",
        augment_probability: float = 1.0,
        **kwargs: Any,
    ):
        self.data_path = data_path
        self.image_dir = image_dir
        self.system_prompt = system_prompt
        self.max_image_resolution = max_image_resolution

        if severity not in ("mild", "moderate", "strong"):
            raise ValueError(f"severity must be mild, moderate, or strong: {severity}")
        if not 0.0 <= augment_probability <= 1.0:
            raise ValueError(f"augment_probability must be in [0, 1]: {augment_probability}")
        self.augment_probability = augment_probability

        self.pipeline = None
        if augment:
            # Deferred import keeps albumentations/cv2 out of the eval path.
            from .augment_images_online import build_pipeline

            self.pipeline = build_pipeline(rotation_limit, severity)

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
        if self.pipeline is not None and random.random() < self.augment_probability:
            degraded = self.pipeline(image=np.array(image))
            image = Image.fromarray(degraded["image"])
        if self.max_image_resolution:
            image.thumbnail((self.max_image_resolution, self.max_image_resolution))

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
