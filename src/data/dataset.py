"""
Dataset class for Sci-Image scientific figure to markdown table task.
"""

import os
import random
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset

from ..utils.io import read_jsonl


def stack_double_table_sample(
    image: Image.Image,
    markdown_table: str,
    sep_px: int = 8,
) -> Tuple[Image.Image, str]:
    """Stacks a figure image vertically on top of itself and repeats the Markdown table.

    Used for multi-table data augmentation so the model learns to extract two tables
    instead of stopping early after the first table.
    """
    w, h = image.size
    stacked_img = Image.new("RGB", (w, h * 2 + sep_px), color=(255, 255, 255))
    stacked_img.paste(image, (0, 0))
    stacked_img.paste(image, (0, h + sep_px))

    clean_table = markdown_table.strip()
    doubled_table = f"{clean_table}\n\n{clean_table}" if clean_table else ""
    return stacked_img, doubled_table


class SciImageTableDataset(Dataset):
    """PyTorch Dataset for scientific figure panels and markdown table targets."""

    def __init__(
        self,
        data_path: str,
        image_dir: Optional[str] = None,
        system_prompt: str = "Extract the plotted data from this figure into a clean Markdown table.",
        max_image_resolution: Optional[int] = None,
        extra_data_paths: Optional[List[str]] = None,
        extra_sample_ratio: Optional[float] = None,
        double_table_aug_prob: float = 0.0,
        seed: int = 42,
        **kwargs: Any,
    ):
        self.data_path = data_path
        self.image_dir = image_dir
        self.system_prompt = system_prompt
        self.max_image_resolution = max_image_resolution
        self.double_table_aug_prob = float(double_table_aug_prob)
        self.rng = random.Random(seed)

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

        self.records: List[Dict[str, Any]] = []
        base_records = read_jsonl(data_path)
        base_dir = os.path.dirname(os.path.abspath(data_path))
        for rec in base_records:
            rec_copy = dict(rec)
            rec_copy["_source_dir"] = base_dir
            self.records.append(rec_copy)

        if extra_data_paths:
            extra_candidates: List[Dict[str, Any]] = []
            for extra_path in extra_data_paths:
                if extra_path and os.path.exists(extra_path):
                    extra_dir = os.path.dirname(os.path.abspath(extra_path))
                    for rec in read_jsonl(extra_path):
                        rec_copy = dict(rec)
                        rec_copy["_source_dir"] = extra_dir
                        extra_candidates.append(rec_copy)

            if extra_sample_ratio is not None and extra_candidates:
                if extra_sample_ratio <= 0.0:
                    extra_candidates = []
                elif extra_sample_ratio < 1.0:
                    target_k = max(1, int(round(len(base_records) * float(extra_sample_ratio))))
                    # Prioritize multi-table and multi-column (>=3 cols) synthetic samples for TEDS structure diversity
                    def _struct_priority(r: Dict[str, Any]) -> Tuple[int, int]:
                        tbl = str(r.get("table", r.get("markdown", "")))
                        is_multi = 1 if "\n\n|" in tbl else 0
                        first_ln = tbl.strip().split("\n")[0] if tbl.strip() else ""
                        n_cols = max(0, len(first_ln.split("|")) - 2)
                        return (is_multi, 1 if n_cols >= 3 else 0)

                    shuffled = list(extra_candidates)
                    self.rng.shuffle(shuffled)
                    shuffled.sort(key=_struct_priority, reverse=True)
                    extra_candidates = shuffled[: min(target_k, len(shuffled))]

            self.records.extend(extra_candidates)

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_image_path(self, img_path: str, source_dir: Optional[str] = None) -> str:
        if os.path.isabs(img_path) and os.path.exists(img_path):
            return img_path
        if source_dir:
            candidate = os.path.join(source_dir, img_path)
            if os.path.exists(candidate):
                return candidate
            basename_candidate = os.path.join(source_dir, "images", os.path.basename(img_path))
            if os.path.exists(basename_candidate):
                return basename_candidate
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

        image_path = self._resolve_image_path(item["image"], source_dir=item.get("_source_dir"))
        image = Image.open(image_path).convert("RGB")
        if self.max_image_resolution:
            image.thumbnail((self.max_image_resolution, self.max_image_resolution))

        markdown_table = item.get("table", item.get("markdown", item.get("target", "")))
        sample_id = item.get("id", str(idx))
        metadata = dict(item.get("metadata", {}))

        # Apply 10% double-table augmentation (stacks image vertically on top of itself + repeats table)
        if self.double_table_aug_prob > 0.0 and self.rng.random() < self.double_table_aug_prob:
            image, markdown_table = stack_double_table_sample(image, markdown_table)
            if self.max_image_resolution:
                image.thumbnail((self.max_image_resolution, self.max_image_resolution * 2))
            metadata["double_table_augmented"] = True

        return {
            "id": sample_id,
            "image": image,
            "image_path": image_path,
            "table": markdown_table,
            "system_prompt": self.system_prompt,
            "metadata": metadata,
        }
