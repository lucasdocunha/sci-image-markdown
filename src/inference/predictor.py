"""
Inference runner for scientific figure table extraction.
"""

from typing import Any, Dict, List, Optional, Union
import os
from PIL import Image
import pandas as pd

from ..models.loader import load_model_and_processor
from ..models.qwen_vl import QwenVLTableExtractor
from ..metrics.table_parser import parse_markdown_to_dataframe, extract_markdown_table_block
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


class TablePredictor:
    """High-level interface for predicting markdown tables from scientific figures."""

    def __init__(self, cfg: Dict[str, Any], adapter_path: Optional[str] = None):
        self.cfg = cfg
        self.model, self.processor = load_model_and_processor(
            cfg, is_training=False, adapter_path=adapter_path
        )
        self.model.eval()

        eval_cfg = cfg.get("evaluation", {})
        data_cfg = cfg.get("data", {})
        self.max_image_resolution = data_cfg.get("max_image_resolution", 280)

        # Build fallback caption index from metadata
        from ..utils.io import read_jsonl
        self.caption_index: Dict[str, str] = {}
        for meta_file in ["data/raw/train_metadata.jsonl", "data/raw/validation_metadata.jsonl", "data/raw/test_metadata.jsonl"]:
            if os.path.exists(meta_file):
                try:
                    for mline in read_jsonl(meta_file):
                        cid = mline.get("id")
                        cap = mline.get("caption")
                        if cid and cap:
                            self.caption_index[cid] = str(cap).strip()
                except Exception:
                    pass

        self.extractor = QwenVLTableExtractor(
            model=self.model,
            processor=self.processor,
            system_prompt=data_cfg.get("system_prompt", "Extract the plotted quantitative data into a clean Markdown table."),
            max_new_tokens=eval_cfg.get("max_new_tokens", 1024),
            temperature=eval_cfg.get("temperature", 0.0),
            min_pixels=data_cfg.get("min_pixels"),
            max_pixels=data_cfg.get("max_pixels"),
        )

    def predict_image(
        self,
        image_input: Union[str, Image.Image],
        caption: Optional[str] = None,
        sample_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Runs table extraction on a single image path or PIL image."""
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
            if not sample_id:
                sample_id = os.path.splitext(os.path.basename(image_input))[0]
        else:
            image = image_input.copy() if hasattr(image_input, "copy") else image_input

        if not caption and sample_id:
            caption = self.caption_index.get(sample_id)

        if self.max_image_resolution:
            image.thumbnail((self.max_image_resolution, self.max_image_resolution))

        raw_output = self.extractor.predict(image, caption=caption)
        table_block = extract_markdown_table_block(raw_output) or raw_output
        df, is_valid = parse_markdown_to_dataframe(table_block)

        return {
            "raw_output": raw_output,
            "markdown_table": table_block,
            "is_valid_table": is_valid,
            "dataframe": df,
        }

    def predict_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Runs extraction over a list of image paths."""
        results = []
        for path in image_paths:
            logger.info(f"Processing image: {path}")
            res = self.predict_image(path)
            res["image_path"] = path
            results.append(res)
        return results
