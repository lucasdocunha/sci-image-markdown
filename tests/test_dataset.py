"""
Unit tests for data generation and dataset loading.
"""

import os
import shutil
import tempfile
import pytest
from PIL import Image

from prepare_data import create_synthetic_demo_data
from src.data.dataset import SciImageTableDataset
from src.data.preprocessor import format_qwen_vl_conversation


def test_synthetic_data_generation_and_loading():
    temp_dir = tempfile.mkdtemp()
    try:
        create_synthetic_demo_data(temp_dir, num_samples=6)

        train_path = os.path.join(temp_dir, "train.jsonl")
        assert os.path.exists(train_path)

        dataset = SciImageTableDataset(data_path=train_path)
        assert len(dataset) > 0

        sample = dataset[0]
        assert "image" in sample
        assert isinstance(sample["image"], Image.Image)
        assert "table" in sample
        assert "|" in sample["table"]

        # Test preprocessor conversation formatting
        messages = format_qwen_vl_conversation(sample["image"], sample["table"])
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == sample["table"].strip()

    finally:
        shutil.rmtree(temp_dir)
