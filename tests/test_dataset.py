"""
Unit tests for data generation and dataset loading.
"""

import os
import shutil
import tempfile
import pytest
from PIL import Image
import numpy as np

from prepare_data import create_synthetic_demo_data
from src.data.dataset import SciImageTableDataset
from src.data.preprocessor import format_qwen_vl_conversation

# Runtime safety: the affine/rotation pipeline makes a degraded image virtually
# always differ in size or pixels from the original, and continuous draws make
# an exact coincidence impossible.
TRIALS = 240
FRACTION_TOL = 0.10


@pytest.fixture(scope="module")
def demo_data():
    temp_dir = tempfile.mkdtemp()
    create_synthetic_demo_data(temp_dir, num_samples=6)
    train_path = os.path.join(temp_dir, "train.jsonl")
    if os.path.getsize(train_path) == 0:
        shutil.rmtree(temp_dir, ignore_errors=True)
        pytest.skip("demo generator produced an empty train split")
    yield train_path
    shutil.rmtree(temp_dir, ignore_errors=True)


def degraded_fraction(dataset, plain_pixels: np.ndarray, trials: int) -> float:
    degraded = 0
    for _ in range(trials):
        arr = np.array(dataset[0]["image"])
        if arr.shape != plain_pixels.shape or not np.array_equal(arr, plain_pixels):
            degraded += 1
    return degraded / trials


def test_augment_probability_gate(demo_data):
    plain = SciImageTableDataset(data_path=demo_data)
    plain_pixels = np.array(plain[0]["image"])

    never = SciImageTableDataset(data_path=demo_data, augment=True, augment_probability=0.0)
    assert np.array_equal(np.array(never[0]["image"]), plain_pixels)
    assert degraded_fraction(never, plain_pixels, 10) == 0.0

    always = SciImageTableDataset(data_path=demo_data, augment=True, augment_probability=1.0)
    assert degraded_fraction(always, plain_pixels, 10) == 1.0
    assert isinstance(always[0]["image"], Image.Image)

    gated = SciImageTableDataset(data_path=demo_data, augment=True, augment_probability=0.75)
    assert isinstance(gated[0]["image"], Image.Image)
    fraction = degraded_fraction(gated, plain_pixels, TRIALS)
    assert abs(fraction - 0.75) < FRACTION_TOL


def test_default_probability_ungated(demo_data):
    plain = SciImageTableDataset(data_path=demo_data)
    plain_pixels = np.array(plain[0]["image"])
    default = SciImageTableDataset(data_path=demo_data, augment=True)
    # augment_probability defaults to 1.0: the pipeline fires on every access.
    assert degraded_fraction(default, plain_pixels, 10) == 1.0


def test_plain_dataset_without_augment_is_static(demo_data):
    plain = SciImageTableDataset(data_path=demo_data)
    pixels_lot1 = np.array(plain[0]["image"])
    pixels_lot2 = np.array(plain[0]["image"])
    assert np.array_equal(pixels_lot1, pixels_lot2)


def test_augment_probability_range_validation(demo_data):
    with pytest.raises(ValueError, match="augment_probability"):
        SciImageTableDataset(data_path=demo_data, augment=True, augment_probability=1.5)
    with pytest.raises(ValueError, match="augment_probability"):
        SciImageTableDataset(data_path=demo_data, augment=True, augment_probability=-0.1)


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
