"""
Unit tests for configuration utilities.
"""

from src.utils.config import load_config, merge_configs


def test_load_default_config():
    cfg = load_config("configs/default.yaml")
    assert cfg["project_name"] == "sci-image-markdown"
    assert "model" in cfg
    assert "training" in cfg
    assert "peft" in cfg


def test_merge_configs():
    base = {
        "model": {"name": "model-a", "type": "vlm"},
        "training": {"epochs": 3, "lr": 1e-4}
    }
    override = {
        "model": {"name": "model-b"},
        "training": {"lr": 5e-5}
    }
    merged = merge_configs(base, override)
    assert merged["model"]["name"] == "model-b"
    assert merged["model"]["type"] == "vlm"
    assert merged["training"]["epochs"] == 3
    assert merged["training"]["lr"] == 5e-5
