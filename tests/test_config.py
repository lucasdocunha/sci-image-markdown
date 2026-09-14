"""
Unit tests for configuration utilities.
"""

import pytest

from src.utils.config import load_config, merge_configs, parse_overrides


def test_load_default_config():
    cfg = load_config("configs/default.yaml")
    assert cfg["project_name"] == "sci-image-markdown"
    assert "model" in cfg
    assert "training" in cfg
    assert "peft" in cfg


def test_default_config_declares_visual_token_budget():
    """The resolution knob must exist and be shared by training and inference."""
    cfg = load_config("configs/default.yaml")
    assert isinstance(cfg["data"]["max_visual_tokens"], int)
    assert isinstance(cfg["data"]["min_visual_tokens"], int)
    assert cfg["data"]["min_visual_tokens"] <= cfg["data"]["max_visual_tokens"]


def test_parse_overrides_builds_nested_dict_with_typed_scalars():
    parsed = parse_overrides([
        "data.max_visual_tokens=752",
        "training.learning_rate=2.0e-4",
        "training.gradient_checkpointing=true",
        "experiment_name=b0_res768",
    ])
    assert parsed["data"]["max_visual_tokens"] == 752
    assert parsed["training"]["learning_rate"] == pytest.approx(2.0e-4)
    assert parsed["training"]["gradient_checkpointing"] is True
    assert parsed["experiment_name"] == "b0_res768"


def test_parse_overrides_merges_onto_base_config():
    cfg = merge_configs(
        load_config("configs/default.yaml"),
        parse_overrides(["data.max_visual_tokens=1337"]),
    )
    assert cfg["data"]["max_visual_tokens"] == 1337
    # Sibling keys must survive the override.
    assert "system_prompt" in cfg["data"]


@pytest.mark.parametrize("bad", ["nokeyvalue", "=orphan"])
def test_parse_overrides_rejects_malformed_input(bad):
    with pytest.raises(ValueError):
        parse_overrides([bad])


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
