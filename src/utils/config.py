"""
Configuration management utilities.
"""

import os
from typing import Any, Dict, Iterable, Optional
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_overrides(overrides: Iterable[str]) -> Dict[str, Any]:
    """Turns 'a.b=value' strings into a nested dict suitable for merge_configs.

    Values are parsed as YAML scalars, so 1024 becomes an int, 2.0e-4 a float,
    and true a bool. Anything unparseable is kept as a string.
    """
    result: Dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Malformed override {item!r}; expected key.path=value")
        path, raw = item.split("=", 1)
        keys = [k for k in path.strip().split(".") if k]
        if not keys:
            raise ValueError(f"Malformed override {item!r}; empty key path")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw

        node = result
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
    return result


def merge_configs(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Deep merge two dictionaries, where override takes precedence."""
    if not override:
        return base.copy()
    
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result
