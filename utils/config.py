"""YAML configuration loader with deep merge support."""
from __future__ import annotations
import copy
import logging
from pathlib import Path
from typing import Any, Dict
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML config file and return a nested dict.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    logger.info("Config loaded from: %s", path)
    return cfg or {}


def merge_configs(base: Dict, override: Dict) -> Dict:
    """
    Recursively merge two config dicts (override wins on conflicts).

    Args:
        base:     Base configuration dict.
        override: Override configuration dict.

    Returns:
        Merged configuration dict.
    """
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = merge_configs(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def load_and_merge(base_path: str, model_path: str) -> Dict[str, Any]:
    """Load base config and model config then merge them."""
    base = load_config(base_path)
    model = load_config(model_path)
    merged = merge_configs(base, model)
    return merged


def print_config(cfg: Dict, indent: int = 0) -> None:
    """Pretty-print a nested config dict."""
    for key, val in cfg.items():
        prefix = "  " * indent
        if isinstance(val, dict):
            print(f"{prefix}{key}:")
            print_config(val, indent + 1)
        else:
            print(f"{prefix}{key}: {val}")
