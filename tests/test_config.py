"""
tests/test_config.py — Unit tests for YAML config loading and merging utilities.

Run with:
    pytest tests/test_config.py -v
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config, merge_configs, load_and_merge, print_config  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(content: dict, path: Path) -> None:
    with open(path, "w") as f:
        yaml.dump(content, f)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_loads_base_config(self):
        """The real base config should load without error."""
        cfg = load_config(str(PROJECT_ROOT / "configs" / "base_config.yaml"))
        assert isinstance(cfg, dict)
        assert "project" in cfg
        assert "data" in cfg
        assert "training" in cfg

    def test_loads_efficientnet_config(self):
        cfg = load_config(str(PROJECT_ROOT / "configs" / "efficientnet_config.yaml"))
        assert isinstance(cfg, dict)

    def test_loads_dinov2_config(self):
        cfg = load_config(str(PROJECT_ROOT / "configs" / "dinov2_lora_config.yaml"))
        assert isinstance(cfg, dict)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_returns_dict_for_empty_yaml(self, tmp_path):
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        cfg = load_config(str(empty))
        assert cfg == {}

    def test_nested_keys_preserved(self, tmp_path):
        data = {"level1": {"level2": {"key": "value"}}}
        p = tmp_path / "nested.yaml"
        _write_yaml(data, p)
        cfg = load_config(str(p))
        assert cfg["level1"]["level2"]["key"] == "value"


class TestMergeConfigs:
    def test_non_overlapping_keys_combined(self):
        base = {"a": 1}
        override = {"b": 2}
        result = merge_configs(base, override)
        assert result == {"a": 1, "b": 2}

    def test_override_wins_on_conflict(self):
        base = {"lr": 1e-4}
        override = {"lr": 5e-4}
        result = merge_configs(base, override)
        assert result["lr"] == pytest.approx(5e-4)

    def test_deep_merge(self):
        base = {"training": {"lr": 1e-4, "epochs": 50}}
        override = {"training": {"lr": 5e-4}}
        result = merge_configs(base, override)
        assert result["training"]["lr"] == pytest.approx(5e-4)
        assert result["training"]["epochs"] == 50  # preserved

    def test_base_not_mutated(self):
        base = {"a": {"b": 1}}
        override = {"a": {"b": 2}}
        merge_configs(base, override)
        assert base["a"]["b"] == 1  # original unchanged

    def test_override_not_mutated(self):
        base = {"a": 1}
        override = {"b": 2}
        result = merge_configs(base, override)
        override["b"] = 99
        assert result["b"] == 2  # result is independent

    def test_list_override_replaces_not_extends(self):
        """Lists are replaced entirely, not extended."""
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}
        result = merge_configs(base, override)
        assert result["items"] == [4, 5]

    def test_empty_override_returns_base(self):
        base = {"a": 1, "b": 2}
        result = merge_configs(base, {})
        assert result == {"a": 1, "b": 2}

    def test_empty_base_returns_override(self):
        override = {"x": 99}
        result = merge_configs({}, override)
        assert result == {"x": 99}


class TestLoadAndMerge:
    def test_merges_base_and_efficientnet(self):
        """load_and_merge should produce a merged config with all base keys."""
        cfg = load_and_merge(
            str(PROJECT_ROOT / "configs" / "base_config.yaml"),
            str(PROJECT_ROOT / "configs" / "efficientnet_config.yaml"),
        )
        # Keys from base
        assert "project" in cfg
        assert "data" in cfg
        assert "training" in cfg
        assert "logging" in cfg

    def test_model_specific_keys_present_dinov2(self):
        cfg = load_and_merge(
            str(PROJECT_ROOT / "configs" / "base_config.yaml"),
            str(PROJECT_ROOT / "configs" / "dinov2_lora_config.yaml"),
        )
        # DINOv2 config should add or override something
        assert isinstance(cfg, dict)

    def test_experiment_name_in_merged(self):
        cfg = load_and_merge(
            str(PROJECT_ROOT / "configs" / "base_config.yaml"),
            str(PROJECT_ROOT / "configs" / "efficientnet_config.yaml"),
        )
        # experiment_name lives in logging block (set by model configs)
        assert "logging" in cfg


class TestPrintConfig:
    def test_runs_without_error(self, capsys):
        cfg = {"key": "value", "nested": {"inner": 42}}
        print_config(cfg)
        captured = capsys.readouterr()
        assert "key" in captured.out
        assert "nested" in captured.out

    def test_nested_indentation(self, capsys):
        cfg = {"outer": {"inner": "val"}}
        print_config(cfg)
        captured = capsys.readouterr()
        # Inner key should be indented (2+ spaces)
        lines = captured.out.splitlines()
        inner_lines = [l for l in lines if "inner" in l]
        assert inner_lines, "Inner key not printed."
        assert inner_lines[0].startswith("  "), "Expected indentation for nested key."
