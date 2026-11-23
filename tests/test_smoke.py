"""Smoke tests for Phase 1 package structure and configuration loading."""

from __future__ import annotations

from pathlib import Path

from tghci import __version__
from tghci.config import load_config


def test_version_and_default_config() -> None:
    """Verify package version and default YAML config structure."""
    assert __version__ == "0.1.0"
    cfg = load_config()
    assert cfg["labels"] == ["open_palm", "fist", "pinch", "neutral"]
    assert Path(cfg["paths"]["hand_model"]).name == "hand_landmarker.task"
