"""Unit tests for session-based dataset loading, validation, and splitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tghci.collect import FEATURE_COLUMNS, write_session_csv
from tghci.dataset import LandmarkDataset, load_sessions, split_by_session, summarize_sessions

LABELS = ["open_palm", "fist", "pinch", "neutral"]


def _write_synthetic_session(
    directory: Path,
    session_id: str,
    rows_per_class: int = 4,
    seed: int = 0,
) -> Path:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for label in LABELS:
        for i in range(rows_per_class):
            row: dict[str, object] = {
                "session_id": session_id,
                "timestamp": f"{1000.0 + i:.3f}",
                "label": label,
                "handedness": "Right",
                "confidence": "0.95",
            }
            vals = rng.normal(0.0, 0.5, size=len(FEATURE_COLUMNS))
            for col, val in zip(FEATURE_COLUMNS, vals, strict=True):
                row[col] = f"{float(val):.6f}"
            rows.append(row)
    return write_session_csv(directory / f"{session_id}.csv", rows)


def test_session_split_has_zero_leakage(tmp_path: Path) -> None:
    """Train, val, and test splits must contain disjoint session IDs."""
    for idx in range(5):
        _write_synthetic_session(tmp_path, f"synth_s{idx}", rows_per_class=3, seed=idx)

    df = load_sessions(tmp_path, LABELS)
    summary = summarize_sessions(df)
    assert summary.num_sessions == 5
    assert summary.total_rows == 60

    splits = split_by_session(df, LABELS, seed=42)
    train_set = set(splits.train_sessions)
    val_set = set(splits.val_sessions)
    test_set = set(splits.test_sessions)

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert len(train_set | val_set | test_set) == 5
    assert splits.X_train.shape[1] == 68


def test_fewer_than_three_sessions_raises(tmp_path: Path) -> None:
    """Splitting with fewer than 3 sessions must raise a clear ValueError."""
    _write_synthetic_session(tmp_path, "synth_s0", seed=0)
    _write_synthetic_session(tmp_path, "synth_s1", seed=1)

    df = load_sessions(tmp_path, LABELS)
    with pytest.raises(ValueError, match="At least 3 distinct recording sessions"):
        split_by_session(df, LABELS)


def test_schema_validation_rejects_bad_label(tmp_path: Path) -> None:
    """Unknown labels in session CSVs must raise ValueError."""
    csv_path = _write_synthetic_session(tmp_path, "synth_bad", seed=7)
    text = csv_path.read_text(encoding="utf-8").replace("open_palm", "unknown_pose", 1)
    csv_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="unknown labels"):
        load_sessions(tmp_path, LABELS)


def test_landmark_dataset_augmentation_preserves_shape(tmp_path: Path) -> None:
    """LandmarkDataset returns (68,) float32 features and scalar int64 targets."""
    for idx in range(3):
        _write_synthetic_session(tmp_path, f"synth_s{idx}", seed=idx)
    df = load_sessions(tmp_path, LABELS)
    splits = split_by_session(df, LABELS, seed=1)

    ds = LandmarkDataset(splits.X_train, splits.y_train, mirror_augment=True, jitter_std=0.01)
    x, y = ds[0]
    assert x.shape == (68,)
    assert y.ndim == 0
