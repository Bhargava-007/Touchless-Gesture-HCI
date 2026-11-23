"""Synthetic smoke tests for GestureMLP, training on CPU and CUDA, and inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tghci.collect import FEATURE_COLUMNS, write_session_csv
from tghci.model import GestureClassifier
from tghci.train import train_model

LABELS = ["open_palm", "fist", "pinch", "neutral"]


def _write_synthetic_dataset(root: Path, num_sessions: int = 4, rows_per_class: int = 6) -> None:
    """Write clearly labelled synthetic session CSVs into ``root``."""
    rng = np.random.default_rng(99)
    for s_idx in range(num_sessions):
        rows: list[dict[str, object]] = []
        for cls_idx, label in enumerate(LABELS):
            for r in range(rows_per_class):
                row: dict[str, object] = {
                    "session_id": f"synth_smoke_s{s_idx}",
                    "timestamp": f"{100.0 + r:.2f}",
                    "label": label,
                    "handedness": "Right",
                    "confidence": "0.99",
                }
                vals = rng.normal(loc=float(cls_idx), scale=0.25, size=len(FEATURE_COLUMNS))
                for col, val in zip(FEATURE_COLUMNS, vals, strict=True):
                    row[col] = f"{float(val):.6f}"
                rows.append(row)
        write_session_csv(root / f"synth_smoke_s{s_idx}.csv", rows)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
def test_synthetic_training_and_inference_smoke(tmp_path: Path, device: str) -> None:
    """Synthetic smoke test: train end-to-end on CPU/CUDA and classify one vector."""
    data_dir = tmp_path / "raw"
    _write_synthetic_dataset(data_dir)
    ckpt = tmp_path / "gesture_mlp.pt"
    log_csv = tmp_path / "train_log.csv"

    meta = train_model(
        device_name=device,
        seed=7,
        data_dir=data_dir,
        checkpoint_path=ckpt,
        log_csv_path=log_csv,
    )
    assert ckpt.is_file()
    assert ckpt.with_suffix(".meta.json").is_file()
    assert log_csv.is_file()
    assert meta["device"] == device

    clf = GestureClassifier(ckpt, device=device)
    label, conf = clf.predict(np.zeros(68, dtype=np.float32))
    assert label in LABELS
    assert 0.0 <= conf <= 1.0
