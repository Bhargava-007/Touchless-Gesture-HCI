"""Tests for latency benchmark CSV output and held-out evaluation code path."""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import MagicMock

from tests.test_classifier import _write_synthetic_dataset

from tghci.benchmark import _git_commit_hash, run_benchmark
from tghci.evaluate import evaluate_model
from tghci.train import train_model


def test_benchmark_writes_csv(tmp_path: Path) -> None:
    """Benchmark runs on CPU (and CUDA if available) and writes expected CSV columns."""
    csv_path = run_benchmark(
        iterations=10,
        warmup=2,
        batch_sizes=(1, 8),
        output_dir=tmp_path,
    )
    assert csv_path.is_file()
    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) >= 2
    assert "mean_ms" in rows[0]
    assert rows[0]["weights_note"] == "random_init"


def test_evaluate_synthetic_code_path(tmp_path: Path) -> None:
    """Evaluate code path produces CSV, metadata JSON, and confusion matrix PNGs."""
    data_dir = tmp_path / "raw"
    _write_synthetic_dataset(data_dir, num_sessions=4, rows_per_class=4)
    ckpt = tmp_path / "gesture_mlp.pt"
    log_csv = tmp_path / "train_log.csv"
    train_model(
        device_name="cpu",
        seed=42,
        data_dir=data_dir,
        checkpoint_path=ckpt,
        log_csv_path=log_csv,
    )

    res_dir = tmp_path / "results"
    fig_dir = tmp_path / "figures"
    meta = evaluate_model(
        checkpoint_path=ckpt,
        data_dir=data_dir,
        results_dir=res_dir,
        figures_dir=fig_dir,
        device_name="cpu",
        seed=42,
    )
    assert Path(meta["eval_csv"]).is_file()
    assert Path(meta["confusion_counts_png"]).is_file()
    assert Path(meta["confusion_norm_png"]).is_file()


def test_git_commit_hash_flags_dirty_tree(monkeypatch) -> None:
    """_git_commit_hash appends -dirty when git status --porcelain is non-empty."""
    def _fake_run(cmd, **_kwargs):
        mock = MagicMock()
        mock.stdout = "abc1234\n" if cmd[1] == "rev-parse" else " M file.py\n"
        return mock

    monkeypatch.setattr("tghci.benchmark.subprocess.run", _fake_run)
    assert _git_commit_hash() == "abc1234-dirty"
