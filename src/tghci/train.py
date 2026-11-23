"""Deterministic training CLI for the gesture classifier MLP."""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tghci.config import load_config
from tghci.dataset import LandmarkDataset, load_sessions, split_by_session, summarize_sessions
from tghci.model import GestureMLP, build_model, save_checkpoint


def set_seed(seed: int) -> None:
    """Set deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_commit_hash() -> str:
    """Return current HEAD commit hash or 'unknown' if unavailable."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def compute_class_weights(targets: np.ndarray, num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced training splits."""
    counts = np.bincount(targets, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def resolve_device(requested: str) -> torch.device:
    """Resolve requested device ('cuda' or 'cpu'), raising if cuda is unavailable."""
    req = requested.lower()
    if req == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested (--device cuda) but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if req == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device={requested!r}; use 'cuda' or 'cpu'.")


def train_model(
    config_path: str | Path | None = None,
    device_name: str = "cuda",
    seed: int | None = None,
    data_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    log_csv_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train ``GestureMLP`` with session-based splits and early stopping."""
    cfg = load_config(config_path)
    train_cfg = cfg.get("training", {})
    feat_cfg = cfg.get("features", {})
    labels: list[str] = list(cfg["labels"])

    effective_seed = int(train_cfg.get("seed", 42) if seed is None else seed)
    set_seed(effective_seed)
    device = resolve_device(device_name)

    raw_dir = Path(data_dir) if data_dir is not None else Path(cfg["paths"]["raw_data_dir"])
    ckpt_out = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else Path(cfg["paths"]["checkpoint_path"])
    )
    log_out = (
        Path(log_csv_path) if log_csv_path is not None else Path(cfg["paths"]["train_log_csv"])
    )

    df = load_sessions(raw_dir, labels)
    summary = summarize_sessions(df)
    include_aux = bool(feat_cfg.get("include_auxiliary", True))
    splits = split_by_session(
        df,
        labels=labels,
        val_fraction=float(train_cfg.get("val_fraction", 0.2)),
        test_fraction=float(train_cfg.get("test_fraction", 0.2)),
        seed=effective_seed,
        include_auxiliary=include_aux,
    )

    train_ds = LandmarkDataset(
        splits.X_train,
        splits.y_train,
        mirror_augment=bool(train_cfg.get("mirror_augment", True)),
        jitter_std=float(train_cfg.get("jitter_std", 0.01)),
        seed=effective_seed,
    )
    val_ds = LandmarkDataset(splits.X_val, splits.y_val, mirror_augment=False, jitter_std=0.0)

    batch_size = int(train_cfg.get("batch_size", 32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model: GestureMLP = build_model(cfg, input_dim=splits.X_train.shape[1]).to(device)
    weights = compute_class_weights(splits.y_train, num_classes=len(labels)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    epochs = int(train_cfg.get("epochs", 80))
    patience = int(train_cfg.get("early_stopping_patience", 12))
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_state: dict[str, torch.Tensor] | None = None
    no_improve = 0
    history: list[dict[str, object]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(yb)
            train_count += len(yb)

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_count = 0
        with torch.inference_mode():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss_sum += float(loss.item()) * len(yb)
                preds = torch.argmax(logits, dim=-1)
                val_correct += int((preds == yb).sum().item())
                val_count += len(yb)

        train_loss = train_loss_sum / max(train_count, 1)
        val_loss = val_loss_sum / max(val_count, 1)
        val_acc = val_correct / max(val_count, 1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_acc": f"{val_acc:.4f}",
            }
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    log_out.parent.mkdir(parents=True, exist_ok=True)
    with log_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(history)

    metadata: dict[str, Any] = {
        "git_commit": _git_commit_hash(),
        "python_version": sys.version.split()[0],
        "torch_version": str(torch.__version__),
        "device": str(device),
        "seed": effective_seed,
        "hyperparameters": {
            "hidden_dims": model.hidden_dims,
            "dropout": model.dropout,
            "batch_size": batch_size,
            "learning_rate": float(train_cfg.get("learning_rate", 1e-3)),
            "weight_decay": float(train_cfg.get("weight_decay", 1e-4)),
            "epochs_run": len(history),
        },
        "data_summary": {
            "total_rows": summary.total_rows,
            "num_sessions": summary.num_sessions,
            "class_counts": summary.class_counts,
            "train_sessions": list(splits.train_sessions),
            "val_sessions": list(splits.val_sessions),
            "test_sessions": list(splits.test_sessions),
        },
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
    }
    save_checkpoint(ckpt_out, model, labels, metadata)
    return metadata


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for `python -m tghci.train`."""
    parser = argparse.ArgumentParser(description="Train gesture classifier MLP")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--data-dir", type=str, default=None, help="Override raw data directory")
    parser.add_argument("--output", type=str, default=None, help="Override checkpoint path")
    parser.add_argument("--log-csv", type=str, default=None, help="Override training log CSV path")
    args = parser.parse_args(argv)

    meta = train_model(
        config_path=args.config,
        device_name=args.device,
        seed=args.seed,
        data_dir=args.data_dir,
        checkpoint_path=args.output,
        log_csv_path=args.log_csv,
    )
    print(
        f"Training complete on {meta['device']} | epochs={meta['hyperparameters']['epochs_run']} "
        f"| best_val_loss={meta['best_val_loss']:.4f} | best_val_acc={meta['best_val_acc']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
