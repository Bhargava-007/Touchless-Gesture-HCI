"""Held-out test split evaluation and confusion matrix generation."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from tghci.config import load_config
from tghci.dataset import load_sessions, split_by_session, summarize_sessions
from tghci.model import load_checkpoint

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _git_commit_hash() -> str:
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


def _save_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    out_path: Path,
    normalised: bool,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix (Normalised)" if normalised else "Confusion Matrix (Counts)",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm[i, j]:.2f}" if normalised else f"{int(cm[i, j])}"
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, text, ha="center", va="center", color=color)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


@torch.inference_mode()
def evaluate_model(
    config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
    figures_dir: str | Path | None = None,
    device_name: str = "cpu",
    seed: int | None = None,
) -> dict[str, Any]:
    """Evaluate a trained checkpoint on the held-out session test split."""
    cfg = load_config(config_path)
    labels: list[str] = list(cfg["labels"])
    train_cfg = cfg.get("training", {})
    feat_cfg = cfg.get("features", {})

    ckpt = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else Path(cfg["paths"]["checkpoint_path"])
    )
    raw_dir = Path(data_dir) if data_dir is not None else Path(cfg["paths"]["raw_data_dir"])
    res_dir = Path(results_dir) if results_dir is not None else Path(cfg["paths"]["results_dir"])
    fig_dir = Path(figures_dir) if figures_dir is not None else Path(cfg["paths"]["figures_dir"])

    eff_seed = int(train_cfg.get("seed", 42) if seed is None else seed)
    df = load_sessions(raw_dir, labels)
    summary = summarize_sessions(df)
    splits = split_by_session(
        df,
        labels=labels,
        val_fraction=float(train_cfg.get("val_fraction", 0.2)),
        test_fraction=float(train_cfg.get("test_fraction", 0.2)),
        seed=eff_seed,
        include_auxiliary=bool(feat_cfg.get("include_auxiliary", True)),
    )

    device = torch.device(device_name if (device_name == "cpu" or torch.cuda.is_available()) else "cpu")
    model, ckpt_labels, _ = load_checkpoint(ckpt, device=device)

    x_test = torch.from_numpy(splits.X_test).to(device)
    logits = model(x_test)
    preds = torch.argmax(logits, dim=-1).cpu().numpy()
    y_true = splits.y_test

    label_indices = list(range(len(ckpt_labels)))
    cm_counts = confusion_matrix(y_true, preds, labels=label_indices)
    row_sums = cm_counts.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_counts.astype(np.float64), np.maximum(row_sums, 1))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, preds, labels=label_indices, zero_division=0
    )
    overall_acc = float(accuracy_score(y_true, preds))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    res_dir.mkdir(parents=True, exist_ok=True)
    eval_csv = res_dir / f"eval_{stamp}.csv"

    rows: list[dict[str, Any]] = []
    for idx, name in enumerate(ckpt_labels):
        rows.append(
            {
                "gesture": name,
                "precision": f"{float(precision[idx]):.4f}",
                "recall": f"{float(recall[idx]):.4f}",
                "f1": f"{float(f1[idx]):.4f}",
                "support": int(support[idx]),
                "overall_accuracy": f"{overall_acc:.4f}",
            }
        )

    with eval_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    counts_png = _save_confusion_matrix(
        cm_counts, ckpt_labels, fig_dir / "confusion_counts.png", normalised=False
    )
    norm_png = _save_confusion_matrix(
        cm_norm, ckpt_labels, fig_dir / "confusion_norm.png", normalised=True
    )

    meta = {
        "git_commit": _git_commit_hash(),
        "timestamp": stamp,
        "checkpoint": str(ckpt),
        "overall_accuracy": overall_acc,
        "test_sessions": list(splits.test_sessions),
        "data_summary": {
            "total_rows": summary.total_rows,
            "num_sessions": summary.num_sessions,
            "class_counts": summary.class_counts,
        },
        "eval_csv": str(eval_csv),
        "confusion_counts_png": str(counts_png),
        "confusion_norm_png": str(norm_png),
    }
    eval_meta = eval_csv.with_suffix(".meta.json")
    eval_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for `python -m tghci.evaluate`."""
    parser = argparse.ArgumentParser(description="Evaluate GestureMLP on held-out test sessions")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint path")
    parser.add_argument("--data-dir", type=str, default=None, help="Override raw data directory")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    args = parser.parse_args(argv)

    meta = evaluate_model(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        device_name=args.device,
        seed=args.seed,
    )
    print(f"Evaluation complete | overall_accuracy={meta['overall_accuracy']:.4f} | csv={meta['eval_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
