"""PyTorch MLP gesture classifier and checkpoint inference helper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from tghci.features import TOTAL_FEATURES


class GestureMLP(nn.Module):
    """Configurable Multi-Layer Perceptron for normalised hand landmark features."""

    def __init__(
        self,
        input_dim: int = TOTAL_FEATURES,
        hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        num_classes: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.num_classes = num_classes
        self.dropout = float(dropout)

        layers: list[nn.Module] = []
        in_features = input_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(in_features, int(h_dim)))
            layers.append(nn.ReLU())
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            in_features = int(h_dim)
        layers.append(nn.Linear(in_features, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw class logits for a batch of feature vectors."""
        return self.net(x)


def build_model(cfg: dict[str, Any], input_dim: int | None = None) -> GestureMLP:
    """Construct ``GestureMLP`` from a parsed YAML config dictionary."""
    labels = list(cfg["labels"])
    model_cfg = cfg.get("model", {})
    feat_cfg = cfg.get("features", {})
    if input_dim is None:
        base = int(feat_cfg.get("num_base_features", 63))
        aux = int(feat_cfg.get("num_aux_features", 5)) if feat_cfg.get("include_auxiliary", True) else 0
        input_dim = base + aux

    return GestureMLP(
        input_dim=input_dim,
        hidden_dims=list(model_cfg.get("hidden_dims", [128, 64])),
        num_classes=len(labels),
        dropout=float(model_cfg.get("dropout", 0.2)),
    )


def save_checkpoint(
    path: str | Path,
    model: GestureMLP,
    labels: list[str],
    metadata: dict[str, Any],
) -> Path:
    """Save model weights, architecture config, labels, and sidecar metadata JSON."""
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    clean_metadata = json.loads(json.dumps(metadata))
    payload = {
        "state_dict": model.state_dict(),
        "input_dim": int(model.input_dim),
        "hidden_dims": [int(h) for h in model.hidden_dims],
        "num_classes": int(model.num_classes),
        "dropout": float(model.dropout),
        "labels": [str(lbl) for lbl in labels],
        "metadata": clean_metadata,
    }
    torch.save(payload, ckpt_path)

    meta_path = ckpt_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(clean_metadata, indent=2), encoding="utf-8")
    return ckpt_path


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[GestureMLP, list[str], dict[str, Any]]:
    """Load a saved ``GestureMLP`` checkpoint onto ``device``."""
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = GestureMLP(
        input_dim=int(payload["input_dim"]),
        hidden_dims=list(payload["hidden_dims"]),
        num_classes=int(payload["num_classes"]),
        dropout=float(payload["dropout"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, list(payload["labels"]), dict(payload.get("metadata", {}))


class GestureClassifier:
    """Inference helper wrapping a trained ``GestureMLP`` checkpoint."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model, self.labels, self.metadata = load_checkpoint(
            checkpoint_path, device=self.device
        )

    @torch.inference_mode()
    def predict(self, features: np.ndarray) -> tuple[str, float]:
        """Classify a single 1D feature vector into ``(label, confidence)``."""
        tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=-1)[0]
        idx = int(torch.argmax(probs).item())
        return self.labels[idx], float(probs[idx].item())
