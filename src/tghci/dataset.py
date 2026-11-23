"""Session-based dataset loading, validation, splitting, and PyTorch Dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from tghci.features import NUM_BASE_FEATURES, NUM_LANDMARKS, compute_auxiliary_features

FEATURE_COLUMNS = [f"f_{i}" for i in range(NUM_BASE_FEATURES)]
REQUIRED_COLUMNS = ["session_id", "timestamp", "label", "handedness", "confidence", *FEATURE_COLUMNS]


@dataclass(frozen=True)
class SessionSummary:
    """Summary of loaded sessions and class counts."""

    total_rows: int
    num_sessions: int
    session_ids: tuple[str, ...]
    class_counts: dict[str, int]


@dataclass(frozen=True)
class SplitData:
    """Session-based train, validation, and test splits."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_sessions: tuple[str, ...]
    val_sessions: tuple[str, ...]
    test_sessions: tuple[str, ...]


def load_sessions(
    raw_dir: str | Path,
    valid_labels: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Load and validate all session CSV files from ``raw_dir``.

    Raises:
        FileNotFoundError: If no CSV files exist in ``raw_dir``.
        ValueError: If required columns are missing, values are non-finite, or labels are invalid.
    """
    raw_path = Path(raw_dir)
    csv_files = sorted(raw_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No session CSV files found in {raw_path}.")

    frames: list[pd.DataFrame] = []
    valid_set = set(valid_labels)
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"CSV {csv_file.name} missing required columns: {missing}")
        if df.empty:
            continue
        unknown = set(df["label"].astype(str).unique()) - valid_set
        if unknown:
            raise ValueError(f"CSV {csv_file.name} contains unknown labels: {sorted(unknown)}")
        feat_vals = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        if not np.all(np.isfinite(feat_vals)):
            raise ValueError(f"CSV {csv_file.name} contains NaN or infinite feature values.")
        frames.append(df)

    if not frames:
        raise ValueError(f"All CSV files in {raw_path} were empty.")
    return pd.concat(frames, ignore_index=True)


def summarize_sessions(df: pd.DataFrame) -> SessionSummary:
    """Compute session and per-class sample counts."""
    sids = tuple(sorted(df["session_id"].astype(str).unique()))
    counts = {str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()}
    return SessionSummary(
        total_rows=int(len(df)),
        num_sessions=len(sids),
        session_ids=sids,
        class_counts=counts,
    )


def _append_auxiliary(base_features: np.ndarray) -> np.ndarray:
    """Append 5 auxiliary geometric features to (N, 63) normalised landmarks."""
    aux_rows = [
        compute_auxiliary_features(row.reshape(NUM_LANDMARKS, 3))
        for row in base_features
    ]
    aux = np.stack(aux_rows, axis=0).astype(np.float32)
    return np.concatenate([base_features.astype(np.float32), aux], axis=1)


def split_by_session(
    df: pd.DataFrame,
    labels: list[str] | tuple[str, ...],
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 42,
    include_auxiliary: bool = True,
) -> SplitData:
    """Split a validated dataframe strictly by ``session_id``.

    Raises:
        ValueError: If fewer than 3 distinct sessions exist.
    """
    session_ids = sorted(df["session_id"].astype(str).unique())
    n_sessions = len(session_ids)
    if n_sessions < 3:
        raise ValueError(
            f"At least 3 distinct recording sessions are required for a session-based "
            f"train/val/test split, found {n_sessions}. Collect more sessions first."
        )

    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(session_ids))

    n_test = max(1, int(round(n_sessions * test_fraction)))
    n_val = max(1, int(round(n_sessions * val_fraction)))
    if n_test + n_val >= n_sessions:
        n_test = 1
        n_val = 1

    test_sids = tuple(shuffled[:n_test])
    val_sids = tuple(shuffled[n_test : n_test + n_val])
    train_sids = tuple(shuffled[n_test + n_val :])

    label_to_idx = {name: i for i, name in enumerate(labels)}

    def _extract(sids: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
        sub = df[df["session_id"].astype(str).isin(sids)]
        x_base = sub[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        x = _append_auxiliary(x_base) if include_auxiliary else x_base
        y = sub["label"].map(label_to_idx).to_numpy(dtype=np.int64)
        return x, y

    x_train, y_train = _extract(train_sids)
    x_val, y_val = _extract(val_sids)
    x_test, y_test = _extract(test_sids)
    return SplitData(
        X_train=x_train,
        y_train=y_train,
        X_val=x_val,
        y_val=y_val,
        X_test=x_test,
        y_test=y_test,
        train_sessions=train_sids,
        val_sessions=val_sids,
        test_sessions=test_sids,
    )


class LandmarkDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset over normalised landmark features with optional augmentation."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        mirror_augment: bool = False,
        jitter_std: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.features = np.asarray(features, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.int64)
        self.mirror_augment = mirror_augment
        self.jitter_std = jitter_std
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return int(len(self.targets))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.features[idx].copy()
        if self.mirror_augment and self._rng.random() < 0.5:
            # Negate x coordinates of the 21x3 base landmarks (indices 0, 3, ..., 60)
            feat[:NUM_BASE_FEATURES:3] *= -1.0
        if self.jitter_std > 0.0:
            noise = self._rng.normal(0.0, self.jitter_std, size=feat.shape).astype(np.float32)
            feat = feat + noise
        return torch.from_numpy(feat), torch.tensor(self.targets[idx], dtype=torch.long)
