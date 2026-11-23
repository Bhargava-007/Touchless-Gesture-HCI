"""Landmark normalisation and geometric feature extraction."""

from __future__ import annotations

import numpy as np

WRIST_IDX = 0
THUMB_TIP_IDX = 4
INDEX_TIP_IDX = 8
MIDDLE_MCP_IDX = 9
MIDDLE_TIP_IDX = 12
PINKY_TIP_IDX = 20

NUM_LANDMARKS = 21
NUM_BASE_FEATURES = NUM_LANDMARKS * 3
NUM_AUX_FEATURES = 5
TOTAL_FEATURES = NUM_BASE_FEATURES + NUM_AUX_FEATURES


def normalise_landmarks(
    landmarks: np.ndarray,
    handedness: str = "Right",
    mirror_left_hand: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    """Translate landmarks to wrist origin, optionally mirror left hands, and scale-normalise.

    Args:
        landmarks: Array of shape (21, 3) containing (x, y, z) coordinates.
        handedness: ``"Left"`` or ``"Right"``.
        mirror_left_hand: If True and ``handedness == "Left"``, negate x after
            wrist translation so left and right hands share a canonical frame.
        eps: Minimum scale floor to prevent division by zero.

    Returns:
        Normalised landmark array of shape (21, 3) and dtype float32.
    """
    arr = np.asarray(landmarks, dtype=np.float32)
    if arr.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"Expected landmarks shape (21, 3), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Landmarks contain NaN or infinite values.")

    # Translate so wrist (landmark 0) is the origin.
    centered = arr - arr[WRIST_IDX : WRIST_IDX + 1, :]

    # Mirror left hand across the x-axis around the wrist origin.
    if mirror_left_hand and handedness.strip().lower() == "left":
        centered = centered.copy()
        centered[:, 0] *= -1.0

    # Scale by palm reference length (wrist -> middle finger MCP).
    palm_scale = float(np.linalg.norm(centered[MIDDLE_MCP_IDX]))
    if palm_scale < eps:
        max_norm = float(np.max(np.linalg.norm(centered, axis=1)))
        palm_scale = max(max_norm, eps)

    return (centered / palm_scale).astype(np.float32)


def compute_auxiliary_features(norm_landmarks: np.ndarray) -> np.ndarray:
    """Compute 5 scale-normalised geometric distance features.

    Features:
        0: thumb tip (4) to index tip (8) distance (pinch cue)
        1: thumb tip (4) to middle tip (12) distance
        2: index tip (8) to wrist (0) distance
        3: middle tip (12) to wrist (0) distance
        4: pinky tip (20) to wrist (0) distance
    """
    thumb_index = float(np.linalg.norm(norm_landmarks[THUMB_TIP_IDX] - norm_landmarks[INDEX_TIP_IDX]))
    thumb_middle = float(
        np.linalg.norm(norm_landmarks[THUMB_TIP_IDX] - norm_landmarks[MIDDLE_TIP_IDX])
    )
    index_wrist = float(np.linalg.norm(norm_landmarks[INDEX_TIP_IDX]))
    middle_wrist = float(np.linalg.norm(norm_landmarks[MIDDLE_TIP_IDX]))
    pinky_wrist = float(np.linalg.norm(norm_landmarks[PINKY_TIP_IDX]))
    return np.array(
        [thumb_index, thumb_middle, index_wrist, middle_wrist, pinky_wrist],
        dtype=np.float32,
    )


def extract_features(
    landmarks: np.ndarray,
    handedness: str = "Right",
    mirror_left_hand: bool = True,
    include_auxiliary: bool = True,
) -> np.ndarray:
    """Extract a fixed-length float32 feature vector from (21, 3) landmarks."""
    norm = normalise_landmarks(
        landmarks,
        handedness=handedness,
        mirror_left_hand=mirror_left_hand,
    )
    flat = norm.reshape(-1)
    if not include_auxiliary:
        return flat.astype(np.float32)
    aux = compute_auxiliary_features(norm)
    return np.concatenate([flat, aux], axis=0).astype(np.float32)


def extract_features_safe(
    landmarks: np.ndarray | None,
    handedness: str = "Right",
    mirror_left_hand: bool = True,
    include_auxiliary: bool = True,
) -> np.ndarray | None:
    """Return feature vector or None when no hand is detected or landmarks are invalid."""
    if landmarks is None:
        return None
    arr = np.asarray(landmarks)
    if arr.shape != (NUM_LANDMARKS, 3) or not np.all(np.isfinite(arr)):
        return None
    return extract_features(
        arr,
        handedness=handedness,
        mirror_left_hand=mirror_left_hand,
        include_auxiliary=include_auxiliary,
    )
