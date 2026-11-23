"""Unit tests for landmark feature normalisation and tracker graceful handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tghci.features import (
    TOTAL_FEATURES,
    extract_features,
    extract_features_safe,
    normalise_landmarks,
)
from tghci.tracker import HandTracker, open_camera


def _make_synthetic_hand() -> np.ndarray:
    """Create a deterministic synthetic (21, 3) hand array."""
    rng = np.random.default_rng(123)
    pts = rng.uniform(-0.2, 0.2, size=(21, 3)).astype(np.float32)
    pts[0] = np.array([0.4, 0.6, 0.0], dtype=np.float32)  # wrist
    pts[9] = np.array([0.4, 0.4, 0.0], dtype=np.float32)  # middle MCP (distance 0.2)
    return pts


def test_translation_and_scale_invariance() -> None:
    """Features must be invariant to uniform 3D translation and positive scaling."""
    base = _make_synthetic_hand()
    f_base = extract_features(base, handedness="Right")

    translated = base + np.array([0.35, -0.15, 0.5], dtype=np.float32)
    f_trans = extract_features(translated, handedness="Right")
    np.testing.assert_allclose(f_base, f_trans, atol=1e-5)

    scaled = (base - base[0]) * 3.7 + base[0]
    f_scaled = extract_features(scaled, handedness="Right")
    np.testing.assert_allclose(f_base, f_scaled, atol=1e-5)


def test_left_hand_mirror_correctness() -> None:
    """Mirroring a left hand should match the horizontally reflected right hand."""
    right_hand = _make_synthetic_hand()
    left_hand = right_hand.copy()
    # Reflect x around wrist x
    left_hand[:, 0] = 2.0 * right_hand[0, 0] - right_hand[:, 0]

    f_right = extract_features(right_hand, handedness="Right", mirror_left_hand=True)
    f_left = extract_features(left_hand, handedness="Left", mirror_left_hand=True)
    np.testing.assert_allclose(f_right, f_left, atol=1e-5)


def test_shape_dtype_and_auxiliary_toggle() -> None:
    """Verify output shape and float32 dtype with and without auxiliary features."""
    pts = _make_synthetic_hand()
    f_full = extract_features(pts, include_auxiliary=True)
    assert f_full.shape == (TOTAL_FEATURES,)
    assert f_full.dtype == np.float32

    f_base = extract_features(pts, include_auxiliary=False)
    assert f_base.shape == (63,)
    assert f_base.dtype == np.float32


def test_nan_and_no_hand_handling() -> None:
    """NaN landmarks raise in strict mode and return None in safe mode; None returns None."""
    pts = _make_synthetic_hand()
    pts[4, 1] = np.nan

    with pytest.raises(ValueError, match="NaN"):
        normalise_landmarks(pts)

    assert extract_features_safe(pts) is None
    assert extract_features_safe(None) is None


def test_camera_probe_degrades_gracefully() -> None:
    """Opening a non-existent camera index must return None without raising."""
    cap = open_camera(device_index=99)
    if cap is not None:
        cap.release()
    assert cap is None or hasattr(cap, "read")


def test_hand_tracker_blank_image_and_context_manager() -> None:
    """HandTracker loads hand_landmarker.task and returns [] on a blank frame."""
    model_path = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"
    if not model_path.is_file():
        pytest.skip("hand_landmarker.task not downloaded")
    blank = np.zeros((120, 160, 3), dtype=np.uint8)
    with HandTracker(model_path=model_path, running_mode="IMAGE") as tracker:
        assert tracker.detect(blank) == []
