"""Tests for HUD overlay drawing and main app loop with fake sources."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tests.test_swipe_actions import FakeBackend

from tghci.app import run_app
from tghci.overlay import OverlayState, draw_overlay
from tghci.tracker import HandDetection


class _FakeTracker:
    def detect(self, bgr_frame: np.ndarray, timestamp_ms: int | None = None):
        pts = np.zeros((21, 3), dtype=np.float32)
        pts[9] = np.array([0.0, -0.2, 0.0], dtype=np.float32)
        return [HandDetection(landmarks=pts, handedness="Right", confidence=0.95)]

    def close(self) -> None:
        return None


class _FakeClassifier:
    def predict(self, features: np.ndarray) -> tuple[str, float]:
        return "open_palm", 0.92


def test_draw_overlay_on_blank_frame() -> None:
    """Overlay drawing modifies a blank frame and preserves shape/dtype."""
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    pts = np.full((21, 3), 0.5, dtype=np.float32)
    state = OverlayState(
        fps=30.0,
        gesture="open_palm",
        confidence=0.95,
        armed=False,
        dry_run=True,
        mode="media",
        last_action="none",
        device="cpu",
    )
    out = draw_overlay(blank, state, landmarks=pts)
    assert out.shape == blank.shape
    assert out.dtype == np.uint8
    assert int(out.sum()) > 0


def test_app_loop_with_fake_frame_source() -> None:
    """App loop runs cleanly over a fake frame source and exits with code 0."""
    frames = iter([np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(5)])
    rc = run_app(
        device_name="cpu",
        frame_source=frames,
        tracker=_FakeTracker(),
        classifier=_FakeClassifier(),
        backend=FakeBackend(),
        display=False,
        max_frames=5,
    )
    assert rc == 0


def test_app_missing_checkpoint_returns_nonzero(tmp_path: Path) -> None:
    """Missing checkpoint must return 1 cleanly without raising."""
    rc = run_app(
        device_name="cpu",
        checkpoint_path=tmp_path / "nonexistent.pt",
        display=False,
    )
    assert rc == 1
