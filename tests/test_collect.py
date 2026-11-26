"""Unit tests for collect.py landmark drawing and countdown-to-recording state machine."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import tghci.collect as collect_mod
from tghci.collect import run_collection
from tghci.tracker import HandDetection


class _FakeTracker:
    def __init__(self, confidences: list[float]) -> None:
        self._confidences = confidences
        self._idx = 0

    def detect(self, bgr_frame: np.ndarray, timestamp_ms: int | None = None):
        conf = self._confidences[min(self._idx, len(self._confidences) - 1)]
        self._idx += 1
        pts = np.zeros((21, 3), dtype=np.float32)
        pts[0] = np.array([0.5, 0.7, 0.0], dtype=np.float32)
        pts[9] = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        return [HandDetection(landmarks=pts, handedness="Right", confidence=conf)]

    def close(self) -> None:
        return None


def test_collect_countdown_transitions_to_recording_and_draws_landmarks(
    tmp_path: Path, monkeypatch
) -> None:
    """Pressing '1' counts down, transitions to recording, draws landmarks, and writes CSV."""
    draw_calls = 0
    orig_draw = collect_mod.draw_landmarks

    def _spy_draw(frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        nonlocal draw_calls
        draw_calls += 1
        return orig_draw(frame, landmarks)

    monkeypatch.setattr(collect_mod, "draw_landmarks", _spy_draw)

    # Frame 0 (t=0.0): press '1' -> starts 2.0s countdown (record_start_at = 2.0)
    # Frame 1 (t=1.0): still in countdown (recording=False, no row)
    # Frame 2 (t=2.5): countdown elapsed -> recording=True, high conf -> records 1 row
    # Frame 3 (t=3.0): low conf (0.2 < 0.6) -> skipped
    # Frame 4 (t=3.5): high conf -> records 1 row, then press 'q' to save and quit
    times = iter([0.0, 1.0, 2.5, 3.0, 3.5])
    keys = iter([ord("1"), 255, 255, 255, ord("q")])
    frames = iter([np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(5)])
    tracker = _FakeTracker([0.95, 0.95, 0.95, 0.20, 0.95])

    rc = run_collection(
        session_id="test_sess",
        min_confidence=0.6,
        countdown_seconds=2.0,
        frame_source=frames,
        tracker=tracker,
        raw_dir_override=tmp_path,
        display=False,
        key_fn=lambda: next(keys),
        time_fn=lambda: next(times),
    )

    assert rc == 0
    assert draw_calls == 5
    csv_path = tmp_path / "test_sess.csv"
    assert csv_path.is_file()
    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert all(r["label"] == "open_palm" for r in rows)


def test_collect_space_pauses_recording(tmp_path: Path) -> None:
    """Pressing Space stops recording so subsequent frames are not saved."""
    # Frame 0 (t=0.0): press '2' (fist, countdown 0.5s -> record_start_at = 0.5)
    # Frame 1 (t=1.0): countdown elapsed -> records 1 row, then press SPACE to pause
    # Frame 2 (t=1.5): paused -> does NOT record, press 'q'
    times = iter([0.0, 1.0, 1.5])
    keys = iter([ord("2"), ord(" "), ord("q")])
    frames = iter([np.zeros((240, 320, 3), dtype=np.uint8) for _ in range(3)])
    tracker = _FakeTracker([0.95, 0.95, 0.95])

    rc = run_collection(
        session_id="pause_sess",
        min_confidence=0.6,
        countdown_seconds=0.5,
        frame_source=frames,
        tracker=tracker,
        raw_dir_override=tmp_path,
        display=False,
        key_fn=lambda: next(keys),
        time_fn=lambda: next(times),
    )

    assert rc == 0
    with (tmp_path / "pause_sess.csv").open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["label"] == "fist"
