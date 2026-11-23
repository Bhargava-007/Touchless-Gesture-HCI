"""MediaPipe Tasks HandLandmarker wrapper for image and video/camera frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


@dataclass(frozen=True)
class HandDetection:
    """Single detected hand with 21x3 normalised image-space landmarks."""

    landmarks: np.ndarray  # shape (21, 3), dtype float32
    handedness: str  # "Left" or "Right"
    confidence: float


class HandTracker:
    """Wrapper around MediaPipe ``HandLandmarker`` with clean resource lifecycle.

    Supports ``VIDEO`` mode (default for sequential webcam/video frames with
    monotonic timestamps) and ``IMAGE`` mode (for independent still frames).
    """

    def __init__(
        self,
        model_path: str | Path,
        num_hands: int = 1,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        running_mode: str = "VIDEO",
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Hand landmarker model not found: {self.model_path}. "
                "Run `python scripts/download_hand_model.py` first."
            )

        mode_upper = running_mode.upper()
        if mode_upper == "VIDEO":
            mp_mode = mp_vision.RunningMode.VIDEO
        elif mode_upper == "IMAGE":
            mp_mode = mp_vision.RunningMode.IMAGE
        else:
            raise ValueError(f"Unsupported running_mode={running_mode!r}; use 'VIDEO' or 'IMAGE'.")

        self.running_mode = mode_upper
        base_options = mp_python.BaseOptions(model_asset_path=str(self.model_path))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_mode,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def __enter__(self) -> HandTracker:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def close(self) -> None:
        """Release underlying MediaPipe resources."""
        if getattr(self, "_landmarker", None) is not None:
            self._landmarker.close()
            self._landmarker = None  # type: ignore[assignment]

    def detect(self, bgr_frame: np.ndarray, timestamp_ms: int | None = None) -> list[HandDetection]:
        """Detect hands in an OpenCV BGR frame.

        Args:
            bgr_frame: Input image in BGR uint8 format (H, W, 3).
            timestamp_ms: Monotonic timestamp in milliseconds (required for VIDEO mode;
                auto-incremented if omitted).

        Returns:
            List of ``HandDetection`` objects sorted by MediaPipe output order.
        """
        if self._landmarker is None:
            raise RuntimeError("HandTracker has already been closed.")
        if bgr_frame is None or bgr_frame.size == 0:
            return []

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self.running_mode == "VIDEO":
            ts = (self._last_timestamp_ms + 33) if timestamp_ms is None else int(timestamp_ms)
            if ts <= self._last_timestamp_ms:
                ts = self._last_timestamp_ms + 1
            self._last_timestamp_ms = ts
            result = self._landmarker.detect_for_video(mp_image, ts)
        else:
            result = self._landmarker.detect(mp_image)

        detections: list[HandDetection] = []
        if not result.hand_landmarks:
            return detections

        for idx, lm_list in enumerate(result.hand_landmarks):
            coords = np.array([[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float32)
            hand_label = "Right"
            conf = 0.0
            if result.handedness and idx < len(result.handedness) and result.handedness[idx]:
                cat = result.handedness[idx][0]
                hand_label = getattr(cat, "category_name", "Right") or "Right"
                conf = float(getattr(cat, "score", 0.0))
            detections.append(
                HandDetection(landmarks=coords, handedness=hand_label, confidence=conf)
            )
        return detections


def open_camera(device_index: int = 0, width: int = 640, height: int = 480) -> Any | None:
    """Open an OpenCV VideoCapture gracefully, returning None if unavailable."""
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap
