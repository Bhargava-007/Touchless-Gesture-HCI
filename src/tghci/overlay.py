"""Pure OpenCV HUD overlay and hand landmark drawing utilities."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


@dataclass(frozen=True)
class OverlayState:
    """Snapshot of runtime status rendered on the HUD."""

    fps: float
    gesture: str
    confidence: float
    armed: bool
    dry_run: bool
    mode: str
    last_action: str
    device: str


def draw_landmarks(frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Draw 21 normalised (x, y, z) hand landmarks and connections onto ``frame``."""
    h, w = frame.shape[:2]
    pts = [(int(lm[0] * w), int(lm[1] * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (180, 180, 180), 2)
    for idx, (px, py) in enumerate(pts):
        color = (0, 255, 255) if idx in (4, 8, 12, 16, 20) else (0, 200, 0)
        cv2.circle(frame, (px, py), 4, color, -1)
    return frame


def draw_overlay(
    frame: np.ndarray,
    state: OverlayState,
    landmarks: np.ndarray | None = None,
) -> np.ndarray:
    """Render landmarks and HUD status onto a copy of ``frame``."""
    out = frame.copy()
    if landmarks is not None and landmarks.shape == (21, 3):
        draw_landmarks(out, landmarks)

    cv2.rectangle(out, (8, 8), (430, 145), (20, 20, 20), -1)
    arm_text = "ARMED" if state.armed else "DISARMED"
    arm_color = (0, 220, 0) if state.armed else (0, 140, 255)
    dry_text = "DRY-RUN" if state.dry_run else "LIVE"

    lines = [
        (f"FPS: {state.fps:.1f} | Device: {state.device}", (255, 255, 255)),
        (f"Gesture: {state.gesture} ({state.confidence:.2f})", (0, 255, 200)),
        (f"Safety: {arm_text} | {dry_text} | Mode: {state.mode}", arm_color),
        (f"Last Action: {state.last_action}", (220, 220, 220)),
        ("Keys: [q] quit [a] arm [d] dry-run [m] mode", (170, 170, 170)),
    ]
    for idx, (text, color) in enumerate(lines):
        cv2.putText(
            out,
            text,
            (16, 30 + idx * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )
    return out
