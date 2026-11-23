"""Rule-based swipe detector over a sliding window of hand centre positions.

This module implements a transparent, config-driven heuristic (not a learned
model) for detecting horizontal left/right swipes independent of handedness.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SwipeSample:
    """Single hand-centre observation in normalised image coordinates [0, 1]."""

    x: float
    y: float
    timestamp: float


class SwipeDetector:
    """Sliding-window horizontal swipe heuristic."""

    def __init__(
        self,
        window_size: int = 8,
        min_travel_norm: float = 0.18,
        min_velocity_norm: float = 0.45,
        max_vertical_ratio: float = 0.6,
        refractory_seconds: float = 0.8,
    ) -> None:
        self.window_size = int(window_size)
        self.min_travel_norm = float(min_travel_norm)
        self.min_velocity_norm = float(min_velocity_norm)
        self.max_vertical_ratio = float(max_vertical_ratio)
        self.refractory_seconds = float(refractory_seconds)
        self._history: deque[SwipeSample] = deque(maxlen=self.window_size)
        self._last_trigger_time: float = -1e9

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> SwipeDetector:
        """Construct ``SwipeDetector`` from the ``swipe`` section of config."""
        sc = cfg.get("swipe", {})
        return cls(
            window_size=int(sc.get("window_size", 8)),
            min_travel_norm=float(sc.get("min_travel_norm", 0.18)),
            min_velocity_norm=float(sc.get("min_velocity_norm", 0.45)),
            max_vertical_ratio=float(sc.get("max_vertical_ratio", 0.6)),
            refractory_seconds=float(sc.get("refractory_seconds", 0.8)),
        )

    def reset(self) -> None:
        """Clear trajectory history (e.g. when hand tracking is lost)."""
        self._history.clear()

    def update(self, x: float | None, y: float | None, timestamp: float) -> str | None:
        """Add a hand-centre observation and return ``'swipe_left'``, ``'swipe_right'``, or None."""
        if x is None or y is None:
            self.reset()
            return None

        self._history.append(SwipeSample(x=float(x), y=float(y), timestamp=float(timestamp)))
        if len(self._history) < self.window_size:
            return None

        if (timestamp - self._last_trigger_time) < self.refractory_seconds:
            return None

        oldest = self._history[0]
        newest = self._history[-1]
        dt = newest.timestamp - oldest.timestamp
        if dt <= 1e-6:
            return None

        dx = newest.x - oldest.x
        dy = newest.y - oldest.y
        abs_dx = abs(dx)
        abs_dy = abs(dy)

        if abs_dx < self.min_travel_norm:
            return None
        if (abs_dy / abs_dx) > self.max_vertical_ratio:
            return None
        if (abs_dx / dt) < self.min_velocity_norm:
            return None

        self._last_trigger_time = float(timestamp)
        self._history.clear()
        return "swipe_right" if dx > 0 else "swipe_left"
