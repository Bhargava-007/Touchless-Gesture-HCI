"""Safety-first action mapper with arm/disarm, debounce, cooldown, and dry-run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class ActionBackend(Protocol):
    """Abstract OS input backend so tests can inject a fake without OS side effects."""

    def press_key(self, key: str) -> None:
        """Press a single named key."""

    def move_cursor_norm(self, x_norm: float, y_norm: float) -> None:
        """Move cursor to normalised screen coordinates in [0, 1]."""

    def click(self) -> None:
        """Perform a left mouse click."""


class PyAutoGuiBackend:
    """Real OS input backend using ``pyautogui`` with failsafe enabled."""

    def __init__(self) -> None:
        import pyautogui

        pyautogui.FAILSAFE = True
        self._pag = pyautogui
        self._screen_w, self._screen_h = pyautogui.size()

    def press_key(self, key: str) -> None:
        self._pag.press(key)

    def move_cursor_norm(self, x_norm: float, y_norm: float) -> None:
        x_clamped = min(max(float(x_norm), 0.0), 1.0)
        y_clamped = min(max(float(y_norm), 0.0), 1.0)
        self._pag.moveTo(x_clamped * self._screen_w, y_clamped * self._screen_h)

    def click(self) -> None:
        self._pag.click()


@dataclass(frozen=True)
class ActionEvent:
    """Record of an action decision (executed or dry-run)."""

    action: str
    executed: bool
    reason: str


class ActionMapper:
    """Maps static gestures and swipes to OS actions with safety controls."""

    def __init__(
        self,
        backend: ActionBackend | None = None,
        start_armed: bool = False,
        dry_run: bool = True,
        debounce_frames: int = 4,
        cooldown_seconds: float = 0.75,
        confidence_threshold: float = 0.70,
        volume_step_threshold: float = 0.04,
    ) -> None:
        self.backend: ActionBackend = backend if backend is not None else PyAutoGuiBackend()
        self.armed = bool(start_armed)
        self.dry_run = bool(dry_run)
        self.debounce_frames = int(debounce_frames)
        self.cooldown_seconds = float(cooldown_seconds)
        self.confidence_threshold = float(confidence_threshold)
        self.volume_step_threshold = float(volume_step_threshold)
        self.mode = "media"  # "media" or "cursor"

        self._last_label: str | None = None
        self._streak = 0
        self._last_action_time: dict[str, float] = {}
        self._pinch_anchor_y: float | None = None
        self.last_action_label: str = "none"

    @classmethod
    def from_config(
        cls, cfg: dict[str, Any], backend: ActionBackend | None = None
    ) -> ActionMapper:
        """Construct ``ActionMapper`` from the ``actions`` section of config."""
        ac = cfg.get("actions", {})
        return cls(
            backend=backend,
            start_armed=bool(ac.get("start_armed", False)),
            dry_run=bool(ac.get("dry_run", True)),
            debounce_frames=int(ac.get("debounce_frames", 4)),
            cooldown_seconds=float(ac.get("cooldown_seconds", 0.75)),
            confidence_threshold=float(ac.get("confidence_threshold", 0.70)),
            volume_step_threshold=float(ac.get("volume_step_threshold", 0.04)),
        )

    def toggle_armed(self) -> bool:
        """Toggle armed state and reset gesture streak."""
        self.armed = not self.armed
        self._streak = 0
        self._pinch_anchor_y = None
        return self.armed

    def toggle_dry_run(self) -> bool:
        """Toggle dry-run logging mode."""
        self.dry_run = not self.dry_run
        return self.dry_run

    def toggle_mode(self) -> str:
        """Switch between 'media' and 'cursor' control modes."""
        self.mode = "cursor" if self.mode == "media" else "media"
        self._streak = 0
        self._pinch_anchor_y = None
        return self.mode

    def _can_fire(self, action_name: str, now: float) -> bool:
        last = self._last_action_time.get(action_name, -1e9)
        return (now - last) >= self.cooldown_seconds

    def _dispatch(self, action_name: str, now: float, fn: Any) -> ActionEvent:
        self._last_action_time[action_name] = now
        if self.dry_run:
            self.last_action_label = f"DRY:{action_name}"
            return ActionEvent(action=action_name, executed=False, reason="dry_run")
        fn()
        self.last_action_label = action_name
        return ActionEvent(action=action_name, executed=True, reason="executed")

    def step(
        self,
        hand_present: bool,
        gesture_label: str | None,
        confidence: float,
        timestamp: float,
        swipe_event: str | None = None,
        hand_y: float | None = None,
        index_tip_xy: tuple[float, float] | None = None,
    ) -> ActionEvent | None:
        """Process one frame's predictions and return an ``ActionEvent`` if triggered."""
        if not hand_present:
            self._last_label = None
            self._streak = 0
            self._pinch_anchor_y = None
            return None

        if not self.armed:
            self._streak = 0
            self._pinch_anchor_y = None
            return None

        # Cursor mode: move pointer from index fingertip and click on debounced pinch.
        if self.mode == "cursor":
            if index_tip_xy is not None and not self.dry_run:
                self.backend.move_cursor_norm(index_tip_xy[0], index_tip_xy[1])
            if gesture_label == "pinch" and confidence >= self.confidence_threshold:
                self._streak = self._streak + 1 if self._last_label == "pinch" else 1
                self._last_label = "pinch"
                if self._streak >= self.debounce_frames and self._can_fire("click", timestamp):
                    self._streak = 0
                    return self._dispatch("click", timestamp, self.backend.click)
            else:
                self._last_label = gesture_label
                self._streak = 0
            return None

        # Media mode: swipes take priority over static holds.
        if swipe_event in ("swipe_left", "swipe_right"):
            key = "left" if swipe_event == "swipe_left" else "right"
            action = "prev_slide" if swipe_event == "swipe_left" else "next_slide"
            if self._can_fire(action, timestamp):
                return self._dispatch(action, timestamp, lambda k=key: self.backend.press_key(k))

        if gesture_label is None or confidence < self.confidence_threshold:
            self._last_label = None
            self._streak = 0
            self._pinch_anchor_y = None
            return None

        self._streak = self._streak + 1 if self._last_label == gesture_label else 1
        self._last_label = gesture_label

        if gesture_label == "fist" and self._streak >= self.debounce_frames:
            if self._can_fire("play_pause", timestamp):
                self._streak = 0
                return self._dispatch(
                    "play_pause", timestamp, lambda: self.backend.press_key("playpause")
                )

        if gesture_label == "pinch" and hand_y is not None:
            if self._pinch_anchor_y is None:
                self._pinch_anchor_y = hand_y
            else:
                dy = hand_y - self._pinch_anchor_y
                if dy <= -self.volume_step_threshold and self._can_fire("volume_up", timestamp):
                    self._pinch_anchor_y = hand_y
                    return self._dispatch(
                        "volume_up", timestamp, lambda: self.backend.press_key("volumeup")
                    )
                if dy >= self.volume_step_threshold and self._can_fire("volume_down", timestamp):
                    self._pinch_anchor_y = hand_y
                    return self._dispatch(
                        "volume_down", timestamp, lambda: self.backend.press_key("volumedown")
                    )
        else:
            self._pinch_anchor_y = None

        return None
