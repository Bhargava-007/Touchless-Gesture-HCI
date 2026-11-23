"""Unit tests for rule-based swipe detector and safety-first action mapper."""

from __future__ import annotations

from dataclasses import dataclass, field

from tghci.actions import ActionMapper
from tghci.swipe import SwipeDetector


@dataclass
class FakeBackend:
    """In-memory fake action backend for tests (never sends OS events)."""

    keys: list[str] = field(default_factory=list)
    moves: list[tuple[float, float]] = field(default_factory=list)
    clicks: int = 0

    def press_key(self, key: str) -> None:
        self.keys.append(key)

    def move_cursor_norm(self, x_norm: float, y_norm: float) -> None:
        self.moves.append((x_norm, y_norm))

    def click(self) -> None:
        self.clicks += 1


def _run_trajectory(detector: SwipeDetector, xs: list[float], ys: list[float], dt: float = 0.03):
    events = []
    t = 0.0
    for x, y in zip(xs, ys, strict=True):
        t += dt
        ev = detector.update(x, y, t)
        if ev is not None:
            events.append(ev)
    return events


def test_swipe_left_right_and_rejections() -> None:
    """Verify left/right swipe detection and rejection of jitter, drift, vertical, partial."""
    det = SwipeDetector(
        window_size=6,
        min_travel_norm=0.20,
        min_velocity_norm=0.50,
        max_vertical_ratio=0.5,
        refractory_seconds=0.5,
    )
    assert _run_trajectory(det, [0.2, 0.25, 0.3, 0.35, 0.42, 0.5], [0.5] * 6) == ["swipe_right"]
    det.reset()
    assert _run_trajectory(
        det, [0.8, 0.74, 0.68, 0.62, 0.56, 0.50], [0.5] * 6, dt=0.03
    ) == []  # inside refractory
    # Advance past refractory and test swipe_left
    events = []
    for i, x in enumerate([0.8, 0.74, 0.68, 0.62, 0.56, 0.50]):
        ev = det.update(x, 0.5, 2.0 + i * 0.03)
        if ev:
            events.append(ev)
    assert events == ["swipe_left"]

    # Jitter (zero net travel)
    det = SwipeDetector(window_size=6)
    assert _run_trajectory(det, [0.5, 0.52, 0.49, 0.51, 0.50, 0.51], [0.5] * 6) == []

    # Slow drift (below velocity threshold)
    det = SwipeDetector(window_size=6, min_travel_norm=0.2, min_velocity_norm=0.5)
    assert _run_trajectory(det, [0.2, 0.25, 0.3, 0.35, 0.4, 0.45], [0.5] * 6, dt=0.5) == []

    # Vertical motion
    det = SwipeDetector(window_size=6, max_vertical_ratio=0.5)
    assert _run_trajectory(det, [0.2, 0.25, 0.3, 0.35, 0.4, 0.45], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]) == []

    # Partial trajectory (fewer than window_size frames)
    det = SwipeDetector(window_size=6)
    assert _run_trajectory(det, [0.2, 0.4, 0.6], [0.5, 0.5, 0.5]) == []


def test_action_mapper_safety_defaults_debounce_cooldown_and_modes() -> None:
    """Verify disarmed default, dry-run, debounce, cooldown, no-hand, and cursor mode."""
    backend = FakeBackend()
    mapper = ActionMapper(
        backend=backend,
        start_armed=False,
        dry_run=True,
        debounce_frames=3,
        cooldown_seconds=1.0,
    )

    # Disarmed: no action even with fist held
    for i in range(5):
        assert mapper.step(True, "fist", 0.95, float(i)) is None
    assert backend.keys == []

    # Arm, but dry_run is still True: logs event without calling backend
    mapper.toggle_armed()
    ev = None
    for i in range(3):
        ev = mapper.step(True, "fist", 0.95, 10.0 + i * 0.1)
    assert ev is not None and ev.action == "play_pause" and not ev.executed
    assert backend.keys == []

    # Disable dry_run: debounce requires 3 frames, then cooldown blocks immediate repeat
    mapper.toggle_dry_run()
    assert mapper.step(True, "fist", 0.95, 20.0) is None
    assert mapper.step(True, "fist", 0.95, 20.1) is None
    ev2 = mapper.step(True, "fist", 0.95, 20.2)
    assert ev2 is not None and ev2.executed is True
    assert backend.keys == ["playpause"]

    # Within cooldown (20.5 < 20.2 + 1.0)
    for i in range(3):
        assert mapper.step(True, "fist", 0.95, 20.3 + i * 0.05) is None
    assert backend.keys == ["playpause"]

    # No hand tracked: never acts
    assert mapper.step(False, "fist", 0.99, 25.0, swipe_event="swipe_right") is None
    assert backend.keys == ["playpause"]

    # Pinch + vertical movement -> volume
    mapper.step(True, "pinch", 0.95, 30.0, hand_y=0.50)
    ev_vol = mapper.step(True, "pinch", 0.95, 30.1, hand_y=0.40)
    assert ev_vol is not None and ev_vol.action == "volume_up" and ev_vol.executed
    assert backend.keys == ["playpause", "volumeup"]

    # Switch to cursor mode: moves pointer and pinch clicks after debounce
    assert mapper.toggle_mode() == "cursor"
    for i in range(3):
        ev_click = mapper.step(True, "pinch", 0.95, 40.0 + i * 0.1, index_tip_xy=(0.4, 0.6))
    assert ev_click is not None and ev_click.action == "click" and ev_click.executed
    assert backend.clicks == 1
    assert len(backend.moves) == 3
