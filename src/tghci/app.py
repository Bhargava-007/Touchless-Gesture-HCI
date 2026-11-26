"""Real-time webcam gesture recognition application loop."""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from tghci.actions import ActionBackend, ActionMapper
from tghci.config import load_config
from tghci.features import extract_features_safe
from tghci.model import GestureClassifier
from tghci.overlay import OverlayState, draw_overlay
from tghci.swipe import SwipeDetector
from tghci.tracker import HandTracker, open_camera


def _resolve_runtime_device(requested: str) -> str:
    """Select 'cuda' or 'cpu', falling back to 'cpu' with a clear message if unavailable."""
    req = requested.lower()
    if req == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but unavailable; falling back to CPU.")
        return "cpu"
    return req


def _iter_camera_frames(cap: Any) -> Iterator[np.ndarray]:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame


def run_app(
    config_path: str | Path | None = None,
    device_name: str = "cuda",
    checkpoint_path: str | Path | None = None,
    frame_source: Iterator[np.ndarray] | None = None,
    tracker: Any | None = None,
    classifier: Any | None = None,
    backend: ActionBackend | None = None,
    display: bool = True,
    max_frames: int | None = None,
) -> int:
    """Run the real-time gesture recognition pipeline."""
    cfg = load_config(config_path)
    device = _resolve_runtime_device(device_name)

    ckpt = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else Path(cfg["paths"]["checkpoint_path"])
    )
    if classifier is None:
        if not ckpt.is_file():
            print(
                f"ERROR: Trained checkpoint not found at {ckpt}. "
                "Collect sessions and run `python -m tghci.train` first."
            )
            return 1
        classifier = GestureClassifier(ckpt, device=device)

    cap = None
    owns_tracker = tracker is None
    if frame_source is None:
        cam_cfg = cfg.get("camera", {})
        cap = open_camera(
            device_index=int(cam_cfg.get("device_index", 0)),
            width=int(cam_cfg.get("width", 640)),
            height=int(cam_cfg.get("height", 480)),
        )
        if cap is None:
            print("ERROR: No webcam available. Exiting cleanly.")
            return 1
        frame_source = _iter_camera_frames(cap)

    if tracker is None:
        tc = cfg.get("tracker", {})
        tracker = HandTracker(
            model_path=Path(cfg["paths"]["hand_model"]),
            num_hands=int(tc.get("num_hands", 1)),
            min_hand_detection_confidence=float(tc.get("min_hand_detection_confidence", 0.5)),
            min_hand_presence_confidence=float(tc.get("min_hand_presence_confidence", 0.5)),
            min_tracking_confidence=float(tc.get("min_tracking_confidence", 0.5)),
            running_mode=str(tc.get("running_mode", "VIDEO")),
        )

    feat_cfg = cfg.get("features", {})
    mirror_left = bool(feat_cfg.get("mirror_left_hand", True))
    include_aux = bool(feat_cfg.get("include_auxiliary", True))
    swipe = SwipeDetector.from_config(cfg)
    mapper = ActionMapper.from_config(cfg, backend=backend)

    prev_time = time.perf_counter()
    fps = 0.0
    frame_idx = 0

    try:
        for frame in frame_source:
            frame_idx += 1
            now_perf = time.perf_counter()
            dt = max(now_perf - prev_time, 1e-6)
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if frame_idx > 1 else (1.0 / dt)
            prev_time = now_perf
            now_wall = time.time()

            detections = tracker.detect(frame, timestamp_ms=int(now_perf * 1000))
            if detections:
                det = detections[0]
                feats = extract_features_safe(
                    det.landmarks,
                    handedness=det.handedness,
                    mirror_left_hand=mirror_left,
                    include_auxiliary=include_aux,
                )
                if feats is not None:
                    label, conf = classifier.predict(feats)
                    centre_x = 1.0 - float((det.landmarks[0, 0] + det.landmarks[9, 0]) * 0.5)
                    centre_y = float((det.landmarks[0, 1] + det.landmarks[9, 1]) * 0.5)
                    swipe_ev = swipe.update(centre_x, centre_y, now_perf)
                    index_xy = (1.0 - float(det.landmarks[8, 0]), float(det.landmarks[8, 1]))
                    mapper.step(
                        hand_present=True,
                        gesture_label=label,
                        confidence=conf,
                        timestamp=now_wall,
                        swipe_event=swipe_ev,
                        hand_y=centre_y,
                        index_tip_xy=index_xy,
                    )
                    lm_draw = det.landmarks.copy()
                    lm_draw[:, 0] = 1.0 - lm_draw[:, 0]
                else:
                    label, conf, lm_draw = "none", 0.0, None
                    swipe.update(None, None, now_perf)
                    mapper.step(False, None, 0.0, now_wall)
            else:
                label, conf, lm_draw = "none", 0.0, None
                swipe.update(None, None, now_perf)
                mapper.step(False, None, 0.0, now_wall)

            hud = OverlayState(
                fps=fps,
                gesture=label,
                confidence=conf,
                armed=mapper.armed,
                dry_run=mapper.dry_run,
                mode=mapper.mode,
                last_action=mapper.last_action_label,
                device=device,
            )
            display_frame = cv2.flip(frame, 1)
            rendered = draw_overlay(display_frame, hud, landmarks=lm_draw)

            if display:
                cv2.imshow("Touchless Gesture HCI", rendered)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("a"):
                    mapper.toggle_armed()
                elif key == ord("d"):
                    mapper.toggle_dry_run()
                elif key == ord("m"):
                    mapper.toggle_mode()

            if max_frames is not None and frame_idx >= max_frames:
                break
    finally:
        if cap is not None:
            cap.release()
        if owns_tracker and hasattr(tracker, "close"):
            tracker.close()
        if display:
            cv2.destroyAllWindows()

    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for `python -m tghci.app`."""
    parser = argparse.ArgumentParser(description="Run Touchless Gesture HCI live app")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint path")
    args = parser.parse_args(argv)
    return run_app(
        config_path=args.config,
        device_name=args.device,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    raise SystemExit(main())
