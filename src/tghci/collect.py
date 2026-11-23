"""Interactive webcam landmark data collection tool (per-session CSVs)."""

from __future__ import annotations

import argparse
import csv
import time
import uuid
from pathlib import Path

import cv2

from tghci.config import load_config
from tghci.features import NUM_BASE_FEATURES, extract_features
from tghci.tracker import HandTracker, open_camera

FEATURE_COLUMNS = [f"f_{i}" for i in range(NUM_BASE_FEATURES)]
CSV_COLUMNS = ["session_id", "timestamp", "label", "handedness", "confidence", *FEATURE_COLUMNS]


def make_session_id(prefix: str = "sess") -> str:
    """Generate a unique session identifier."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"{prefix}_{stamp}_{short}"


def write_session_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    """Write recorded landmark rows to a session CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_collection(
    config_path: str | Path | None = None,
    session_id: str | None = None,
    min_confidence: float = 0.6,
    countdown_seconds: float = 2.0,
) -> int:
    """Run interactive webcam landmark collection.

    Keys:
        1..4: select gesture label and start countdown to record
        space: pause/stop recording
        q: save session CSV (if non-empty) and quit
    """
    cfg = load_config(config_path)
    labels: list[str] = list(cfg["labels"])
    cam_cfg = cfg.get("camera", {})
    tracker_cfg = cfg.get("tracker", {})
    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    model_path = Path(cfg["paths"]["hand_model"])

    cap = open_camera(
        device_index=int(cam_cfg.get("device_index", 0)),
        width=int(cam_cfg.get("width", 640)),
        height=int(cam_cfg.get("height", 480)),
    )
    if cap is None:
        print("ERROR: No webcam available. Cannot run data collection.")
        return 1

    sid = session_id or make_session_id()
    rows: list[dict[str, object]] = []
    active_label: str | None = None
    record_start_at: float | None = None
    recording = False

    try:
        with HandTracker(
            model_path=model_path,
            num_hands=int(tracker_cfg.get("num_hands", 1)),
            min_hand_detection_confidence=float(
                tracker_cfg.get("min_hand_detection_confidence", 0.5)
            ),
            min_hand_presence_confidence=float(
                tracker_cfg.get("min_hand_presence_confidence", 0.5)
            ),
            min_tracking_confidence=float(tracker_cfg.get("min_tracking_confidence", 0.5)),
            running_mode="VIDEO",
        ) as tracker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                now = time.time()
                detections = tracker.detect(frame, timestamp_ms=int(now * 1000))

                status_text = f"Session: {sid} | Samples: {len(rows)}"
                if active_label and record_start_at and now < record_start_at:
                    rem = record_start_at - now
                    status_text += f" | Starting {active_label} in {rem:.1f}s"
                    recording = False
                elif active_label and recording:
                    status_text += f" | RECORDING: {active_label}"
                else:
                    status_text += " | Press 1-4 to record, SPACE pause, Q quit"

                if recording and active_label and detections:
                    det = detections[0]
                    if det.confidence >= min_confidence:
                        feats = extract_features(
                            det.landmarks,
                            handedness=det.handedness,
                            mirror_left_hand=bool(
                                cfg.get("features", {}).get("mirror_left_hand", True)
                            ),
                            include_auxiliary=False,
                        )
                        row: dict[str, object] = {
                            "session_id": sid,
                            "timestamp": f"{now:.6f}",
                            "label": active_label,
                            "handedness": det.handedness,
                            "confidence": f"{det.confidence:.4f}",
                        }
                        for i, val in enumerate(feats):
                            row[f"f_{i}"] = f"{float(val):.6f}"
                        rows.append(row)

                cv2.putText(
                    frame,
                    status_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0) if recording else (255, 255, 255),
                    2,
                )
                cv2.imshow("tghci-collect", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    recording = False
                    record_start_at = None
                if ord("1") <= key <= ord("9"):
                    idx = key - ord("1")
                    if idx < len(labels):
                        active_label = labels[idx]
                        record_start_at = now + countdown_seconds
                        recording = True
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if rows:
        out_path = raw_dir / f"{sid}.csv"
        write_session_csv(out_path, rows)
        print(f"Saved {len(rows)} samples to {out_path}")
    else:
        print("No samples recorded; nothing written.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for webcam data collection."""
    parser = argparse.ArgumentParser(description="Record hand landmark sessions")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--session-id", type=str, default=None, help="Explicit session ID")
    parser.add_argument("--min-confidence", type=float, default=0.6, help="Min hand confidence")
    parser.add_argument("--countdown", type=float, default=2.0, help="Countdown seconds")
    args = parser.parse_args(argv)
    return run_collection(
        config_path=args.config,
        session_id=args.session_id,
        min_confidence=args.min_confidence,
        countdown_seconds=args.countdown,
    )


if __name__ == "__main__":
    raise SystemExit(main())
