"""Configuration loading and environment verification CLI for tghci."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


@dataclass(frozen=True)
class EnvCheckReport:
    """Summary of runtime environment capabilities."""

    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_device_name: str | None
    cuda_capability: tuple[int, int] | None
    mediapipe_version: str
    opencv_version: str
    camera_available: bool
    hand_model_exists: bool
    hand_model_path: str


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration from disk.

    Args:
        path: Optional path to a YAML config file. Defaults to configs/default.yaml.

    Returns:
        Parsed configuration dictionary.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in config {cfg_path}, got {type(data).__name__}")
    return data


def _probe_camera(device_index: int = 0) -> bool:
    """Check whether an OpenCV camera device can be opened cleanly."""
    try:
        import cv2

        cap = cv2.VideoCapture(device_index)
        try:
            return bool(cap.isOpened())
        finally:
            cap.release()
    except Exception:
        return False


def run_env_check(config_path: str | Path | None = None) -> EnvCheckReport:
    """Collect versions, CUDA status, camera availability, and model presence."""
    import sys

    import cv2
    import mediapipe as mp
    import torch

    cfg = load_config(config_path)
    model_rel = cfg.get("paths", {}).get("hand_model", "models/hand_landmarker.task")
    project_root = Path(__file__).resolve().parents[2]
    model_path = (project_root / model_rel).resolve()
    cam_index = int(cfg.get("camera", {}).get("device_index", 0))

    cuda_ok = bool(torch.cuda.is_available())
    dev_name = torch.cuda.get_device_name(0) if cuda_ok else None
    cap = torch.cuda.get_device_capability(0) if cuda_ok else None

    return EnvCheckReport(
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        cuda_available=cuda_ok,
        cuda_device_name=dev_name,
        cuda_capability=cap,
        mediapipe_version=getattr(mp, "__version__", "unknown"),
        opencv_version=cv2.__version__,
        camera_available=_probe_camera(cam_index),
        hand_model_exists=model_path.is_file(),
        hand_model_path=str(model_path),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for `python -m tghci.config --check`."""
    parser = argparse.ArgumentParser(description="tghci configuration and environment check")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--check", action="store_true", help="Print environment check report")
    args = parser.parse_args(argv)

    if args.check:
        report = run_env_check(args.config)
        print(f"python: {report.python_version}")
        print(f"torch: {report.torch_version}")
        print(f"cuda_available: {report.cuda_available}")
        print(f"cuda_device: {report.cuda_device_name}")
        print(f"cuda_capability: {report.cuda_capability}")
        print(f"mediapipe: {report.mediapipe_version}")
        print(f"opencv: {report.opencv_version}")
        print(f"camera_available: {report.camera_available}")
        print(f"hand_model_exists: {report.hand_model_exists} ({report.hand_model_path})")
        return 0

    cfg = load_config(args.config)
    print(f"Loaded config keys: {list(cfg.keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
