"""Download the official MediaPipe hand_landmarker.task model bundle."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_DEST = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"


def sha256_file(path: Path) -> str:
    """Compute hex SHA256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(url: str = MODEL_URL, dest: Path = DEFAULT_DEST) -> Path:
    """Download the hand landmarker task file if missing and print its SHA256."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"Downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
    else:
        print(f"Model already present at {dest}")
    checksum = sha256_file(dest)
    print(f"SHA256: {checksum}")
    print(f"Bytes: {dest.stat().st_size}")
    return dest


if __name__ == "__main__":
    download_model()
