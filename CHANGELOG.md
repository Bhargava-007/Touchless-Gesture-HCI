# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-11-24
### Added
- `tghci.tracker.HandTracker` wrapper around MediaPipe `HandLandmarker` (`VIDEO` and `IMAGE` modes) plus graceful `open_camera`.
- `tghci.features` wrist-origin translation, palm-scale normalisation, left-hand mirroring, and 5 auxiliary geometric distance features (68-D `float32`).
- Unit tests in `tests/test_landmarks.py` and formula documentation in `docs/GESTURES.md`.

## [0.1.0] - 2026-11-24
### Added
- Repository scaffold (`src/tghci/`, `configs/default.yaml`, `scripts/download_hand_model.py`, `tests/`, `docs/`).
- Environment verification CLI (`python -m tghci.config --check`).
- Initial documentation (`README.md`, `CHANGELOG.md`, `ERRORLOG.md`, `DECISIONS.md`, `HANDOFF.md`, `docs/ARCHITECTURE.md`, `docs/GESTURES.md`, `docs/DATA.md`, `docs/RESULTS.md`).
- Pinned `requirements.txt` and `requirements-dev.txt` with verified MediaPipe and PyTorch CUDA environment.
