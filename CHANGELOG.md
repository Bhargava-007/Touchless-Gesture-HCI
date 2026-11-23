# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.0] - 2026-11-24
### Added
- `tghci.benchmark` CPU vs CUDA inference latency benchmark with warmup, `torch.cuda.synchronize()`, and CSV export (`results/benchmark_20261124_035254.csv`).
- `tghci.evaluate` held-out session evaluation, per-gesture precision/recall/F1 CSV, and confusion matrix PNG generation.
- Measured latency table in `docs/RESULTS.md` and synthetic code-path tests in `tests/test_bench_eval.py`.

## [0.6.0] - 2026-11-24
### Added
- `tghci.overlay` pure OpenCV HUD and hand landmark rendering utilities.
- `tghci.app` real-time pipeline loop with keyboard controls (`q`, `a`, `d`, `m`), graceful camera/checkpoint/GPU handling, and injectable test seams.
- Unit tests in `tests/test_app.py` verifying overlay rendering and fake-source loop execution.

## [0.5.0] - 2026-11-24
### Added
- `tghci.swipe.SwipeDetector` rule-based horizontal swipe heuristic with refractory period.
- `tghci.actions.ActionMapper` with disarmed-by-default safety, dry-run mode, debounce, cooldown, media/cursor modes, and injectable `ActionBackend`.
- Unit tests in `tests/test_swipe_actions.py` using synthetic trajectories and `FakeBackend`.

## [0.4.0] - 2026-11-24
### Added
- `tghci.model.GestureMLP`, checkpoint save/load with sidecar `.meta.json`, and `GestureClassifier` inference helper.
- `tghci.train` deterministic training CLI (`--config`, `--device`, `--seed`) with class weights, early stopping, and `results/train_log.csv` logging.
- End-to-end synthetic training + inference smoke tests on CPU and CUDA in `tests/test_classifier.py`.

## [0.3.0] - 2026-11-24
### Added
- `tghci.collect` interactive webcam landmark recorder writing per-session CSVs to `data/raw/`.
- `tghci.dataset` session loader, schema validator, strict session-based train/val/test splitter (requires >= 3 sessions), and `LandmarkDataset` with mirror + jitter augmentation.
- Team recording protocol and privacy note in `docs/DATA.md`, plus synthetic CSV tests in `tests/test_data.py`.

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
