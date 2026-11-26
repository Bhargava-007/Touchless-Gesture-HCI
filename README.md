# Hand Gesture Recognition for Touchless Interfaces

Clean-room course project for **Deep Learning with Accelerated Computing (25CS3112E)**.

Real-time touchless HCI pipeline combining MediaPipe `HandLandmarker` (21 3D landmarks), 68-D geometric feature normalisation, a PyTorch `GestureMLP` classifier (`open_palm`, `fist`, `pinch`, `neutral`), a rule-based horizontal swipe detector, and a safety-first `pyautogui` action mapper.

## Setup (Native Windows 11, PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe scripts\download_hand_model.py
.\.venv\Scripts\python.exe -m tghci.config --check
```

Note: `torch` is installed from the PyTorch CUDA 12.8 index (`cu128`) for NVIDIA Blackwell (`sm_120`) GPU support. `scripts\download_hand_model.py` downloads the official MediaPipe `hand_landmarker.task` bundle into `models/` and verifies its SHA256 checksum.

## Usage

```powershell
# 1. Record session-tagged webcam landmarks (Human Gate 1)
.\.venv\Scripts\python.exe -m tghci.collect

# 2. Train the gesture classifier MLP (requires >= 3 recorded sessions)
.\.venv\Scripts\python.exe -m tghci.train --device cuda --seed 42

# 3. Evaluate on the held-out session test split
.\.venv\Scripts\python.exe -m tghci.evaluate --device cuda

# 4. Run CPU vs GPU inference latency benchmark
.\.venv\Scripts\python.exe -m tghci.benchmark --checkpoint models\gesture_mlp.pt --iterations 1000

# 5. Launch live webcam application (Human Gate 2)
.\.venv\Scripts\python.exe -m tghci.app --device cuda
```

## Safety Controls (Live App)
- Starts **DISARMED** and in **DRY-RUN** mode by default (`configs/default.yaml`).
- Keyboard controls in the OpenCV window:
  - `a`: Arm / disarm OS actions
  - `d`: Toggle dry-run logging vs live `pyautogui` actions
  - `m`: Toggle `media` mode (swipes, play/pause, pinch-volume) vs `cursor` mode (index pointer + pinch click)
  - `q`: Quit cleanly
- `pyautogui.FAILSAFE` remains enabled (with a 2 px edge margin in `PyAutoGuiBackend.move_cursor_norm`); no action is dispatched when no hand is tracked.

## Measured Results

All metrics below are taken directly from recorded outputs in `results/` and documented in `notes/RESULTS.md`.

- **Training (`results/train_log.csv`):** 5 real webcam sessions (4,999 total rows: 3 train = 1,901, 1 val = 1,086, 1 test = 2,012) trained on `cuda` (`seed=42`). Stopped at epoch 36 (best epoch 24) with best validation loss `0.1340` and best validation accuracy `0.9751`.
- **Held-Out Evaluation (`results/eval_20261126_221933.csv`):** Evaluated on unseen session `sess_20261126_221452_1a06ce` (2,012 samples), achieving `0.7023` overall accuracy (`fist` F1 `0.8532`, `pinch` F1 `0.7370`, `neutral` F1 `0.5663`, `open_palm` F1 `0.5341`). Confusion matrices are saved in `results/figures/`.
- **CPU vs CUDA Latency (`results/benchmark_20261124_035254.csv` [random-init] & `results/benchmark_20261126_221940.csv` [trained checkpoint]):** `GestureMLP` (`68 -> 128 -> 64 -> 4`) forward-pass latency across batch sizes `1, 8, 64, 512` (1,000 iterations, `torch==2.11.0+cu128`, RTX 5050 Laptop GPU). For the trained checkpoint at batch size `1`, CPU median latency was `0.0532 ms` vs CUDA `0.5633 ms` (CPU was faster than CUDA across all tested batch sizes).

## Repository Layout
- `src/tghci/`: Core package modules (`config`, `tracker`, `features`, `collect`, `dataset`, `model`, `train`, `swipe`, `actions`, `overlay`, `app`, `evaluate`, `benchmark`).
- `configs/default.yaml`: Hyperparameters, swipe thresholds, safety defaults, and key mappings.
- `scripts/download_hand_model.py`: Official MediaPipe `hand_landmarker.task` downloader with SHA256 verification.
- `tests/`: Automated unit and synthetic integration tests (`pytest`).
- `docs/`: Course deliverables (proposal abstract and review slides PDFs).
- `notes/`: Technical notes (`ARCHITECTURE.md`, `DATA.md`, `GESTURES.md`, `RESULTS.md`).
- `models/`, `results/`: Local model assets/checkpoints (gitignored) and measured CSV/figure outputs.

## Limitations & Future Work
- **Single-user dataset:** The current checkpoint is trained on 5 sessions from a single user, contributing to the generalisation gap between validation (`0.9751`) and held-out test (`0.7023`) accuracy; collecting multi-user sessions across varied lighting and backgrounds is the primary next step.
- **Cursor click ergonomics:** Cursor mode now includes `PAUSE = 0.0`, EMA smoothing, `[0.15, 0.85]` active ROI mapping, horizontal mirroring, and pinch-freeze during click debounce, though dedicated click/drag ergonomics remain an area for further refinement.
- **Tiny-MLP GPU overhead:** CPU forward-pass inference is faster than CUDA for this 68-D 3-layer MLP at small batch sizes (`1`–`512`); GPU acceleration becomes more relevant if scaling to heavier temporal/image models.

## Credits
Built from scratch using official documentation for:
- [PyTorch](https://pytorch.org/docs/stable/index.html)
- [MediaPipe Hand Landmarker (Tasks API)](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- [OpenCV](https://docs.opencv.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [PyAutoGUI](https://pyautogui.readthedocs.io/)

## License
Licensed under the [MIT License](LICENSE).
