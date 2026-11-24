# Hand Gesture Recognition for Touchless Interfaces

Clean-room course project for **Deep Learning with Accelerated Computing (25CS3112E)**.

## Setup (Native Windows 11, PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-dev.txt
pip install -e . --no-deps
python scripts\download_hand_model.py
python -m tghci.config --check
```

Note: `torch` is installed separately using the PyTorch CUDA 12.8 index (`cu128`) for NVIDIA Blackwell (`sm_120`) GPU support.

## Usage

```powershell
# 1. Record session-tagged webcam landmarks (Human Gate 1)
python -m tghci.collect

# 2. Train the gesture classifier MLP (requires >= 3 recorded sessions)
python -m tghci.train --device cuda --seed 42

# 3. Evaluate on the held-out session test split
python -m tghci.evaluate --device cpu --seed 42

# 4. Run CPU vs GPU inference latency benchmark
python -m tghci.benchmark --iterations 1000

# 5. Launch live webcam application (Human Gate 2)
python -m tghci.app --device cuda
```

## Safety Controls (Live App)
- Starts **DISARMED** and in **DRY-RUN** mode by default (`configs/default.yaml`).
- Keyboard controls in the OpenCV window:
  - `a`: Arm / disarm OS actions
  - `d`: Toggle dry-run logging vs live `pyautogui` actions
  - `m`: Toggle `media` mode vs `cursor` mode
  - `q`: Quit cleanly
- `pyautogui.FAILSAFE` remains enabled; no action is dispatched when no hand is tracked.

## Repository Layout
- `src/tghci/`: Core package modules (`config`, `tracker`, `features`, `collect`, `dataset`, `model`, `train`, `swipe`, `actions`, `overlay`, `app`, `evaluate`, `benchmark`).
- `configs/default.yaml`: Hyperparameters, swipe thresholds, safety defaults, and key mappings.
- `scripts/download_hand_model.py`: Official MediaPipe `hand_landmarker.task` downloader with SHA256 verification.
- `tests/`: Automated unit and synthetic integration tests (`pytest`).
- `docs/`: course PDFs; `notes/`: architecture, data protocol, gesture definitions and measured results.
- `models/`, `results/`: local model assets/checkpoints (gitignored) and measured CSV/figure outputs.

## Project Documentation
- Course deliverables (proposal abstract and review slides PDFs) live in `docs/`.
- Technical notes (`ARCHITECTURE.md`, `DATA.md`, `GESTURES.md`, `RESULTS.md`) live in `notes/`.

## Human Gates

### Human Gate 1 — Real Data Collection
Each team member should record at least 3 independent sessions (minimum 9 total) across varied lighting and distances:
```powershell
.\.venv\Scripts\python.exe -m tghci.collect
```
Press `1`–`4` (`open_palm`, `fist`, `pinch`, `neutral`), `Space` to pause, `q` to save session CSV to `data/raw/`.

### Human Gate 2 — Live Webcam Demo
After training a checkpoint (see Human Gate 3):
```powershell
.\.venv\Scripts\python.exe -m tghci.app --device cuda
```
Starts disarmed in dry-run mode. Press `a` to arm, `d` to toggle live actions, `m` to switch `media`/`cursor` mode, `q` to quit.

### Human Gate 3 — Train, Evaluate, and Re-Benchmark on Real Data
```powershell
.\.venv\Scripts\python.exe -m tghci.train --device cuda --seed 42
.\.venv\Scripts\python.exe -m tghci.evaluate --device cpu --seed 42
.\.venv\Scripts\python.exe -m tghci.benchmark --checkpoint models\gesture_mlp.pt --iterations 1000
```

## Current Status & Limitations
- All code modules, unit tests, synthetic training/evaluation smoke tests, and the CPU vs GPU latency benchmark (`results/benchmark_20261124_035254.csv`) are implemented and verified.
- Real webcam data collection, real-data model training/evaluation, and live webcam demoing require human interaction (**Human Gates 1–3** above).
