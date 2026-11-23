# Hand Gesture Recognition for Touchless Interfaces

Clean-room implementation for **Deep Learning with Accelerated Computing (25CS3112E)**.

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

Note: `torch` is installed separately using the verified PyTorch CUDA 12.8 index (`cu128`) for Blackwell (`sm_120`) GPU support.

## Repository Layout
- `src/tghci/`: Core package modules (`config`, `tracker`, `features`, `dataset`, `model`, `train`, `swipe`, `actions`, `overlay`, `collect`, `evaluate`, `benchmark`, `app`).
- `configs/`: YAML configuration (`default.yaml`).
- `scripts/`: Helper scripts (`download_hand_model.py`).
- `tests/`: Automated unit and smoke tests.
- `docs/`: Course PDFs and technical documentation (`ARCHITECTURE.md`, `GESTURES.md`, `DATA.md`, `RESULTS.md`).
- `data/processed/`, `models/`, `results/`: Local data, model assets/checkpoints, and measured outputs.
