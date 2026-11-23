# Human Handoff

## Current State
- **Phase 8 (`v0.8.0`)**: All code modules, tests, documentation, and CPU vs GPU latency benchmark are complete and merged to `main`. Tag `v1.0.0` is intentionally withheld until a human completes the 3 Human Gates below.

## Verified vs Not Verified
- **Verified:**
  - PyTorch `2.11.0+cu128` CUDA matmul + 5-step `nn.Linear` training on RTX 5050 (`sm_120`).
  - MediaPipe `1.2.3` `HandLandmarker` API + `hand_landmarker.task` download (`SHA256: fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`).
  - 68-D landmark normalisation, session CSV validation, zero-leakage session split, synthetic CPU/CUDA training, swipe heuristic, safety-first action mapper (`FakeBackend`), HUD overlay, fake-source app loop, and evaluation code path (`pytest`, `ruff`).
  - Real CPU vs GPU inference latency benchmark saved to `results/benchmark_20261124_035254.csv` and recorded in `docs/RESULTS.md`.
- **Not Verified (Requires Human + Webcam):**
  - Real webcam landmark capture (`data/raw/*.csv`).
  - Model accuracy on real held-out human sessions.
  - Live interactive webcam demo with OS actions.

## Human Gates (Exact Commands)

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
Then update `docs/RESULTS.md` with the measured evaluation table and tag `v1.0.0` once verified.

## Known Issues / Notes
- `py -0p` does not list per-user Python installs in this environment; invoke `.venv\Scripts\python.exe` directly.
- `git -C .. status` fails inside the IDE sandbox due to parent `.git` permissions; run all git commands inside `Project-TG-HCI` only.
- `git remote` push URL is set to `DISABLED` to prevent accidental pushes.
