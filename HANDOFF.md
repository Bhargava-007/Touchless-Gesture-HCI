# Human Handoff

## Current State
- **Phase 7 (`v0.7.0`)**: CPU vs GPU latency benchmark (`tghci.benchmark`) and held-out evaluation (`tghci.evaluate`) implemented and verified; real benchmark CSV recorded in `results/benchmark_20261124_035254.csv`.

## Verified vs Not Verified
- **Verified:** PyTorch CUDA matmul + 5-step `nn.Linear` training on RTX 5050 (`sm_120`); MediaPipe `HandLandmarker` (`1.2.3`); 68-D feature normalisation; session CSV validation & split; synthetic CPU/CUDA training; swipe & safety-first action mapper; HUD overlay & fake-source app loop; real CPU vs CUDA latency benchmark (`results/benchmark_20261124_035254.csv`); synthetic evaluation code path (`pytest`, `ruff`).
- **Not Yet Implemented / Verified:** Phase 8 (final polish); **Human Gate 1** (real data collection); **Human Gate 2** (live webcam demo); **Human Gate 3** (real training, evaluation, and trained-checkpoint benchmark).

## Human Gates (Pending)
1. **Human Gate 1 — Real Data Collection (Phase 3):** Record multi-session webcam landmark data (`python -m tghci.collect`).
2. **Human Gate 2 — Live Demo (Phase 6):** Run real-time webcam app (`python -m tghci.app`).
3. **Human Gate 3 — Real Training & Evaluation (Phase 7):** Train on real sessions and run held-out evaluation.
