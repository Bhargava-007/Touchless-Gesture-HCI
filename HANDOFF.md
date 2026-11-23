# Human Handoff

## Current State
- **Phase 6 (`v0.6.0`)**: HUD overlay (`tghci.overlay`) and main application loop (`tghci.app`) implemented and verified with a fake frame source.

## Verified vs Not Verified
- **Verified:** PyTorch CUDA matmul + 5-step `nn.Linear` training on RTX 5050 (`sm_120`); MediaPipe `HandLandmarker` (`1.2.3`); 68-D feature normalisation; session CSV validation & split; synthetic CPU/CUDA training; swipe & safety-first action mapper; HUD overlay and fake-source app loop (`pytest`, `ruff`).
- **Not Yet Implemented / Verified:** Phases 7–8 (benchmark, evaluation, final polish); **Human Gate 1** (real data collection); **Human Gate 2** (live webcam demo).

## Human Gates (Pending)
1. **Human Gate 1 — Real Data Collection (Phase 3):** Record multi-session webcam landmark data (`python -m tghci.collect`).
2. **Human Gate 2 — Live Demo (Phase 6):** Run real-time webcam app (`python -m tghci.app`).
3. **Human Gate 3 — Real Training & Evaluation (Phase 7):** Train on real sessions and run held-out evaluation.
