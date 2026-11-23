# Human Handoff

## Current State
- **Phase 2 (`v0.2.0`)**: `HandTracker` (`VIDEO`/`IMAGE` modes) and 68-D landmark feature normalisation implemented and verified on synthetic inputs.

## Verified vs Not Verified
- **Verified:** PyTorch CUDA matmul + 5-step `nn.Linear` training on RTX 5050 (`sm_120`); `mediapipe.tasks.python.vision.HandLandmarker` (`1.2.3`); `hand_landmarker.task` SHA256; feature invariance, mirroring, NaN/no-hand handling, and camera probe smoke test (`pytest`, `ruff`).
- **Not Yet Implemented / Verified:** Phases 3–8 (data collection, classifier training, swipe/actions, live app, benchmark, evaluation).

## Human Gates (Pending)
1. **Human Gate 1 — Real Data Collection (Phase 3):** Record multi-session webcam landmark data (`python -m tghci.collect`).
2. **Human Gate 2 — Live Demo (Phase 6):** Run real-time webcam app (`python -m tghci.app`).
3. **Human Gate 3 — Real Training & Evaluation (Phase 7):** Train on real sessions and run held-out evaluation.
