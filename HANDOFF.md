# Human Handoff

## Current State
- **Phase 4 (`v0.4.0`)**: `GestureMLP`, deterministic training CLI (`tghci.train`), and `GestureClassifier` inference helper implemented and smoke-tested on synthetic data (CPU and CUDA).

## Verified vs Not Verified
- **Verified:** PyTorch CUDA matmul + 5-step `nn.Linear` training on RTX 5050 (`sm_120`); MediaPipe `HandLandmarker` (`1.2.3`); 68-D feature normalisation; session CSV validation & split; synthetic end-to-end training + inference on CPU and CUDA (`pytest`, `ruff`).
- **Not Yet Implemented / Verified:** Phases 5–8 (swipe/actions, live app, benchmark, evaluation); **Human Gate 1** (real webcam data collection).

## Human Gates (Pending)
1. **Human Gate 1 — Real Data Collection (Phase 3):** Record multi-session webcam landmark data (`python -m tghci.collect`).
2. **Human Gate 2 — Live Demo (Phase 6):** Run real-time webcam app (`python -m tghci.app`).
3. **Human Gate 3 — Real Training & Evaluation (Phase 7):** Train on real sessions and run held-out evaluation.
