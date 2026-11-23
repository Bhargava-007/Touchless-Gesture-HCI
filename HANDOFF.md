# Human Handoff

## Current State
- **Phase 5 (`v0.5.0`)**: Rule-based swipe detector (`tghci.swipe`) and safety-first action mapper (`tghci.actions`) implemented and tested with `FakeBackend`.

## Verified vs Not Verified
- **Verified:** PyTorch CUDA matmul + 5-step `nn.Linear` training on RTX 5050 (`sm_120`); MediaPipe `HandLandmarker` (`1.2.3`); 68-D feature normalisation; session CSV validation & split; synthetic CPU/CUDA training; swipe trajectories and safety-first action gating (`pytest`, `ruff`).
- **Not Yet Implemented / Verified:** Phases 6–8 (live app/overlay, benchmark, evaluation); **Human Gate 1** (real webcam data collection); **Human Gate 2** (live demo).

## Human Gates (Pending)
1. **Human Gate 1 — Real Data Collection (Phase 3):** Record multi-session webcam landmark data (`python -m tghci.collect`).
2. **Human Gate 2 — Live Demo (Phase 6):** Run real-time webcam app (`python -m tghci.app`).
3. **Human Gate 3 — Real Training & Evaluation (Phase 7):** Train on real sessions and run held-out evaluation.
