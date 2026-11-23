# Human Handoff

## Current State
- **Phase 3 (`v0.3.0`)**: Session-tagged webcam data collector (`tghci.collect`) and session-based dataset loader/splitter (`tghci.dataset`) implemented and verified on synthetic CSVs.

## Verified vs Not Verified
- **Verified:** PyTorch CUDA matmul + 5-step `nn.Linear` training on RTX 5050 (`sm_120`); MediaPipe `HandLandmarker` (`1.2.3`); 68-D feature normalisation; session CSV schema validation, zero-leakage session split, and `< 3` session guard (`pytest`, `ruff`).
- **Not Yet Implemented / Verified:** Phases 4–8 (classifier training, swipe/actions, live app, benchmark, evaluation); **Human Gate 1** (real webcam data collection).

## Human Gates (Pending)
1. **Human Gate 1 — Real Data Collection (Phase 3):** Record multi-session webcam landmark data (`python -m tghci.collect`).
2. **Human Gate 2 — Live Demo (Phase 6):** Run real-time webcam app (`python -m tghci.app`).
3. **Human Gate 3 — Real Training & Evaluation (Phase 7):** Train on real sessions and run held-out evaluation.
