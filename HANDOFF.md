# Human Handoff

## Current State
- **Phase 1 (`v0.1.0`)**: Repository scaffold, `.venv` (Python 3.13.16), MediaPipe `1.2.3`, PyTorch `2.11.0+cu128` (`sm_120` verified), and `hand_landmarker.task` downloaded.

## Verified vs Not Verified
- **Verified:** PyTorch CUDA matmul + 5-step `nn.Linear` training on RTX 5050 (`sm_120`); `mediapipe.tasks.python.vision.HandLandmarker` package install; `hand_landmarker.task` SHA256; `pytest` smoke test; `ruff` lint.
- **Not Yet Implemented / Verified:** Phases 2–8 (landmark tracker, feature normalisation, data collection, classifier training, swipe/actions, live app, benchmark, evaluation).

## Human Gates (Pending)
1. **Human Gate 1 — Real Data Collection (Phase 3):** Record multi-session webcam landmark data (`python -m tghci.collect`).
2. **Human Gate 2 — Live Demo (Phase 6):** Run real-time webcam app (`python -m tghci.app`).
3. **Human Gate 3 — Real Training & Evaluation (Phase 7):** Train on real sessions and run held-out evaluation.
