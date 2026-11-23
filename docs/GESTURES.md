# Gesture Definitions and Feature Specification

## Static Gesture Classes
1. `open_palm`: All five fingers extended and spread.
2. `fist`: Fingers curled into the palm.
3. `pinch`: Thumb tip (landmark 4) and index fingertip (landmark 8) brought together.
4. `neutral`: Relaxed or transitional hand poses not corresponding to any command.

## Dynamic Heuristic Gestures
- `swipe_left` / `swipe_right`: Detected by `tghci.swipe` from rapid horizontal displacement of the hand centre over a short frame window.

## Feature Normalisation Formula
Detailed formula and auxiliary distance ratios will be finalised and documented in Phase 2 (`feat/landmarks`).
