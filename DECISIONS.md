# Architecture Decision Records (ADRs)

## ADR-001: Clean-Room Implementation
- **Context:** Course project for Deep Learning with Accelerated Computing (25CS3112E).
- **Decision:** Write all application, feature extraction, training, evaluation, and action-mapping code from scratch using only official library documentation (MediaPipe, PyTorch, OpenCV). No third-party gesture repositories are cloned or copied.
- **Consequences:** Full ownership and traceability of every line of code; slightly more upfront engineering effort, zero licensing ambiguity.

## ADR-002: PyTorch over TensorFlow for GPU Acceleration
- **Context:** Target hardware is an NVIDIA GeForce RTX 5050 Laptop GPU (Blackwell, `sm_120`) on native Windows 11.
- **Decision:** Use PyTorch with CUDA (`cu128` wheel) instead of TensorFlow, because TensorFlow dropped native-Windows GPU support after 2.10.
- **Consequences:** Native Windows 11 CUDA execution works without WSL2 or Docker, and `sm_120` is supported on recent PyTorch CUDA 12.8+ builds.

## ADR-003: MediaPipe Tasks `HandLandmarker` API
- **Context:** Recent `mediapipe` releases (`>=1.2`) removed the legacy `mediapipe.solutions` namespace.
- **Decision:** Use `mediapipe.tasks.python.vision.HandLandmarker` with the official `hand_landmarker.task` float16 model bundle.
- **Consequences:** Compatible with modern MediaPipe on Python 3.13; requires downloading `hand_landmarker.task` into `models/` (gitignored).

## ADR-004: Explicit `neutral` Gesture Class
- **Context:** During live webcam use, the user's hand is often visible in relaxed or transitional poses that are not intended as commands.
- **Decision:** Include a fourth `neutral` / other class alongside `open_palm`, `fist`, and `pinch`.
- **Consequences:** Reduces false-positive triggers between intentional gestures; requires collecting negative/relaxed frames during data collection.

## ADR-005: Session-Based Train/Val/Test Split
- **Context:** Consecutive webcam frames within a single recording session are nearly identical, so a random frame-level split leaks near-duplicate poses across train and test sets and inflates accuracy.
- **Decision:** Split strictly by `session_id` (requires at least 3 distinct sessions for train, val, and test).
- **Consequences:** Honest generalisation metrics across sessions/lighting/people; enforces collecting multiple sessions before training.

## ADR-006: Rule-Based Swipe Detector
- **Context:** Swipes are short temporal trajectories of the hand centre rather than static hand shapes.
- **Decision:** Implement swipe detection as a transparent, config-driven heuristic over a sliding window of hand-centre positions (displacement, velocity, vertical-ratio, and refractory period) rather than a sequence neural network.
- **Consequences:** Zero extra training data needed for swipes, predictable latency, and easy threshold tuning in `configs/default.yaml`.

## ADR-007: Safety-First Action Mapper
- **Context:** Sending OS-level keyboard/mouse events from a live camera risks unintended clicks or keystrokes.
- **Decision:** Start disarmed (`start_armed: false`), enable `dry_run: true` by default, require `N`-frame debounce and per-action cooldown, keep `pyautogui.FAILSAFE = True`, and never act when no hand is tracked.
- **Consequences:** Safe to test and demo; user must explicitly press `a` (arm) and `d` (disable dry-run) to send real OS events.
