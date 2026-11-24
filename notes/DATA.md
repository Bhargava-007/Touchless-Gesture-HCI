# Data & Model Asset Documentation

## Official MediaPipe Hand Landmarker Asset
- **URL:** `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`
- **Local path (gitignored):** `models/hand_landmarker.task`
- **Size:** `7,819,105` bytes
- **SHA256:** `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`
- **Download command:** `python scripts/download_hand_model.py`

## Privacy Note
No raw webcam images or video frames are saved to disk. `tghci.collect` writes only normalised 3D hand landmark coordinates and session metadata to `data/raw/*.csv` (which is gitignored).

## Team Data Collection Protocol (Human Gate 1)
To prevent session-level overfitting and ensure real-world generalisation:
1. **Participants & Sessions:** Each of the 3 team members should record at least **3 independent sessions** (minimum 9 sessions total across the team).
2. **Conditions:** Vary lighting (indoor room light, daylight, side-lit), camera distance (40–90 cm), and hand (both Left and Right).
3. **Classes:** Record balanced samples for all 4 classes (`open_palm`, `fist`, `pinch`, `neutral`), including relaxed/transitional hand poses for `neutral`.
4. **Command:**
   ```powershell
   .\.venv\Scripts\python.exe -m tghci.collect
   ```
   Press `1`–`4` to select a gesture and start the countdown, `Space` to pause, and `q` to save the session CSV to `data/raw/`.

## Session CSV Schema
Each file `data/raw/<session_id>.csv` contains:
- `session_id`: Unique recording session identifier (used for strict session-based splitting).
- `timestamp`: Unix timestamp in seconds.
- `label`: One of `open_palm`, `fist`, `pinch`, `neutral`.
- `handedness`: `Left` or `Right`.
- `confidence`: MediaPipe handedness confidence score.
- `f_0` .. `f_62`: 63 wrist-normalised, scale-normalised landmark coordinates (`21 x 3`).
