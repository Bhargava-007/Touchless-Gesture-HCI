# Gesture Definitions and Feature Specification

## Static Gesture Classes
1. `open_palm`: All five fingers extended and spread.
2. `fist`: Fingers curled into the palm.
3. `pinch`: Thumb tip (landmark 4) and index fingertip (landmark 8) brought together.
4. `neutral`: Relaxed or transitional hand poses not corresponding to any command.

## Dynamic Heuristic Gestures (`tghci.swipe`)
`swipe_left` and `swipe_right` are detected by a **rule-based heuristic** (not a learned model) over a sliding window of `window_size` hand-centre observations $(x_t, y_t, t)$:
- Horizontal displacement $|\Delta x| \ge \text{min\_travel\_norm}$
- Horizontal velocity $|\Delta x| / \Delta t \ge \text{min\_velocity\_norm}$
- Vertical dominance guard $|\Delta y| / |\Delta x| \le \text{max\_vertical\_ratio}$
- Refractory period $\text{refractory\_seconds}$ after each trigger

All thresholds are configured in `configs/default.yaml`.

## Feature Normalisation Formula (`tghci.features`)
Given raw 3D landmarks $P \in \mathbb{R}^{21 \times 3}$:

1. **Wrist-origin translation:**
   $$P' = P - P_0$$
   where $P_0$ is the wrist landmark (index `0`).
2. **Left-hand mirroring (when `handedness == "Left"` and `mirror_left_hand=True`):**
   $$P'_x \leftarrow -P'_x$$
3. **Scale normalisation:**
   $$s = \max\left(\|P'_9\|_2,\ \epsilon\right), \quad \hat{P} = \frac{P'}{s}$$
   where $P'_9$ is the middle-finger MCP landmark (index `9`) and $\epsilon = 10^{-6}$ (falling back to $\max_i \|P'_i\|_2$ if $\|P'_9\|_2 < \epsilon$).
4. **Flattened base vector:** $\hat{P}.\text{reshape}(63)$ (`float32`).
5. **Auxiliary distance features (5 values, `float32`):**
   - $\|\hat{P}_4 - \hat{P}_8\|_2$ (thumb-index tip distance, pinch cue)
   - $\|\hat{P}_4 - \hat{P}_{12}\|_2$ (thumb-middle tip distance)
   - $\|\hat{P}_8\|_2$ (index tip to wrist)
   - $\|\hat{P}_{12}\|_2$ (middle tip to wrist)
   - $\|\hat{P}_{20}\|_2$ (pinky tip to wrist)

Total feature dimension: $63 + 5 = 68$ (`float32`).
