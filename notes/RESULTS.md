# Measured Results

All numbers in this file come directly from commands executed in this repository with raw outputs saved in `results/`.

## Training Summary (Real Sessions)
Source CSV: `results/train_log.csv` (checkpoint `models/gesture_mlp.pt`, trained on `cuda`, `seed=42`, 5 recorded sessions / 4,999 total rows: 3 train sessions = 1,901 rows, 1 val session = 1,086 rows, 1 test session = 2,012 rows).

- **Epochs run:** 36 (early stopping triggered with patience 12; best epoch = 24)
- **Best validation loss:** `0.1340` (`0.134042` in `results/train_log.csv` epoch 24)
- **Best validation accuracy:** `0.9751`

## Held-Out Test Set Evaluation (Real Session Split)
Source files: `results/eval_20261126_221933.csv` and `results/eval_20261126_221933.meta.json` (held-out test session `sess_20261126_221452_1a06ce`, 2,012 samples; confusion matrices saved to `results/figures/confusion_counts.png` and `results/figures/confusion_norm.png`).

- **Overall held-out test accuracy:** `0.7023` (`0.702286282306163`)

| Gesture | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `open_palm` | 0.3847 | 0.8728 | 0.5341 | 283 |
| `fist` | 0.9212 | 0.7946 | 0.8532 | 662 |
| `pinch` | 0.8812 | 0.6334 | 0.7370 | 761 |
| `neutral` | 0.6270 | 0.5163 | 0.5663 | 306 |

## CPU vs GPU Inference Latency Benchmark

### 1. Random-Initialized Weights (`results/benchmark_20261124_035254.csv`)
Random-initialised `GestureMLP` (`68 -> 128 -> 64 -> 4`; 1000 iterations, 50 warmup iterations, `torch==2.11.0+cu128`, CUDA `12.8`, GPU `NVIDIA GeForce RTX 5050 Laptop GPU`).

| Device | Batch Size | Iterations | Mean (ms) | Median (ms) | p95 (ms) | p99 (ms) | Throughput (samples/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| cpu | 1 | 1000 | 0.0274 | 0.0272 | 0.0278 | 0.0314 | 36554.9 |
| cpu | 8 | 1000 | 0.0357 | 0.0333 | 0.0464 | 0.0605 | 224232.2 |
| cpu | 64 | 1000 | 0.0527 | 0.0471 | 0.0694 | 0.0971 | 1214405.1 |
| cpu | 512 | 1000 | 0.1197 | 0.1168 | 0.1514 | 0.1913 | 4278128.5 |
| cuda | 1 | 1000 | 0.1162 | 0.1057 | 0.1580 | 0.1972 | 8606.3 |
| cuda | 8 | 1000 | 0.1242 | 0.1056 | 0.1529 | 0.2490 | 64409.3 |
| cuda | 64 | 1000 | 0.1412 | 0.1323 | 0.1923 | 0.2208 | 453359.6 |
| cuda | 512 | 1000 | 0.1302 | 0.1233 | 0.1680 | 0.2111 | 3933369.3 |

### 2. Trained Checkpoint Weights (`results/benchmark_20261126_221940.csv`)
Trained `GestureMLP` checkpoint (`models/gesture_mlp.pt`, `weights_note=checkpoint`; 1000 iterations, 50 warmup iterations, `torch==2.11.0+cu128`, CUDA `12.8`, GPU `NVIDIA GeForce RTX 5050 Laptop GPU`).

| Device | Batch Size | Iterations | Mean (ms) | Median (ms) | p95 (ms) | p99 (ms) | Throughput (samples/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| cpu | 1 | 1000 | 0.0664 | 0.0532 | 0.1197 | 0.1897 | 15049.7 |
| cpu | 8 | 1000 | 0.0875 | 0.0792 | 0.1580 | 0.2663 | 91464.8 |
| cpu | 64 | 1000 | 0.1109 | 0.0995 | 0.1813 | 0.3136 | 577069.4 |
| cpu | 512 | 1000 | 0.2547 | 0.2406 | 0.3404 | 0.4595 | 2009965.0 |
| cuda | 1 | 1000 | 0.6155 | 0.5633 | 1.0428 | 1.3938 | 1624.6 |
| cuda | 8 | 1000 | 0.4854 | 0.4550 | 1.0499 | 1.3417 | 16482.4 |
| cuda | 64 | 1000 | 0.4669 | 0.4189 | 0.7199 | 1.0768 | 137071.1 |
| cuda | 512 | 1000 | 0.4945 | 0.4346 | 0.9092 | 1.2958 | 1035317.9 |

**Observation:** Observed CPU faster than CUDA at all four batch sizes (1, 8, 64, 512) in both benchmark runs; the cause was not measured here and GPU kernel-launch overhead is an untested hypothesis.

**What is measured:** `benchmark_device` allocates `x = torch.randn(batch_size, input_dim, dtype=torch.float32, device=device)` once before the warm-up and timed loops, so the timed loop measures `model(x)` forward pass only (plus `torch.cuda.synchronize()` on CUDA). Host-to-device transfer, output readback, feature preprocessing, and MediaPipe landmark inference are not included.

**Provenance & Threads:** In `results/benchmark_20261124_035254.csv`, `git_commit` (`85d563b40a5a6e710d8102e8327ced4daec8f971`) is the base commit before the benchmark code was committed (working tree was dirty); in `results/benchmark_20261126_221940.csv`, `git_commit` is `f24efd0bbf4ae676619b94f110758091a5fed317-dirty` (run after training before committing results). PyTorch CPU thread count on this machine: `torch.get_num_threads() = 8`.
