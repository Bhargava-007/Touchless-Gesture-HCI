# Measured Results

All numbers in this file come directly from commands executed in this repository with raw outputs saved in `results/`.

## CPU vs GPU Inference Latency Benchmark
Source CSV: `results/benchmark_20261124_035254.csv` (random-initialised `GestureMLP` of the same `68 -> 128 -> 64 -> 4` architecture; 1000 iterations, 50 warmup iterations, `torch==2.11.0+cu128`, CUDA `12.8`, GPU `NVIDIA GeForce RTX 5050 Laptop GPU`).

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

**Observation:** Observed CPU faster than CUDA at all four batch sizes (1, 8, 64, 512); the cause was not measured here and GPU kernel-launch overhead is an untested hypothesis.

**What is measured:** `benchmark_device` allocates `x = torch.randn(batch_size, input_dim, dtype=torch.float32, device=device)` once before the warm-up and timed loops, so the timed loop measures `model(x)` forward pass only (plus `torch.cuda.synchronize()` on CUDA). Host-to-device transfer, output readback, feature preprocessing, and MediaPipe landmark inference are not included.

**Provenance & Threads:** In `results/benchmark_20261124_035254.csv`, `git_commit` (`85d563b40a5a6e710d8102e8327ced4daec8f971`) is the base commit before the benchmark code was committed (working tree was dirty); the CSV itself is unchanged. PyTorch CPU thread count on this machine: `torch.get_num_threads() = 8`.

## Held-Out Test Set Evaluation
_PENDING: requires human-collected data (Human Gate 1 & Human Gate 3)._

| Gesture | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `open_palm` | _pending_ | _pending_ | _pending_ | - |
| `fist` | _pending_ | _pending_ | _pending_ | - |
| `pinch` | _pending_ | _pending_ | _pending_ | - |
| `neutral` | _pending_ | _pending_ | _pending_ | - |
