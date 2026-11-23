"""CPU vs GPU inference latency benchmark for GestureMLP."""

from __future__ import annotations

import argparse
import csv
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tghci.config import load_config
from tghci.model import build_model, load_checkpoint


def _git_commit_hash() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


@torch.inference_mode()
def benchmark_device(
    model: torch.nn.Module,
    device: torch.device,
    input_dim: int,
    batch_size: int,
    iterations: int = 1000,
    warmup: int = 50,
) -> dict[str, float]:
    """Measure per-batch inference latency statistics (ms) on ``device``."""
    model = model.to(device)
    model.eval()
    x = torch.randn(batch_size, input_dim, dtype=torch.float32, device=device)

    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies_ms: list[float] = []
    for _ in range(iterations):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    arr = np.asarray(latencies_ms, dtype=np.float64)
    mean_ms = float(np.mean(arr))
    throughput = float(batch_size / (mean_ms / 1000.0)) if mean_ms > 0 else 0.0
    return {
        "mean_ms": mean_ms,
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "throughput_samples_per_s": throughput,
    }


def run_benchmark(
    config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    iterations: int = 1000,
    warmup: int = 50,
    batch_sizes: tuple[int, ...] = (1, 8, 64, 512),
    output_dir: str | Path | None = None,
) -> Path:
    """Run CPU and (if available) CUDA latency benchmark and write a CSV report."""
    cfg = load_config(config_path)
    if checkpoint_path is not None and Path(checkpoint_path).is_file():
        model, _, _ = load_checkpoint(checkpoint_path, device="cpu")
        weights_note = "checkpoint"
    else:
        model = build_model(cfg)
        weights_note = "random_init"

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(output_dir) if output_dir is not None else Path(cfg["paths"]["results_dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / f"benchmark_{stamp}.csv"

    commit = _git_commit_hash()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    cuda_ver = str(torch.version.cuda or "none")
    cpu_name = platform.processor() or platform.machine() or "unknown"
    torch_ver = str(torch.__version__)

    rows: list[dict[str, Any]] = []
    for dev_str in devices:
        dev = torch.device(dev_str)
        for bs in batch_sizes:
            stats = benchmark_device(
                model=model,
                device=dev,
                input_dim=model.input_dim,
                batch_size=bs,
                iterations=iterations,
                warmup=warmup,
            )
            rows.append(
                {
                    "device": dev_str,
                    "batch_size": bs,
                    "iterations": iterations,
                    "mean_ms": f"{stats['mean_ms']:.4f}",
                    "median_ms": f"{stats['median_ms']:.4f}",
                    "p95_ms": f"{stats['p95_ms']:.4f}",
                    "p99_ms": f"{stats['p99_ms']:.4f}",
                    "throughput_samples_per_s": f"{stats['throughput_samples_per_s']:.1f}",
                    "torch_version": torch_ver,
                    "cuda_version": cuda_ver,
                    "gpu_name": gpu_name,
                    "cpu_name": cpu_name,
                    "weights_note": weights_note,
                    "timestamp": stamp,
                    "git_commit": commit,
                }
            )

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Benchmark saved to {csv_path}")
    return csv_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for `python -m tghci.benchmark`."""
    parser = argparse.ArgumentParser(description="Benchmark GestureMLP CPU vs GPU latency")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional trained checkpoint")
    parser.add_argument("--iterations", type=int, default=1000, help="Timed iterations per config")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 64, 512],
        help="Batch sizes to benchmark",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Override results directory")
    args = parser.parse_args(argv)

    run_benchmark(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        iterations=args.iterations,
        warmup=args.warmup,
        batch_sizes=tuple(args.batch_sizes),
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
