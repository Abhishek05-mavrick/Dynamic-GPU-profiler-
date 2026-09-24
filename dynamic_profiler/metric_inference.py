"""
Metric Inference — Infer GPU hardware metrics from timing experiments.

Since Jupyter environments can't access CUPTI hardware counters directly,
we infer interference-relevant metrics by running controlled timing experiments:

1. SM Saturation:      colocate kernel with itself → slowdown ratio
2. Memory Bandwidth:   known data size / measured time → achieved BW
3. Compute vs Memory:  colocate with known compute/memory reference kernels
4. L2 Sensitivity:     sweep data sizes across L2 boundary → latency knee
"""

import torch
from typing import Callable, Tuple, Dict, List, Optional

from .gpu_config import GPUConfig
from .kernel_timer import (
    measure_kernel_alone,
    measure_kernels_colocated,
)


def infer_sm_saturation(
    kernel_fn: Callable,
    args: tuple = (),
    num_runs: int = 10,
) -> Dict[str, float]:
    """
    Infer SM saturation by colocating a kernel with itself.

    Logic (from Paper §4.1.1 — Thread Block Scheduler):
    - If colocated time ≈ alone time → kernel uses few SMs, room to share
    - If colocated time ≈ 2× alone time → kernel saturates all SMs, serialized

    Returns:
        Dict with 'alone_ms', 'colocated_ms', 'slowdown_ratio', 'sm_saturation_score'
    """
    alone_ms = measure_kernel_alone(kernel_fn, args, num_runs=num_runs)

    makespan, _, _ = measure_kernels_colocated(
        kernel_fn, args, kernel_fn, args, num_runs=num_runs
    )

    if alone_ms <= 0:
        slowdown = 1.0
    else:
        slowdown = makespan / alone_ms

    # Score: 0.0 = no saturation (slowdown ≈ 1.0), 1.0 = full (slowdown ≈ 2.0)
    score = min(max(slowdown - 1.0, 0.0), 1.0)

    return {
        "alone_ms": alone_ms,
        "colocated_ms": makespan,
        "slowdown_ratio": slowdown,
        "sm_saturation_score": score,
    }


def infer_memory_bandwidth(
    kernel_fn: Callable,
    args: tuple,
    data_bytes: int,
    gpu: GPUConfig,
    num_runs: int = 10,
    read_write_factor: float = 2.0,
) -> Dict[str, float]:
    """
    Infer achieved memory bandwidth from timing and known data size.

    Logic (from Paper §4.1.3 — Memory Bandwidth):
    - bandwidth = (data_bytes × read_write_factor × iterations) / time
    - score = achieved_bw / peak_bw

    Args:
        kernel_fn: Kernel to profile
        args: Kernel arguments
        data_bytes: Total bytes the kernel reads+writes per invocation
        gpu: GPU configuration (for peak bandwidth)
        read_write_factor: Multiplier for read+write (default 2.0 for copy)

    Returns:
        Dict with 'achieved_bw_gbps', 'peak_bw_gbps', 'memory_bw_score'
    """
    duration_ms = measure_kernel_alone(kernel_fn, args, num_runs=num_runs)
    duration_sec = duration_ms / 1000.0

    if duration_sec <= 0:
        return {
            "achieved_bw_gbps": 0.0,
            "peak_bw_gbps": gpu.peak_bandwidth_gbps,
            "memory_bw_score": 0.0,
        }

    total_bytes = data_bytes * read_write_factor
    achieved_bw_gbps = (total_bytes / (1024 ** 3)) / duration_sec

    score = min(achieved_bw_gbps / gpu.peak_bandwidth_gbps, 1.0)

    return {
        "achieved_bw_gbps": achieved_bw_gbps,
        "peak_bw_gbps": gpu.peak_bandwidth_gbps,
        "memory_bw_score": score,
    }


def _make_compute_reference_kernel(device: torch.device):
    """
    Create a pure compute-bound reference kernel.
    Uses repeated matrix multiply on small matrices to saturate ALUs.
    Sized to work on GPUs with as little as 4 GB (RTX 2050).
    """
    # Small matrices → compute bound, not memory bound
    size = 256  # 256×256 FP32 = 256 KB per matrix — safe for any GPU
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    def kernel():
        # Chain of matmuls to keep compute units busy
        result = a
        for _ in range(10):
            result = torch.mm(result, b)
        return result

    return kernel


def _make_memory_reference_kernel(device: torch.device, gpu: GPUConfig):
    """
    Create a pure memory-bound reference kernel.
    Uses large tensor copy to saturate memory bandwidth.
    Sized conservatively to fit in GPU memory (uses 10% of total).
    """
    # Use 10% of GPU memory for the copy buffer — safe for RTX 2050 (4 GB → 400 MB)
    num_elements = min(
        gpu.safe_test_array_bytes(fraction=0.10) // 4,  # float32 = 4 bytes
        50_000_000,  # Cap at 200 MB to avoid OOM on small GPUs
    )
    src = torch.randn(num_elements, device=device)

    def kernel():
        return src.clone()

    return kernel


def infer_compute_vs_memory_bound(
    kernel_fn: Callable,
    args: tuple,
    gpu: GPUConfig,
    num_runs: int = 8,
) -> Dict[str, float]:
    """
    Infer whether a kernel is compute-bound or memory-bound by colocating
    it with known reference kernels.

    Logic (from Paper §3 — Pitfall 1: Compute/Memory classification):
    - Colocate target with a pure compute kernel → measure slowdown
    - Colocate target with a pure memory kernel → measure slowdown
    - Higher slowdown with compute ref → target is compute-bound
    - Higher slowdown with memory ref → target is memory-bound

    Returns:
        Dict with 'compute_sensitivity', 'memory_sensitivity', 'bound_type'
    """
    device = torch.device(f"cuda:{gpu.device_index}")
    alone_ms = measure_kernel_alone(kernel_fn, args, num_runs=num_runs)

    # Create reference kernels
    compute_ref = _make_compute_reference_kernel(device)
    memory_ref = _make_memory_reference_kernel(device, gpu)

    # Colocate with compute reference
    makespan_with_compute, _, _ = measure_kernels_colocated(
        kernel_fn, args, compute_ref, (), num_runs=num_runs
    )
    compute_slowdown = makespan_with_compute / alone_ms if alone_ms > 0 else 1.0
    compute_sensitivity = max(compute_slowdown - 1.0, 0.0)

    # Colocate with memory reference
    makespan_with_memory, _, _ = measure_kernels_colocated(
        kernel_fn, args, memory_ref, (), num_runs=num_runs
    )
    memory_slowdown = makespan_with_memory / alone_ms if alone_ms > 0 else 1.0
    memory_sensitivity = max(memory_slowdown - 1.0, 0.0)

    # Classify
    if compute_sensitivity > memory_sensitivity * 1.3:
        bound_type = "compute"
    elif memory_sensitivity > compute_sensitivity * 1.3:
        bound_type = "memory"
    else:
        bound_type = "balanced"

    # Cap sensitivities at 1.0
    return {
        "compute_sensitivity": min(compute_sensitivity, 1.0),
        "memory_sensitivity": min(memory_sensitivity, 1.0),
        "bound_type": bound_type,
    }


def infer_l2_cache_sensitivity(
    data_factory: Callable[[int], Tuple[Callable, tuple]],
    gpu: GPUConfig,
    num_sizes: int = 8,
    num_runs: int = 8,
) -> Dict[str, float]:
    """
    Infer L2 cache sensitivity by sweeping data sizes across L2 boundary.

    Logic (from Paper §4.1.2 — L2 Cache):
    - Run kernel with increasing data sizes
    - When data exceeds L2 capacity, latency jumps → cache misses
    - Ratio of (post-L2 latency / pre-L2 latency) indicates sensitivity

    Args:
        data_factory: Function that takes size_bytes and returns (kernel_fn, args).
                      Example: lambda sz: (lambda: tensor[:sz//4].clone(), ())
        gpu: GPU configuration (for L2 cache size)
        num_sizes: Number of data sizes to sweep
        num_runs: Runs per measurement

    Returns:
        Dict with 'l2_sensitivity_score', 'results' (list of size→latency pairs)
    """
    l2_bytes = gpu.l2_cache_bytes

    # Generate sizes: from 10% of L2 to 5× L2
    sizes = []
    for i in range(num_sizes):
        fraction = 0.1 + (i / max(num_sizes - 1, 1)) * 4.9  # 0.1× to 5.0× L2
        size = int(l2_bytes * fraction)
        # Don't exceed available memory (safe limit = 20% of GPU mem)
        size = min(size, gpu.safe_test_array_bytes(fraction=0.20))
        sizes.append(size)

    results = []
    for size in sizes:
        try:
            kernel_fn, args = data_factory(size)
            latency = measure_kernel_alone(kernel_fn, args, num_runs=num_runs)
            results.append({"size_bytes": size, "latency_ms": latency})
        except RuntimeError:
            # OOM or other CUDA error — stop sweeping
            break

    if len(results) < 2:
        return {"l2_sensitivity_score": 0.0, "results": results}

    # Find the latency ratio between smallest and largest successful size
    baseline_latency = results[0]["latency_ms"]
    max_latency = max(r["latency_ms"] for r in results)

    if baseline_latency <= 0:
        return {"l2_sensitivity_score": 0.0, "results": results}

    # Score: how much latency increased from smallest to largest data size
    # A score of 0.0 = no L2 sensitivity, 1.0 = very sensitive (>3× slowdown)
    ratio = max_latency / baseline_latency
    score = min(max(ratio - 1.0, 0.0) / 2.0, 1.0)  # normalize: 3× → 1.0

    return {
        "l2_sensitivity_score": score,
        "results": results,
    }
