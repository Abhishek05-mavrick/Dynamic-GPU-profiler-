"""
Profiler — Main entry point for building kernel fingerprints.

Orchestrates all metric inference experiments to produce a complete
KernelFingerprint for any given CUDA kernel. Designed to run inside
Jupyter notebooks with no special permissions.
"""

import torch
from typing import Callable, Optional

from .gpu_config import GPUConfig
from .fingerprint import KernelFingerprint
from .kernel_timer import measure_kernel_alone
from .metric_inference import (
    infer_sm_saturation,
    infer_memory_bandwidth,
    infer_compute_vs_memory_bound,
)


def build_fingerprint(
    name: str,
    kernel_fn: Callable,
    args: tuple = (),
    gpu: Optional[GPUConfig] = None,
    data_bytes: Optional[int] = None,
    num_runs: int = 10,
    verbose: bool = True,
) -> KernelFingerprint:
    """
    Build a complete interference fingerprint for a CUDA kernel.

    Runs 3-4 timing experiments (~30-60 seconds total) to characterize
    the kernel's resource usage across all interference dimensions.

    Args:
        name: Human-readable name for this kernel (e.g., "torch.mm_1024x1024")
        kernel_fn: Callable that launches the CUDA kernel.
                   Example: lambda: torch.mm(a, b)
        args: Arguments to pass to kernel_fn (default empty)
        gpu: GPUConfig instance. If None, auto-detects.
        data_bytes: If known, total bytes the kernel reads/writes per call.
                    Enables memory bandwidth scoring.
        num_runs: Number of timing runs per experiment (more = more accurate)
        verbose: Print progress messages

    Returns:
        KernelFingerprint with all scores populated

    Example:
        >>> gpu = GPUConfig.detect()
        >>> a = torch.randn(1024, 1024, device='cuda')
        >>> b = torch.randn(1024, 1024, device='cuda')
        >>> fp = build_fingerprint("matmul_1024", lambda: torch.mm(a, b), gpu=gpu)
        >>> print(fp)
    """
    if gpu is None:
        gpu = GPUConfig.detect()

    fp = KernelFingerprint(name=name, gpu_name=gpu.name)

    def log(msg: str):
        if verbose:
            print(msg)

    log(f"Profiling '{name}' on {gpu.name}...")

    # Step 1: Baseline alone time
    log(f"  [1/3] Measuring baseline (alone) latency...")
    fp.alone_time_ms = measure_kernel_alone(kernel_fn, args, num_runs=num_runs)
    log(f"         Alone time: {fp.alone_time_ms:.3f} ms")

    # Step 2: SM saturation (self-colocation)
    log(f"  [2/3] Measuring SM saturation (self-colocation)...")
    sm_result = infer_sm_saturation(kernel_fn, args, num_runs=num_runs)
    fp.sm_saturation_score = sm_result["sm_saturation_score"]
    fp.self_interference = max(sm_result["slowdown_ratio"] - 1.0, 0.0)
    fp.self_interference = min(fp.self_interference, 1.0)
    log(f"         SM saturation: {fp.sm_saturation_score:.2f} "
        f"(slowdown: {sm_result['slowdown_ratio']:.2f}x)")

    # Step 3: Compute vs memory sensitivity
    log(f"  [3/3] Measuring compute/memory sensitivity...")
    bound_result = infer_compute_vs_memory_bound(
        kernel_fn, args, gpu, num_runs=max(num_runs // 2, 4)
    )
    fp.compute_sensitivity = bound_result["compute_sensitivity"]
    fp.memory_sensitivity = bound_result["memory_sensitivity"]
    fp.bound_type = bound_result["bound_type"]
    log(f"         Compute sensitivity: {fp.compute_sensitivity:.2f}")
    log(f"         Memory sensitivity:  {fp.memory_sensitivity:.2f}")
    log(f"         Bound type: {fp.bound_type}")

    # Step 4 (optional): Memory bandwidth scoring
    if data_bytes is not None and data_bytes > 0:
        log(f"  [4/4] Measuring memory bandwidth...")
        bw_result = infer_memory_bandwidth(
            kernel_fn, args, data_bytes, gpu, num_runs=num_runs
        )
        fp.memory_bw_score = bw_result["memory_bw_score"]
        log(f"         Achieved BW: {bw_result['achieved_bw_gbps']:.1f} GB/s "
            f"({fp.memory_bw_score:.0%} of peak)")

    log(f"  Done! Fingerprint ready.")
    log("")

    return fp
