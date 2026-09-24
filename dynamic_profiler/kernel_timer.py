"""
Kernel Timer — Precise CUDA event-based timing for alone, sequential, and colocated execution.

Mirrors the measurement methodology from the gpu-util-interference C++ codebase
(cuda_helper.cuh: run_kernel_alone, run_kernels_sequential, run_kernels_colocated)
but implemented in pure Python using torch.cuda.Event.
"""

import torch
import statistics
from typing import Callable, Any, Tuple, List, Optional


def _warmup_kernel(kernel_fn: Callable, args: tuple, stream: Optional[torch.cuda.Stream] = None):
    """Run kernel once to avoid CUDA lazy-loading artifacts."""
    if stream:
        with torch.cuda.stream(stream):
            kernel_fn(*args)
    else:
        kernel_fn(*args)
    torch.cuda.synchronize()


def measure_kernel_alone(
    kernel_fn: Callable,
    args: tuple = (),
    num_runs: int = 10,
    warmup: int = 2,
) -> float:
    """
    Measure median kernel latency when running alone (no contention).

    This is the baseline measurement. Equivalent to run_kernel_alone() in cuda_helper.cuh.

    Args:
        kernel_fn: Callable that launches a CUDA kernel (e.g., lambda: torch.mm(a, b))
        args: Arguments to pass to kernel_fn
        num_runs: Number of timed runs
        warmup: Number of warmup runs (untimed)

    Returns:
        Median latency in milliseconds
    """
    # Warmup
    for _ in range(warmup):
        kernel_fn(*args)
    torch.cuda.synchronize()

    durations = []
    for _ in range(num_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        kernel_fn(*args)
        end.record()
        torch.cuda.synchronize()

        durations.append(start.elapsed_time(end))

    return statistics.median(durations)


def measure_kernels_sequential(
    kernel_a_fn: Callable,
    args_a: tuple,
    kernel_b_fn: Callable,
    args_b: tuple,
    num_runs: int = 10,
    warmup: int = 2,
) -> float:
    """
    Measure median latency when two kernels run sequentially (one after the other).

    Equivalent to run_kernels_sequential() in cuda_helper.cuh.

    Args:
        kernel_a_fn: First kernel callable
        args_a: Arguments for first kernel
        kernel_b_fn: Second kernel callable
        args_b: Arguments for second kernel
        num_runs: Number of timed runs
        warmup: Number of warmup runs

    Returns:
        Median total sequential latency in milliseconds
    """
    for _ in range(warmup):
        kernel_a_fn(*args_a)
        kernel_b_fn(*args_b)
    torch.cuda.synchronize()

    durations = []
    for _ in range(num_runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        kernel_a_fn(*args_a)
        kernel_b_fn(*args_b)
        end.record()
        torch.cuda.synchronize()

        durations.append(start.elapsed_time(end))

    return statistics.median(durations)


def measure_kernels_colocated(
    kernel_a_fn: Callable,
    args_a: tuple,
    kernel_b_fn: Callable,
    args_b: tuple,
    num_runs: int = 10,
    warmup: int = 2,
) -> Tuple[float, float, float]:
    """
    Measure median makespan when two kernels run concurrently via CUDA streams.

    Equivalent to run_kernels_colocated() in cuda_helper.cuh.
    Uses two non-blocking streams for true concurrent execution.

    Args:
        kernel_a_fn: First kernel callable
        args_a: Arguments for first kernel
        kernel_b_fn: Second kernel callable
        args_b: Arguments for second kernel
        num_runs: Number of timed runs
        warmup: Number of warmup runs

    Returns:
        Tuple of (makespan_ms, time_a_ms, time_b_ms) — all medians
    """
    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()

    # Warmup on both streams (avoid lazy loading — see paper codebase)
    with torch.cuda.stream(stream_a):
        kernel_a_fn(*args_a)
    with torch.cuda.stream(stream_b):
        kernel_b_fn(*args_b)
    torch.cuda.synchronize()

    # Additional warmups
    for _ in range(warmup):
        with torch.cuda.stream(stream_a):
            kernel_a_fn(*args_a)
        with torch.cuda.stream(stream_b):
            kernel_b_fn(*args_b)
        torch.cuda.synchronize()

    makespans = []
    times_a = []
    times_b = []

    for _ in range(num_runs):
        start_a = torch.cuda.Event(enable_timing=True)
        end_a = torch.cuda.Event(enable_timing=True)
        start_b = torch.cuda.Event(enable_timing=True)
        end_b = torch.cuda.Event(enable_timing=True)

        # Launch both kernels concurrently on separate streams
        with torch.cuda.stream(stream_a):
            start_a.record(stream_a)
            kernel_a_fn(*args_a)
            end_a.record(stream_a)

        with torch.cuda.stream(stream_b):
            start_b.record(stream_b)
            kernel_b_fn(*args_b)
            end_b.record(stream_b)

        torch.cuda.synchronize()

        time_a = start_a.elapsed_time(end_a)
        time_b = start_b.elapsed_time(end_b)

        # Compute makespan = max span across both streams
        # Same logic as cuda_helper.cuh lines 280-288
        makespan_1 = start_a.elapsed_time(end_b)
        makespan_2 = start_b.elapsed_time(end_a)
        makespan = max(makespan_1, makespan_2, time_a, time_b)

        makespans.append(makespan)
        times_a.append(time_a)
        times_b.append(time_b)

    return (
        statistics.median(makespans),
        statistics.median(times_a),
        statistics.median(times_b),
    )


def compute_slowdown(alone_ms: float, colocated_makespan_ms: float) -> float:
    """
    Compute slowdown ratio from co-location.

    Returns:
        Slowdown ratio (1.0 = no slowdown, 2.0 = doubled latency)
    """
    if alone_ms <= 0:
        return 1.0
    return colocated_makespan_ms / alone_ms
