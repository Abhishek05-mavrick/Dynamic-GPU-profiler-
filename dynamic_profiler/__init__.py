"""
Dynamic GPU Profiler — Jupyter-compatible kernel interference profiler.

Works on any NVIDIA GPU (RTX 2050, RTX 3090, A100, H100, H200).
Auto-detects GPU properties and builds kernel interference fingerprints
using timing-based inference (no CUPTI/root access required).

Usage:
    from dynamic_profiler import GPUConfig, build_fingerprint, predict_interference

    # Auto-detect GPU
    gpu = GPUConfig.detect()
    print(gpu)

    # Profile a kernel
    fp = build_fingerprint("my_kernel", my_kernel_fn, args=(), gpu=gpu)

    # Predict interference between two kernels
    score = predict_interference(fp_a, fp_b)
"""

from .gpu_config import GPUConfig
from .fingerprint import KernelFingerprint
from .kernel_timer import (
    measure_kernel_alone,
    measure_kernels_colocated,
    measure_kernels_sequential,
)
from .metric_inference import (
    infer_sm_saturation,
    infer_memory_bandwidth,
    infer_compute_vs_memory_bound,
    infer_l2_cache_sensitivity,
)
from .interference_predictor import predict_interference
from .profiler import build_fingerprint

__version__ = "0.1.0"
__all__ = [
    "GPUConfig",
    "KernelFingerprint",
    "measure_kernel_alone",
    "measure_kernels_colocated",
    "measure_kernels_sequential",
    "infer_sm_saturation",
    "infer_memory_bandwidth",
    "infer_compute_vs_memory_bound",
    "infer_l2_cache_sensitivity",
    "predict_interference",
    "build_fingerprint",
]
