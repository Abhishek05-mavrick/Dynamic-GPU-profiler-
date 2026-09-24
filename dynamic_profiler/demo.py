"""
Demo / Test script for the Dynamic GPU Profiler.

Run this to verify the profiler works on your GPU:
    python -m dynamic_profiler.demo

Or from Jupyter:
    %run dynamic_profiler/demo.py

Compatible with any NVIDIA GPU (RTX 2050, RTX 3090, A100, H100, H200).
"""

import torch
import sys
import os

# Add parent dir to path so we can import dynamic_profiler
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dynamic_profiler import (
    GPUConfig,
    build_fingerprint,
    predict_interference,
)
from dynamic_profiler.interference_predictor import explain_interference


def main():
    print("=" * 60)
    print("  Dynamic GPU Profiler - Demo")
    print("=" * 60)
    print()

    # Step 1: Detect GPU
    gpu = GPUConfig.detect()
    print(gpu)
    print()

    # Step 2: Create test kernels
    device = torch.device(f"cuda:{gpu.device_index}")

    # Kernel A: Matrix multiplication (compute-heavy)
    # Use smaller matrices for RTX 2050 (4 GB) vs larger GPUs
    mat_size = 1024 if gpu.total_memory_bytes > 6 * 1024**3 else 512
    mat_a = torch.randn(mat_size, mat_size, device=device)
    mat_b = torch.randn(mat_size, mat_size, device=device)
    mm_kernel = lambda: torch.mm(mat_a, mat_b)

    # Kernel B: Large tensor copy (memory-heavy)
    # Use 5% of GPU memory for copy buffer
    copy_elements = min(
        gpu.safe_test_array_bytes(fraction=0.05) // 4,  # float32
        10_000_000,  # Cap at 40 MB
    )
    copy_src = torch.randn(copy_elements, device=device)
    copy_bytes = copy_elements * 4  # float32
    copy_kernel = lambda: copy_src.clone()

    # Kernel C: Element-wise operation (balanced)
    ewise_size = min(copy_elements, 5_000_000)
    ewise_tensor = torch.randn(ewise_size, device=device)
    ewise_kernel = lambda: torch.relu(ewise_tensor) * 2.0 + 1.0

    # Step 3: Build fingerprints
    print("-" * 60)
    print("Building kernel fingerprints...")
    print("-" * 60)
    print()

    fp_mm = build_fingerprint(
        f"torch.mm_{mat_size}x{mat_size}",
        mm_kernel,
        gpu=gpu,
        num_runs=8,
    )
    print(fp_mm)
    print()

    fp_copy = build_fingerprint(
        f"tensor.clone_{copy_elements // 1_000_000}M",
        copy_kernel,
        gpu=gpu,
        data_bytes=copy_bytes,
        num_runs=8,
    )
    print(fp_copy)
    print()

    fp_ewise = build_fingerprint(
        f"relu_scale_{ewise_size // 1_000_000}M",
        ewise_kernel,
        gpu=gpu,
        num_runs=8,
    )
    print(fp_ewise)
    print()

    # Step 4: Predict interference between all pairs
    print("-" * 60)
    print("Interference Predictions")
    print("-" * 60)
    print()

    pairs = [
        (fp_mm, fp_copy, "MatMul vs Copy"),
        (fp_mm, fp_ewise, "MatMul vs Elementwise"),
        (fp_copy, fp_ewise, "Copy vs Elementwise"),
        (fp_mm, fp_mm, "MatMul vs MatMul (self)"),
        (fp_copy, fp_copy, "Copy vs Copy (self)"),
    ]

    for fp_a, fp_b, label in pairs:
        print(f"--- {label} ---")
        print(explain_interference(fp_a, fp_b))
        print()

    # Step 5: Save fingerprints
    save_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "fingerprints"
    )
    os.makedirs(save_dir, exist_ok=True)

    for fp in [fp_mm, fp_copy, fp_ewise]:
        path = os.path.join(save_dir, f"{fp.name.replace(' ', '_')}.json")
        fp.save(path)
        print(f"Saved: {path}")

    print()
    print("=" * 60)
    print("  Demo complete! All fingerprints saved to fingerprints/")
    print("=" * 60)


if __name__ == "__main__":
    main()
