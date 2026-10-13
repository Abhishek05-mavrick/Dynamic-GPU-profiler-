# Dynamic GPU Profiler

A lightweight, non-privileged GPU profiling and interference prediction system based on the principles of **"Measuring GPU Utilization One Level Deeper"** ([arXiv:2501.16909](https://arxiv.org/abs/2501.16909)).

Designed to run in **restricted environments** (such as Jupyter notebooks, shared SLURM partitions, and containerized clusters) without requiring `CAP_SYS_ADMIN`, root permissions, or raw hardware performance counter APIs.

---

## Highlights

- **Universal GPU Compatibility**: Automatically inspects device properties at runtime. Pre-calibrated for architectures from mobile/desktop GPUs (**RTX 2050, RTX 3090, RTX 4090**) to data-center accelerators (**A100, H100, H200**).
- **Non-Privileged Timing Inference**: Uses CUDA events (`torch.cuda.Event`) and concurrent multi-stream execution (`torch.cuda.Stream`) to infer low-level resource bottlenecks from latency behavior instead of blocked CUPTI hardware counters.
- **Multi-Dimensional Interference Vectors**: Moves beyond deceptive top-level GPU utilization to analyze the 4 fundamental contention channels:
  1. **Thread Block / SM Scheduler Contention** (Wave serialization)
  2. **DRAM Memory Bandwidth Saturation**
  3. **L2 Cache Thrashing & Sensitivity**
  4. **Warp Scheduler / IPC & Pipeline Pressure**
- **Sub-Millisecond Real-Time Scoring**: Kernel profiling is performed once to generate a JSON fingerprint. Pairwise interference prediction runs in **< 0.1 ms**, making it ready for real-time task schedulers and inference servers.

---

## Directory Structure

```text
Dynamic-GPU-profiler-/
├── README.md
├── dynamic_profiler/
│   ├── __init__.py                # Package exports
│   ├── gpu_config.py              # Hardware auto-detection & spec lookup
│   ├── telemetry.py               # Restricted-VM NVML and host telemetry
│   ├── kernel_timer.py            # CUDA event timing (alone, sequential, colocated)
│   ├── fingerprint.py             # KernelFingerprint data structure & serialization
│   ├── metric_inference.py        # Timing-based inference experiments
│   ├── interference_predictor.py  # Pairwise prediction engine & explanation
│   ├── profiler.py                # Main profiling orchestrator (build_fingerprint)
│   └── demo.py                    # Verification script & benchmark demo
└── fingerprints/                  # Saved JSON fingerprint profiles
```

---

## Installation & Requirements

- Python 3.9+
- NVIDIA GPU with CUDA driver installed
- PyTorch with CUDA support (`torch >= 2.0`)

Install PyTorch with CUDA matching your system:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Quickstart

### Restricted-VM telemetry

The telemetry sampler is designed for fractional H200 VMs and falls back to
RTX/unknown NVIDIA devices. It does not require root, `CAP_SYS_ADMIN`, DCGM
host control, or `nvidia-smi` polling. Install optional bindings in the VM:

```bash
pip install nvidia-ml-py psutil
```

Then collect scheduler samples:

```python
from dynamic_profiler import TelemetrySampler

with TelemetrySampler(device_index=0) as sampler:
    print(sampler.gpu_name)
    print(sampler.capabilities())
    sample = sampler.sample()
    print(sample.to_dict())
```

NVML-backed memory, coarse GPU busy time, and PCIe values are collected when
the VM exposes them. CPU, host memory, and network values come from the host
OS. SM/compute, Tensor, DRAM, and exact GPU bandwidth counters are explicitly
marked unsupported until DCGM or CUPTI is exposed. NVLink is only reported
when links and counters are visible inside the VM; an RTX 2050 normally has no
NVLink.

### 1. Run the Verification Demo
To verify GPU compatibility and profile sample workloads:

```bash
python -m dynamic_profiler.demo
```

### 2. Basic Python / Jupyter Usage

```python
import torch
from dynamic_profiler import GPUConfig, build_fingerprint, predict_interference
from dynamic_profiler.interference_predictor import explain_interference

# 1. Auto-detect GPU configuration
gpu = GPUConfig.detect()
print(gpu)

# 2. Define your CUDA kernels / PyTorch ops
device = torch.device(f"cuda:{gpu.device_index}")
a = torch.randn(1024, 1024, device=device)
b = torch.randn(1024, 1024, device=device)
x = torch.randn(20_000_000, device=device)

matmul_kernel = lambda: torch.mm(a, b)
clone_kernel = lambda: x.clone()

# 3. Profile kernels to generate fingerprints
fp_matmul = build_fingerprint("matmul_1024", matmul_kernel, gpu=gpu)
fp_clone = build_fingerprint(
    "clone_20M",
    clone_kernel,
    gpu=gpu,
    data_bytes=x.nbytes,  # Pass bytes to measure achieved DRAM bandwidth
)

# 4. Predict Co-location Interference
prediction = predict_interference(fp_matmul, fp_clone)
print("Total Interference Score:", prediction["total_score"])
print("Scheduling Recommendation:", prediction["recommendation"])

# 5. Print Detailed Hardware Contention Breakdown
print(explain_interference(fp_matmul, fp_clone))

# 6. Save fingerprints for real-time scheduler lookup
fp_matmul.save("fingerprints/matmul_1024.json")
fp_clone.save("fingerprints/clone_20M.json")
```

---

## Understanding Interference Scores

The predictor outputs a total interference score between `0.0` (zero contention) and `1.0` (severe degradation):

| Total Score | Level | Recommendation | Action for Schedulers |
| :---: | :---: | :--- | :--- |
| **0.00 – 0.24** | `[SAFE]` | Minimal interference expected | Co-locate concurrently on distinct CUDA streams |
| **0.25 – 0.49** | `[MODERATE]` | Moderate interference | Co-locate with telemetry/latency monitoring |
| **0.50 – 0.74** | `[SIGNIFICANT]` | Significant contention | Co-locate only when throughput strictly outweighs tail latency |
| **0.75 – 1.00** | `[HIGH]` | Severe interference / serialization | **Serialize execution** (run one after the other) |

---

## Verified Hardware Environments

- **NVIDIA GeForce RTX 2050 Laptop** (16 SMs, Compute Capability 8.6, 4 GB GDDR6, 128 GB/s)
- **NVIDIA H200 Tensor Core GPU** (132 SMs, Compute Capability 9.0, 141 GB HBM3e, 4.8 TB/s)