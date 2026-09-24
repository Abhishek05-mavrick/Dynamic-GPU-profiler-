"""
Kernel Fingerprint — Data structure representing a kernel's interference profile.

The fingerprint captures how a kernel uses GPU resources across all dimensions
identified in the paper "Measuring GPU Utilization one level deeper" (arXiv:2501.16909).
"""

from dataclasses import dataclass, field
from typing import Optional, List
import json


@dataclass
class KernelFingerprint:
    """
    Interference fingerprint for a CUDA kernel.

    Scores range from 0.0 (no pressure) to 1.0 (fully saturated).
    These are populated by the profiler via timing-based inference.
    """

    # === Identity ===
    name: str = ""
    gpu_name: str = ""

    # === Grid Configuration ===
    grid_dim: tuple = (0, 0, 0)
    block_dim: tuple = (0, 0, 0)

    # === Runtime Baseline ===
    alone_time_ms: float = 0.0

    # === Inter-SM Pressure Scores (GPU-wide contention) ===

    # SM coverage: what fraction of SMs does this kernel occupy?
    # Inferred from self-colocation: if colocated time ≈ 2x alone → score ≈ 1.0
    # Paper §4.1.1 — Thread Block Scheduler
    sm_saturation_score: float = 0.0

    # Memory bandwidth: what fraction of peak BW does this kernel consume?
    # Inferred from timing + known data transfer size
    # Paper §4.1.3 — Memory Bandwidth
    memory_bw_score: float = 0.0

    # L2 cache pressure: how sensitive is this kernel to L2 eviction?
    # Inferred from latency change when data size crosses L2 capacity
    # Paper §4.1.2 — L2 Cache
    l2_sensitivity_score: float = 0.0

    # === Intra-SM Pressure Scores (per-SM contention) ===

    # How much does this kernel slow down when colocated with a compute-heavy kernel?
    # Paper §4.2.2 — IPC / Warp Scheduler, §4.2.3 — Pipeline
    compute_sensitivity: float = 0.0

    # How much does this kernel slow down when colocated with a memory-heavy kernel?
    # Paper §4.2.1 — L1 Cache, §4.1.3 — Memory BW
    memory_sensitivity: float = 0.0

    # How much does this kernel slow down when colocated with itself?
    # Captures total self-interference across all resources
    self_interference: float = 0.0

    # === Derived Classification ===
    # Whether the kernel is primarily compute-bound or memory-bound
    bound_type: str = "unknown"  # "compute", "memory", or "balanced"

    def to_vector(self) -> List[float]:
        """
        Return normalized feature vector for the scheduler.
        Order: [sm_sat, mem_bw, l2_sens, compute_sens, memory_sens, self_interf]
        """
        return [
            self.sm_saturation_score,
            self.memory_bw_score,
            self.l2_sensitivity_score,
            self.compute_sensitivity,
            self.memory_sensitivity,
            self.self_interference,
        ]

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage / caching."""
        return {
            "name": self.name,
            "gpu_name": self.gpu_name,
            "grid_dim": list(self.grid_dim),
            "block_dim": list(self.block_dim),
            "alone_time_ms": self.alone_time_ms,
            "sm_saturation_score": self.sm_saturation_score,
            "memory_bw_score": self.memory_bw_score,
            "l2_sensitivity_score": self.l2_sensitivity_score,
            "compute_sensitivity": self.compute_sensitivity,
            "memory_sensitivity": self.memory_sensitivity,
            "self_interference": self.self_interference,
            "bound_type": self.bound_type,
            "vector": self.to_vector(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KernelFingerprint":
        """Deserialize from dictionary."""
        return cls(
            name=d["name"],
            gpu_name=d.get("gpu_name", ""),
            grid_dim=tuple(d.get("grid_dim", (0, 0, 0))),
            block_dim=tuple(d.get("block_dim", (0, 0, 0))),
            alone_time_ms=d["alone_time_ms"],
            sm_saturation_score=d["sm_saturation_score"],
            memory_bw_score=d["memory_bw_score"],
            l2_sensitivity_score=d.get("l2_sensitivity_score", 0.0),
            compute_sensitivity=d["compute_sensitivity"],
            memory_sensitivity=d["memory_sensitivity"],
            self_interference=d["self_interference"],
            bound_type=d.get("bound_type", "unknown"),
        )

    def save(self, filepath: str):
        """Save fingerprint to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "KernelFingerprint":
        """Load fingerprint from JSON file."""
        with open(filepath, "r") as f:
            return cls.from_dict(json.load(f))

    def summary(self) -> str:
        """Human-readable summary of the fingerprint."""
        bars = {
            "SM Saturation":       self.sm_saturation_score,
            "Memory BW":           self.memory_bw_score,
            "L2 Sensitivity":      self.l2_sensitivity_score,
            "Compute Sensitivity": self.compute_sensitivity,
            "Memory Sensitivity":  self.memory_sensitivity,
            "Self-Interference":   self.self_interference,
        }
        lines = [
            f"=== Kernel Fingerprint: {self.name} ===",
            f"  GPU: {self.gpu_name}",
            f"  Alone time: {self.alone_time_ms:.3f} ms",
            f"  Bound type: {self.bound_type}",
            f"  --- Pressure Scores (0.0 = none, 1.0 = saturated) ---",
        ]
        for label, score in bars.items():
            bar_len = int(score * 20)
            bar = "#" * bar_len + "-" * (20 - bar_len)
            lines.append(f"  {label:22s} [{bar}] {score:.2f}")
        lines.append(f"  Vector: {self.to_vector()}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()
