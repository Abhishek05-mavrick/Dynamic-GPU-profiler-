"""
Interference Predictor — Predict co-location interference between two kernels.

Uses the fingerprint vectors to score how much two kernels will interfere
when run concurrently on the same GPU. The scoring logic maps directly to
the interference sources identified in the paper:
  §4.1.1 Thread Block Scheduler → sm_saturation_score
  §4.1.2 L2 Cache              → l2_sensitivity_score
  §4.1.3 Memory Bandwidth      → memory_bw_score
  §4.2.1 L1 Cache              → memory_sensitivity (partial)
  §4.2.2 IPC / Warp Scheduler  → compute_sensitivity + self_interference
  §4.2.3 Pipeline               → compute_sensitivity (partial)
"""

from .fingerprint import KernelFingerprint
from typing import Dict


def predict_interference(
    fp_a: KernelFingerprint,
    fp_b: KernelFingerprint,
) -> Dict[str, float]:
    """
    Predict interference severity between two kernels.

    Args:
        fp_a: Fingerprint of kernel A
        fp_b: Fingerprint of kernel B

    Returns:
        Dict with:
          - 'total_score': Overall interference (0.0 = safe, 1.0 = severe)
          - 'sm_scheduler': Thread block scheduler contention score
          - 'memory_bw': Memory bandwidth contention score
          - 'l2_cache': L2 cache contention score
          - 'warp_scheduler': Warp scheduler / IPC contention score
          - 'recommendation': Human-readable scheduling recommendation
    """
    scores = {}

    # 1. Thread Block Scheduler interference (Paper §4.1.1)
    # If both kernels saturate SMs, hardware serializes thread block waves
    combined_sm = fp_a.sm_saturation_score + fp_b.sm_saturation_score
    if combined_sm > 1.0:
        scores["sm_scheduler"] = min(combined_sm * 0.5, 1.0)
    else:
        scores["sm_scheduler"] = combined_sm * 0.3

    # 2. Memory Bandwidth interference (Paper §4.1.3)
    # If combined BW usage approaches peak, both kernels stall on DRAM
    combined_bw = fp_a.memory_bw_score + fp_b.memory_bw_score
    if combined_bw > 0.85:
        scores["memory_bw"] = min(combined_bw * 0.7, 1.0)
    else:
        scores["memory_bw"] = combined_bw * 0.3

    # 3. L2 Cache interference (Paper §4.1.2)
    # If both kernels are L2-sensitive, eviction causes cache thrashing
    combined_l2 = fp_a.l2_sensitivity_score + fp_b.l2_sensitivity_score
    scores["l2_cache"] = min(combined_l2 * 0.5, 1.0)

    # 4. Warp Scheduler / IPC / Pipeline interference (Paper §4.2.2, §4.2.3)
    # THE KEY INSIGHT: even "complementary" kernels (compute + memory)
    # can interfere if both have high IPC → warp scheduler saturates.
    # We approximate this via compute_sensitivity and self_interference.
    ipc_signal = (
        fp_a.compute_sensitivity * fp_b.self_interference
        + fp_b.compute_sensitivity * fp_a.self_interference
    ) / 2.0
    scores["warp_scheduler"] = min(ipc_signal, 1.0)

    # 5. Cross-type penalty (Paper §3 — Pitfall 1)
    # If both are the same type (both compute or both memory), higher risk
    if fp_a.bound_type == fp_b.bound_type and fp_a.bound_type != "balanced":
        type_penalty = 0.15
    else:
        type_penalty = 0.0

    # Total: max across all interference channels + type penalty
    total = max(scores.values()) + type_penalty
    total = min(total, 1.0)
    scores["total_score"] = total

    # Recommendation
    if total < 0.25:
        scores["recommendation"] = "[SAFE] Safe to co-locate - minimal interference expected"
    elif total < 0.50:
        scores["recommendation"] = "[MODERATE] Moderate interference - co-locate with monitoring"
    elif total < 0.75:
        scores["recommendation"] = "[SIGNIFICANT] Significant interference - co-locate only if necessary"
    else:
        scores["recommendation"] = "[HIGH] High interference - avoid co-location"

    return scores


def explain_interference(
    fp_a: KernelFingerprint,
    fp_b: KernelFingerprint,
) -> str:
    """
    Generate a human-readable explanation of interference between two kernels.
    """
    result = predict_interference(fp_a, fp_b)

    lines = [
        f"=== Interference Analysis ===",
        f"  Kernel A: {fp_a.name} ({fp_a.bound_type}-bound, alone={fp_a.alone_time_ms:.3f}ms)",
        f"  Kernel B: {fp_b.name} ({fp_b.bound_type}-bound, alone={fp_b.alone_time_ms:.3f}ms)",
        f"",
        f"  --- Interference Channels ---",
    ]

    channel_names = {
        "sm_scheduler":   "SM / Thread Block Scheduler",
        "memory_bw":      "Memory Bandwidth (DRAM)",
        "l2_cache":       "L2 Cache",
        "warp_scheduler": "Warp Scheduler / IPC / Pipeline",
    }

    for key, label in channel_names.items():
        score = result[key]
        bar_len = int(score * 20)
        bar = "#" * bar_len + "-" * (20 - bar_len)
        lines.append(f"  {label:38s} [{bar}] {score:.2f}")

    lines.extend([
        f"",
        f"  Total Score: {result['total_score']:.2f}",
        f"  Recommendation: {result['recommendation']}",
    ])

    return "\n".join(lines)
