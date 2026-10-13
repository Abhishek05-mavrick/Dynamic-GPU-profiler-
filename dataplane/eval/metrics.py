"""
eval/metrics.py — Evaluation metrics (agent.md §12).

All metrics follow the survey definitions [V]:
  C_j = completion time
  W_j = C_j - r_j - p_j  (waiting time = response - submit - service)
  makespan = max_j C_j
  slowdown = (C_j - r_j) / p_j  (response / service)
  Jain's fairness index = (Σx)² / (n * Σx²)
  SLO attainment = fraction of jobs with W_j ≤ SLO threshold
  utilization = mean fraction of GPU capacity used over time
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..schema.models import JobRecord, JobStatus


@dataclass
class JobMetrics:
    job_id: str
    submit_ts: float
    start_ts: Optional[float]
    end_ts: Optional[float]
    timelimit_s: Optional[float]
    status: JobStatus

    @property
    def completion_time(self) -> Optional[float]:
        return self.end_ts

    @property
    def wait_time(self) -> Optional[float]:
        """W_j = start_ts - submit_ts (queue wait)."""
        if self.start_ts is None:
            return None
        return self.start_ts - self.submit_ts

    @property
    def service_time(self) -> Optional[float]:
        """p_j = end_ts - start_ts."""
        if self.start_ts is None or self.end_ts is None:
            return None
        return self.end_ts - self.start_ts

    @property
    def response_time(self) -> Optional[float]:
        """C_j - r_j = end_ts - submit_ts."""
        if self.end_ts is None:
            return None
        return self.end_ts - self.submit_ts

    @property
    def slowdown(self) -> Optional[float]:
        """(C_j - r_j) / p_j — slowdown vs solo."""
        rt = self.response_time
        svc = self.service_time
        if rt is None or svc is None or svc <= 0:
            return None
        return rt / svc


@dataclass
class RunMetrics:
    """Aggregate metrics for a simulation run."""

    policy: str
    seed: int
    n_jobs_total: int = 0
    n_jobs_completed: int = 0
    n_jobs_failed: int = 0
    n_jobs_pending: int = 0

    # Time metrics (seconds)
    makespan: float = 0.0
    mean_wait_s: float = 0.0
    p95_wait_s: float = 0.0
    mean_response_s: float = 0.0
    mean_slowdown: float = 0.0
    max_slowdown: float = 0.0

    # Fairness
    jains_index: float = 0.0

    # Throughput
    throughput_per_hour: float = 0.0

    # Utilization (mean GPU fraction used over observation window)
    mean_gpu_utilization: float = 0.0

    # SLO attainment
    slo_threshold_s: float = 600.0
    slo_attainment: float = 0.0

    # Isolation
    isolation_violations: int = 0

    # Data origin
    data_origin: str = "synthetic_sim"

    def to_dict(self) -> dict:
        return {
            "policy": self.policy,
            "seed": self.seed,
            "n_jobs_total": self.n_jobs_total,
            "n_jobs_completed": self.n_jobs_completed,
            "makespan_s": self.makespan,
            "mean_wait_s": self.mean_wait_s,
            "p95_wait_s": self.p95_wait_s,
            "mean_response_s": self.mean_response_s,
            "mean_slowdown": self.mean_slowdown,
            "max_slowdown": self.max_slowdown,
            "jains_index": self.jains_index,
            "throughput_per_hour": self.throughput_per_hour,
            "mean_gpu_utilization": self.mean_gpu_utilization,
            "slo_attainment": self.slo_attainment,
            "isolation_violations": self.isolation_violations,
            "data_origin": self.data_origin,
        }


def compute_metrics(
    jobs: List[JobRecord],
    policy: str,
    seed: int = 0,
    sim_duration: float = 3600.0,
    n_gpu_total: float = 1.0,
    isolation_violations: int = 0,
) -> RunMetrics:
    """
    Compute all evaluation metrics from a list of completed job records.

    Args:
        jobs: All job records after simulation
        policy: Policy name tag
        seed: Run seed for traceability
        sim_duration: Total simulation wall time (for throughput)
        n_gpu_total: Total GPU fractions across the cluster (for utilisation)
        isolation_violations: Count from IsolationEngine
    """
    rm = RunMetrics(
        policy=policy,
        seed=seed,
        n_jobs_total=len(jobs),
        isolation_violations=isolation_violations,
    )

    completed = [j for j in jobs if j.status == JobStatus.COMPLETED]
    rm.n_jobs_completed = len(completed)
    rm.n_jobs_failed = sum(1 for j in jobs if j.status == JobStatus.FAILED)
    rm.n_jobs_pending = sum(1 for j in jobs if j.start_ts is None)

    if not completed:
        return rm

    jm_list = [
        JobMetrics(
            job_id=j.job_id,
            submit_ts=j.submit_ts,
            start_ts=j.start_ts,
            end_ts=j.end_ts,
            timelimit_s=j.timelimit_s,
            status=j.status,
        )
        for j in completed
    ]

    # Makespan
    end_times = [jm.end_ts for jm in jm_list if jm.end_ts is not None]
    rm.makespan = max(end_times) if end_times else 0.0

    # Wait times
    wait_times = [jm.wait_time for jm in jm_list if jm.wait_time is not None]
    if wait_times:
        rm.mean_wait_s = statistics.mean(wait_times)
        rm.p95_wait_s = _percentile(wait_times, 95)

    # Response times
    resp_times = [jm.response_time for jm in jm_list if jm.response_time is not None]
    if resp_times:
        rm.mean_response_s = statistics.mean(resp_times)

    # Slowdown
    slowdowns = [jm.slowdown for jm in jm_list if jm.slowdown is not None]
    if slowdowns:
        rm.mean_slowdown = statistics.mean(slowdowns)
        rm.max_slowdown = max(slowdowns)

    # Jain's fairness index on response times [V]
    if resp_times:
        n = len(resp_times)
        sum_x = sum(resp_times)
        sum_x2 = sum(x ** 2 for x in resp_times)
        rm.jains_index = (sum_x ** 2) / (n * sum_x2) if sum_x2 > 0 else 1.0

    # Throughput
    if sim_duration > 0:
        rm.throughput_per_hour = rm.n_jobs_completed / (sim_duration / 3600.0)

    # GPU utilisation proxy: sum of completed job GPU-seconds / (sim_duration * n_gpu_total)
    gpu_seconds = sum(
        (j.end_ts - j.start_ts) * j.req_gpu_frac
        for j in completed
        if j.start_ts is not None and j.end_ts is not None
    )
    if sim_duration > 0 and n_gpu_total > 0:
        rm.mean_gpu_utilization = min(gpu_seconds / (sim_duration * n_gpu_total), 1.0)

    # SLO attainment
    slo = rm.slo_threshold_s
    slo_hits = sum(1 for wt in wait_times if wt <= slo)
    rm.slo_attainment = slo_hits / len(wait_times) if wait_times else 0.0

    return rm


def compare_policies(results: List[RunMetrics]) -> str:
    """Format a comparison table of multiple policy results."""
    header = (
        f"{'Policy':<22} {'Completed':>9} {'MeanWait(s)':>12} "
        f"{'P95Wait(s)':>11} {'MeanSlowdown':>13} {'Utilization':>12} {'SLO%':>6} {'Jain':>6}"
    )
    sep = "-" * len(header)
    rows = [header, sep]
    for r in sorted(results, key=lambda x: x.mean_wait_s):
        rows.append(
            f"{r.policy:<22} {r.n_jobs_completed:>9} {r.mean_wait_s:>12.1f} "
            f"{r.p95_wait_s:>11.1f} {r.mean_slowdown:>13.2f} "
            f"{r.mean_gpu_utilization:>12.2%} {r.slo_attainment:>5.1%} {r.jains_index:>6.3f}"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile(data: List[float], p: int) -> float:
    """Compute the p-th percentile of data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (p / 100.0) * (len(sorted_data) - 1)
    lower = int(idx)
    upper = lower + 1
    if upper >= len(sorted_data):
        return sorted_data[-1]
    frac = idx - lower
    return sorted_data[lower] * (1 - frac) + sorted_data[upper] * frac
