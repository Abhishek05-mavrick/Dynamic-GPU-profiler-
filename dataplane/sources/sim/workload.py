"""
sources/sim/workload.py — Synthetic workload generator (agent.md §8.5).

Every default is flagged [A] (assumed).  The mixture duration model is
implemented so that P90/mean > 3.0 is achievable (see calibration warning
in agent.md §8.5).

Supported arrival modes:
  - homogeneous_poisson  (validation baseline)
  - diurnal              (non-homogeneous Poisson with 24h rate profile)
  - bursty               (compound Poisson with heavy-tailed group size)

Duration model: mixture of a near-zero mass + heavy-tailed exponential.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional

from ...schema.models import (
    DataOrigin, JobClass, JobRecord, JobStatus, hash_tenant,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class WorkloadConfig:
    """All workload parameters — every value is [A] (assumed)."""

    # Arrival
    arrival_mode: str = "homogeneous_poisson"   # homogeneous_poisson | diurnal | bursty
    arrival_rate: float = 2.0                   # jobs/minute for homogeneous Poisson [A]

    # Duration mixture  (see calibration warning in agent.md §8.5)
    short_fraction: float = 0.75               # fraction of near-zero-duration jobs [A]
    short_mean_s: float = 5.0                  # mean of short component (seconds) [A]
    long_mean_s: float = 600.0                 # mean of long exponential tail [A]

    # GPU demand distribution [A]
    gpu_frac_probs: tuple = (0.3, 0.2, 0.2, 0.15, 0.1, 0.05)
    gpu_frac_values: tuple = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)

    # Job class distribution [A]
    class_probs: tuple = (0.5, 0.35, 0.15)     # training, inference, interactive
    class_values: tuple = ("training", "inference", "interactive")

    # Tenants
    num_tenants: int = 5
    tenant_zipf_alpha: float = 1.2             # Zipf activity distribution [A]

    # Estimate noise
    timelimit_lognormal_sigma: float = 0.5     # log-normal noise on time limits [A]

    # Failure probability
    failure_prob: float = 0.02                 # per-job failure probability [A]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class WorkloadGenerator:
    """
    Generate a list of JobRecord objects from a WorkloadConfig.

    All randomness flows through a single seeded RNG for determinism.
    """

    def __init__(self, config: WorkloadConfig, seed: int = 42):
        self.cfg = config
        self._rng = random.Random(seed)
        # Build tenant pool with Zipf weights
        self._tenants = [f"tenant_{i:03d}" for i in range(config.num_tenants)]
        self._tenant_weights = self._zipf_weights(config.num_tenants, config.tenant_zipf_alpha)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, n_jobs: int, start_ts: float = 0.0) -> List[JobRecord]:
        """Generate `n_jobs` job records starting from `start_ts`."""
        arrivals = self._arrival_times(n_jobs, start_ts)
        jobs = []
        for i, ts in enumerate(arrivals):
            job = self._make_job(job_idx=i, submit_ts=ts)
            jobs.append(job)
        return jobs

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _arrival_times(self, n: int, start: float) -> List[float]:
        mode = self.cfg.arrival_mode
        rate_per_s = self.cfg.arrival_rate / 60.0

        if mode == "homogeneous_poisson":
            return self._homogeneous_poisson(n, start, rate_per_s)
        elif mode == "diurnal":
            return self._diurnal_poisson(n, start, rate_per_s)
        elif mode == "bursty":
            return self._bursty_poisson(n, start, rate_per_s)
        else:
            raise ValueError(f"Unknown arrival_mode: {mode}")

    def _homogeneous_poisson(self, n: int, start: float, rate: float) -> List[float]:
        times = []
        t = start
        while len(times) < n:
            t += self._rng.expovariate(rate)
            times.append(t)
        return times

    def _diurnal_poisson(self, n: int, start: float, base_rate: float) -> List[float]:
        """Thinning method for non-homogeneous Poisson with a diurnal profile."""
        times = []
        t = start
        max_rate = base_rate * 3.0   # peak is 3× mean [A]
        while len(times) < n:
            t += self._rng.expovariate(max_rate)
            # Rate multiplier: sinusoidal 24h cycle (peak at 14:00 UTC) [A]
            hour = (t / 3600) % 24
            multiplier = 0.3 + 0.7 * (1 + math.sin(2 * math.pi * (hour - 14) / 24)) / 2
            if self._rng.random() < multiplier:
                times.append(t)
        return times

    def _bursty_poisson(self, n: int, start: float, base_rate: float) -> List[float]:
        """Compound Poisson — sweeps of heavy-tailed group size."""
        times: List[float] = []
        t = start
        burst_rate = base_rate / 5.0   # bursts arrive less frequently [A]
        while len(times) < n:
            t += self._rng.expovariate(burst_rate)
            # Heavy-tailed group size: Pareto with shape 1.5 [A]
            group_size = max(1, int(self._rng.paretovariate(1.5)))
            for offset in range(group_size):
                times.append(t + offset * 0.5)
                if len(times) >= n:
                    break
        return times[:n]

    def _sample_duration(self) -> float:
        """Sample from the mixture model (see agent.md §8.5 calibration warning)."""
        if self._rng.random() < self.cfg.short_fraction:
            # Near-zero component
            return self._rng.expovariate(1.0 / self.cfg.short_mean_s)
        else:
            # Heavy-tailed component
            return self._rng.expovariate(1.0 / self.cfg.long_mean_s)

    def _sample_gpu_frac(self) -> float:
        return self._rng.choices(
            self.cfg.gpu_frac_values,
            weights=self.cfg.gpu_frac_probs,
        )[0]

    def _sample_class(self) -> JobClass:
        name = self._rng.choices(
            self.cfg.class_values,
            weights=self.cfg.class_probs,
        )[0]
        return JobClass(name)

    def _sample_tenant(self) -> str:
        raw = self._rng.choices(self._tenants, weights=self._tenant_weights)[0]
        return hash_tenant(raw)

    def _make_job(self, job_idx: int, submit_ts: float) -> JobRecord:
        gpu_frac  = self._sample_gpu_frac()
        duration  = self._sample_duration()
        job_class = self._sample_class()
        tenant    = self._sample_tenant()

        # Noisy time-limit estimate (log-normal) [A]
        sigma = self.cfg.timelimit_lognormal_sigma
        timelimit_noise = math.exp(self._rng.gauss(0, sigma))
        timelimit_s = duration * timelimit_noise

        # CPU and memory correlated with GPU demand [A]
        req_cpu = max(1.0, gpu_frac * 4.0 + self._rng.gauss(0, 1.0))
        req_mem = max(1.0, gpu_frac * 16.0 + self._rng.gauss(0, 2.0))

        return JobRecord(
            job_id=f"job_{job_idx:06d}",
            instance_id=f"inst_{job_idx:06d}",
            submit_ts=submit_ts,
            start_ts=None,
            end_ts=None,
            status=JobStatus.PENDING,
            req_gpu_frac=float(gpu_frac),
            req_gpu_count=max(1, math.ceil(gpu_frac)) if gpu_frac > 0 else 0,
            req_cpu_cores=round(req_cpu, 2),
            req_mem_gib=round(req_mem, 2),
            timelimit_s=timelimit_s,
            tenant_id=tenant,
            job_class=job_class,
            data_origin=DataOrigin.SYNTHETIC_SIM,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _zipf_weights(n: int, alpha: float) -> List[float]:
        weights = [1.0 / (i ** alpha) for i in range(1, n + 1)]
        total = sum(weights)
        return [w / total for w in weights]
