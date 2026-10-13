"""
features/builder.py — Point-in-time feature builder (agent.md §9).

Rules:
  - A feature computed for decision time t may only use information with ts <= t.
  - A job's own post-start utilization never appears in its own placement features.
  - Features built in streaming mode equal features rebuilt from the full log.

Feature groups (per agent.md §9 table):
  1. Job request:     req_gpu_frac, req_cpu_cores, req_mem_gib, timelimit, class
  2. Candidate HW:    gpu_model, gpu_count, capacities, mig_enabled, sharing_mode
  3. Node state:      avail_*, running_jobs, node-load from telemetry
  4. Co-location:     resident SM/DRAM demand sums
  5. History:         placeholder (no repeat-job history in v0)
  6. Time context:    hour_of_day, queue_length
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..schema.models import (
    ClusterSnapshot, DataOrigin, JobRecord, NodeRecord, TelemetryRecord,
)
from ..store.event_log import EventLog
from ..ledger.state import ClusterLedger


@dataclass
class FeatureRow:
    """One row per (job_id, candidate_node_id) at decision time t."""

    # Identity
    job_id: str
    node_id: str
    decision_ts: float

    # Group 1: Job request
    req_gpu_frac: float = 0.0
    req_cpu_cores: float = 0.0
    req_mem_gib: float = 0.0
    timelimit_s: float = 0.0
    job_class: str = "unknown"

    # Group 2: Candidate hardware
    gpu_model: str = ""
    gpu_count: int = 0
    node_cpu_cores: float = 0.0
    node_mem_gib: float = 0.0
    mig_enabled: bool = False
    sharing_mode: str = "exclusive"

    # Group 3: Node state at t
    avail_gpu_frac: float = 0.0
    avail_cpu_cores: float = 0.0
    avail_mem_gib: float = 0.0
    running_jobs: int = 0
    node_gpu_util_pct: float = 0.0   # mean telemetry over last 60 s

    # Group 4: Co-location context
    resident_sm_demand: float = 0.0
    resident_dram_demand: float = 0.0

    # Group 6: Time context
    hour_of_day: float = 0.0
    queue_length: int = 0

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "node_id": self.node_id,
            "decision_ts": self.decision_ts,
            "req_gpu_frac": self.req_gpu_frac,
            "req_cpu_cores": self.req_cpu_cores,
            "req_mem_gib": self.req_mem_gib,
            "timelimit_s": self.timelimit_s,
            "job_class": self.job_class,
            "gpu_model": self.gpu_model,
            "gpu_count": self.gpu_count,
            "node_cpu_cores": self.node_cpu_cores,
            "node_mem_gib": self.node_mem_gib,
            "mig_enabled": self.mig_enabled,
            "sharing_mode": self.sharing_mode,
            "avail_gpu_frac": self.avail_gpu_frac,
            "avail_cpu_cores": self.avail_cpu_cores,
            "avail_mem_gib": self.avail_mem_gib,
            "running_jobs": self.running_jobs,
            "node_gpu_util_pct": self.node_gpu_util_pct,
            "resident_sm_demand": self.resident_sm_demand,
            "resident_dram_demand": self.resident_dram_demand,
            "hour_of_day": self.hour_of_day,
            "queue_length": self.queue_length,
        }


class FeatureBuilder:
    """
    Build FeatureRow objects for all (job, candidate_node) pairs at a given t.

    Leakage guard: uses only log.jobs_at(t), log.telemetry_at(node, metric, t),
    and ledger.cluster_snapshot(t) — all of which respect point-in-time cutoff.
    """

    def __init__(self, log: EventLog, ledger: ClusterLedger):
        self._log = log
        self._ledger = ledger

    def build(self, job: JobRecord, t: float) -> List[FeatureRow]:
        """
        Return one FeatureRow per feasible candidate node for `job` at time `t`.

        A node is a candidate if it satisfies GPU type (if specified) and
        has available capacity.
        """
        # Point-in-time leakage guard: job must have submit_ts <= t
        assert job.submit_ts <= t, "Leakage: job submitted after decision time"
        # Leakage guard: job's own utilization is excluded (it hasn't started)
        assert job.start_ts is None or job.start_ts > t, \
            "Leakage: using post-start data for placement features"

        pending = self._log.pending_jobs_at(t)
        queue_length = len(pending)
        hour_of_day = (t / 3600.0) % 24.0

        feasible_nodes = self._ledger.feasible_nodes(job, t)
        rows: List[FeatureRow] = []

        for node_id in feasible_nodes:
            node = self._log.node_at(node_id, t)
            if node is None:
                continue

            # Check GPU type constraint
            if job.req_gpu_type and job.req_gpu_type not in node.gpu_model:
                continue

            avail_gpu, avail_cpu, avail_mem = self._ledger.node_available_at(node_id, t)
            running_jobs = sum(
                1 for j in self._log.running_jobs_at(t, node_id=node_id)
            )

            # Telemetry: mean GPU util over last 60 s (excluding the pending job itself)
            tel_rows = self._log.telemetry_at(node_id, "gpu_util_pct", t, window_s=60.0)
            gpu_util = (
                sum(r.value for r in tel_rows if r.value is not None) / len(tel_rows)
                if tel_rows else 0.0
            )

            # Co-location context: sum demands of current residents
            resident_sm = resident_dram = 0.0
            for resident in self._log.running_jobs_at(t, node_id=node_id):
                # We don't have per-job demand in log; use GPU frac as proxy [A]
                resident_sm   += resident.req_gpu_frac * 0.6   # proxy [A]
                resident_dram += resident.req_gpu_frac * 0.4   # proxy [A]

            rows.append(FeatureRow(
                job_id=job.job_id,
                node_id=node_id,
                decision_ts=t,
                req_gpu_frac=job.req_gpu_frac,
                req_cpu_cores=job.req_cpu_cores,
                req_mem_gib=job.req_mem_gib,
                timelimit_s=job.timelimit_s or 0.0,
                job_class=job.job_class.value,
                gpu_model=node.gpu_model,
                gpu_count=node.gpu_count,
                node_cpu_cores=node.cpu_cores,
                node_mem_gib=node.mem_gib,
                mig_enabled=node.mig_enabled,
                sharing_mode=node.sharing_mode.value,
                avail_gpu_frac=avail_gpu,
                avail_cpu_cores=avail_cpu,
                avail_mem_gib=avail_mem,
                running_jobs=running_jobs,
                node_gpu_util_pct=gpu_util,
                resident_sm_demand=resident_sm,
                resident_dram_demand=resident_dram,
                hour_of_day=hour_of_day,
                queue_length=queue_length,
            ))

        return rows

    def build_all_pending(self, t: float) -> Dict[str, List[FeatureRow]]:
        """Build features for every pending job at time t."""
        pending = self._log.pending_jobs_at(t)
        return {job.job_id: self.build(job, t) for job in pending}
