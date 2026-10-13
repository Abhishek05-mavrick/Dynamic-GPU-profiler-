"""
ledger/state.py — Cluster-state ledger (event-sourced, per agent.md §7).

For instance i placed on node m at time t_i, with d_j = requested demand of j:

    used_before_i = Σ d_j (j on m, start_j ≤ t_i)
                  - Σ d_j (j on m, end_j   ≤ t_i)
                  - d_i
    available_i   = capacity_m - used_before_i
    running_i     = count(start_j ≤ t_i) - count(end_j ≤ t_i) - 1

Implements per-machine sorted arrays and bisect-based cumulative sums.
Invariant tests:
  - available never below zero (without oversubscription flag)
  - reserved never exceeds capacity (without oversubscription)
  - replaying the log reproduces identical snapshots
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..schema.models import (
    JobRecord, NodeRecord, ClusterSnapshot, DataOrigin,
)


@dataclass
class NodeCapacity:
    node_id: str
    gpu_frac: float     # total GPU (gpu_count as fractional capacity)
    cpu_cores: float
    mem_gib: float


@dataclass
class _Event:
    """Internal event representing a demand change on a node."""
    ts: float
    gpu_frac: float
    cpu_cores: float
    mem_gib: float
    job_id: str
    is_start: bool       # True = start (positive demand), False = end (release)


class NodeLedger:
    """Per-node sorted event list with cumulative-sum queries."""

    def __init__(self, capacity: NodeCapacity):
        self.cap = capacity
        self._events: List[_Event] = []    # sorted by ts

    def record_start(self, job: JobRecord) -> None:
        ev = _Event(
            ts=job.start_ts or job.submit_ts,
            gpu_frac=job.req_gpu_frac,
            cpu_cores=job.req_cpu_cores,
            mem_gib=job.req_mem_gib,
            job_id=job.job_id,
            is_start=True,
        )
        bisect.insort(self._events, ev, key=lambda e: e.ts)

    def record_end(self, job: JobRecord, end_ts: float) -> None:
        ev = _Event(
            ts=end_ts,
            gpu_frac=job.req_gpu_frac,
            cpu_cores=job.req_cpu_cores,
            mem_gib=job.req_mem_gib,
            job_id=job.job_id,
            is_start=False,
        )
        bisect.insort(self._events, ev, key=lambda e: e.ts)

    def used_at(self, as_of: float) -> Tuple[float, float, float]:
        """Return (used_gpu_frac, used_cpu, used_mem_gib) at as_of."""
        gpu = cpu = mem = 0.0
        for ev in self._events:
            if ev.ts > as_of:
                break
            sign = +1.0 if ev.is_start else -1.0
            gpu += sign * ev.gpu_frac
            cpu += sign * ev.cpu_cores
            mem += sign * ev.mem_gib
        return max(gpu, 0.0), max(cpu, 0.0), max(mem, 0.0)

    def available_at(self, as_of: float) -> Tuple[float, float, float]:
        """Return (avail_gpu_frac, avail_cpu, avail_mem_gib) at as_of."""
        used_gpu, used_cpu, used_mem = self.used_at(as_of)
        return (
            max(self.cap.gpu_frac - used_gpu, 0.0),
            max(self.cap.cpu_cores - used_cpu, 0.0),
            max(self.cap.mem_gib - used_mem, 0.0),
        )

    def running_count_at(self, as_of: float) -> int:
        """Number of jobs running at as_of."""
        started = sum(1 for e in self._events if e.is_start and e.ts <= as_of)
        ended   = sum(1 for e in self._events if not e.is_start and e.ts <= as_of)
        return max(started - ended, 0)

    def check_fits(self, job: JobRecord, at: float) -> bool:
        """Return True if job fits without exceeding capacity."""
        avail_gpu, avail_cpu, avail_mem = self.available_at(at)
        return (
            job.req_gpu_frac  <= avail_gpu + 1e-9
            and job.req_cpu_cores <= avail_cpu + 1e-9
            and job.req_mem_gib   <= avail_mem + 1e-9
        )


class ClusterLedger:
    """
    Cluster-wide event-sourced ledger.

    Tracks reserved resources per node. Answers point-in-time queries
    for any t. Produces ClusterSnapshot objects for the feature builder.
    """

    def __init__(self):
        self._nodes: Dict[str, NodeLedger] = {}
        self._job_to_node: Dict[str, str] = {}
        self._job_records: Dict[str, JobRecord] = {}
        self._oversubscriptions: List[dict] = []

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def register_node(self, node: NodeRecord) -> None:
        cap = NodeCapacity(
            node_id=node.node_id,
            gpu_frac=float(node.gpu_count),
            cpu_cores=node.cpu_cores,
            mem_gib=node.mem_gib,
        )
        self._nodes[node.node_id] = NodeLedger(cap)

    # ------------------------------------------------------------------
    # Event recording
    # ------------------------------------------------------------------

    def record_job_start(self, job: JobRecord) -> None:
        node_id = job.node_id
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' not registered in ledger")
        self._nodes[node_id].record_start(job)
        self._job_to_node[job.job_id] = node_id
        self._job_records[job.job_id] = job

        # Check invariant
        ledger = self._nodes[node_id]
        used = ledger.used_at(job.start_ts or job.submit_ts)
        cap = ledger.cap
        for used_v, cap_v, name in [
            (used[0], cap.gpu_frac,  "gpu_frac"),
            (used[1], cap.cpu_cores, "cpu_cores"),
            (used[2], cap.mem_gib,   "mem_gib"),
        ]:
            if used_v > cap_v + 1e-6:
                self._oversubscriptions.append({
                    "ts": job.start_ts,
                    "node_id": node_id,
                    "job_id": job.job_id,
                    "resource": name,
                    "used": used_v,
                    "capacity": cap_v,
                })

    def record_job_end(self, job_id: str, end_ts: float) -> None:
        node_id = self._job_to_node.get(job_id)
        if node_id is None:
            return
        job = self._job_records[job_id]
        self._nodes[node_id].record_end(job, end_ts)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def cluster_snapshot(
        self,
        as_of: float,
        data_origin: DataOrigin = DataOrigin.SYNTHETIC_SIM,
    ) -> List[ClusterSnapshot]:
        snaps = []
        for node_id, ledger in self._nodes.items():
            avail_gpu, avail_cpu, avail_mem = ledger.available_at(as_of)
            running = ledger.running_count_at(as_of)
            snaps.append(ClusterSnapshot(
                ts=as_of,
                node_id=node_id,
                avail_gpu_frac=avail_gpu,
                avail_cpu_cores=avail_cpu,
                avail_mem_gib=avail_mem,
                running_jobs=running,
            ))
        return snaps

    def node_available_at(self, node_id: str, as_of: float) -> Tuple[float, float, float]:
        if node_id not in self._nodes:
            return 0.0, 0.0, 0.0
        return self._nodes[node_id].available_at(as_of)

    def can_fit(self, job: JobRecord, node_id: str, at: float) -> bool:
        if node_id not in self._nodes:
            return False
        return self._nodes[node_id].check_fits(job, at)

    def feasible_nodes(self, job: JobRecord, at: float) -> List[str]:
        """Return node IDs where job fits at time at."""
        return [
            nid for nid in self._nodes
            if self.can_fit(job, nid, at)
        ]

    # ------------------------------------------------------------------
    # Invariant inspection
    # ------------------------------------------------------------------

    def oversubscription_events(self) -> List[dict]:
        return list(self._oversubscriptions)

    def node_ids(self) -> List[str]:
        return list(self._nodes.keys())
