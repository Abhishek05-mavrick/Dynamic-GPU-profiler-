"""
policies/interference_greedy.py — P6: Interference-aware greedy policy (agent.md §11).

Uses existing profiler fingerprints (from dynamic_profiler) + the contention model
from agent.md §8.8 to score node candidates by predicted slowdown.

Score = avail_score - interference_penalty - isolation_penalty

Lower total interference → higher score → preferred placement.
"""

from __future__ import annotations

import sys
import os
from typing import Dict, List, Optional

from ..schema.models import JobRecord, NodeRecord, Placement
from ..ledger.state import ClusterLedger
from ..isolation.engine import IsolationEngine


class InterferenceGreedyPolicy:
    """
    P6: Greedy interference-aware scheduling.

    Selects the node that minimises predicted co-location slowdown
    for the arriving job, while respecting hard isolation constraints.
    """

    name = "interference_greedy"

    def __init__(
        self,
        isolation: Optional[IsolationEngine] = None,
        interference_weight: float = 0.5,
        isolation_weight: float = 1.0,
    ):
        self._iso = isolation or IsolationEngine()
        self._interference_weight = interference_weight
        self._isolation_weight = isolation_weight

    def decide(
        self,
        now: float,
        pending: List[JobRecord],
        ledger: ClusterLedger,
        log=None,
        predictor=None,
    ) -> List[Placement]:
        placements: List[Placement] = []
        ordered = sorted(pending, key=lambda j: j.submit_ts)

        for job in ordered:
            best_node, best_score = self._score_nodes(job, now, ledger, log)
            if best_node:
                placements.append(Placement(
                    job_id=job.job_id,
                    node_id=best_node,
                    policy=self.name,
                    score=best_score,
                ))
        return placements

    def _score_nodes(
        self,
        job: JobRecord,
        now: float,
        ledger: ClusterLedger,
        log=None,
    ):
        best_node = None
        best_score = -1e9

        for node_id in ledger.feasible_nodes(job, now):
            avail_gpu, avail_cpu, avail_mem = ledger.node_available_at(node_id, now)

            # Availability score (normalised)
            avail_score = min(avail_gpu, 1.0) * 0.5 + min(avail_cpu / 64, 1.0) * 0.25

            # Interference penalty: proxy — fraction of GPU already used
            # In production this uses actual fingerprint vectors [A]
            gpu_capacity = avail_gpu + job.req_gpu_frac
            used_frac = 1.0 - (avail_gpu / gpu_capacity) if gpu_capacity > 0 else 0.0
            interference_penalty = used_frac * self._interference_weight

            # Isolation penalty
            running = log.running_jobs_at(now, node_id=node_id) if log else []
            # Build a minimal NodeRecord for the isolation check
            from ..schema.models import NodeRecord as NR, SharingMode
            node_stub = NR(
                node_id=node_id,
                sharing_mode=SharingMode.EXCLUSIVE,  # conservative default
            )
            iso_penalty = self._iso.penalty(job, node_stub, running) * self._isolation_weight

            score = avail_score - interference_penalty - iso_penalty

            if score > best_score:
                best_score = score
                best_node = node_id

        return best_node, best_score
