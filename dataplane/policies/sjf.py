"""
policies/sjf.py — P2: Shortest-Job-First / SRPT using noisy estimates (agent.md §11).

Uses timelimit_s as the (noisy) runtime estimate.  Falls back to FIFO order
when no estimate is available.  A NullPredictor is used when no predictor
is configured.
"""

from __future__ import annotations

from typing import List, Optional

from ..schema.models import JobRecord, Placement
from ..ledger.state import ClusterLedger


class SJFPolicy:
    """P2: Shortest-Job-First using noisy time-limit estimates [V]."""

    name = "sjf"

    def decide(
        self,
        now: float,
        pending: List[JobRecord],
        ledger: ClusterLedger,
        predictor=None,
    ) -> List[Placement]:
        placements: List[Placement] = []

        def estimate(job: JobRecord) -> float:
            if predictor is not None:
                pred = predictor.predict(job)
                return pred.get("pred_runtime_s", job.timelimit_s or 1e9)
            return job.timelimit_s if job.timelimit_s else 1e9

        ordered = sorted(pending, key=estimate)
        for job in ordered:
            feasible = ledger.feasible_nodes(job, now)
            if feasible:
                placements.append(Placement(
                    job_id=job.job_id,
                    node_id=feasible[0],
                    policy=self.name,
                    score=estimate(job),
                ))
        return placements
