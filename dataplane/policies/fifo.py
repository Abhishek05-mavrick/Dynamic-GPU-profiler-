"""
policies/fifo.py — P1: FIFO policy (agent.md §11).

Simplest baseline.  Place pending jobs in order of arrival (submit_ts).
First-feasible node assignment.
"""

from __future__ import annotations

from typing import List, Optional

from ..schema.models import JobRecord, Placement
from ..ledger.state import ClusterLedger


class FIFOPolicy:
    """P1: First-In-First-Out scheduling — survey baseline [V]."""

    name = "fifo"

    def decide(
        self,
        now: float,
        pending: List[JobRecord],
        ledger: ClusterLedger,
        predictor=None,
    ) -> List[Placement]:
        placements: List[Placement] = []
        ordered = sorted(pending, key=lambda j: j.submit_ts)
        for job in ordered:
            feasible = ledger.feasible_nodes(job, now)
            if feasible:
                placements.append(Placement(
                    job_id=job.job_id,
                    node_id=feasible[0],
                    policy=self.name,
                ))
        return placements
