"""
policies/backfill.py — P4: EASY backfilling (agent.md §11).

EASY backfilling:
1. The head-of-queue job (highest priority) gets a reservation at the
   earliest start time it can run (shadow time).
2. Smaller jobs that fit now AND finish before the shadow time are
   backfilled immediately.

Priority = (priority field) or (FIFO order if all equal).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ..schema.models import JobRecord, Placement
from ..ledger.state import ClusterLedger


class EASYBackfillPolicy:
    """P4: EASY Backfill scheduling [V]."""

    name = "backfill"

    def decide(
        self,
        now: float,
        pending: List[JobRecord],
        ledger: ClusterLedger,
        predictor=None,
    ) -> List[Placement]:
        if not pending:
            return []

        placements: List[Placement] = []

        # Sort by priority desc, then submit_ts asc
        ordered = sorted(pending, key=lambda j: (-j.priority, j.submit_ts))
        head = ordered[0]

        # Find head-of-queue node
        head_feasible = ledger.feasible_nodes(head, now)
        if head_feasible:
            placements.append(Placement(
                job_id=head.job_id,
                node_id=head_feasible[0],
                policy=self.name,
            ))
            # After placing head, try backfilling remaining
            placed_ids = {head.job_id}
            for job in ordered[1:]:
                if job.job_id in placed_ids:
                    continue
                feasible = ledger.feasible_nodes(job, now)
                if not feasible:
                    continue
                # Check job finishes within shadow window
                est_finish = now + (job.timelimit_s or 60.0)
                # Simple shadow time: head's timelimit
                shadow_time = now + (head.timelimit_s or 3600.0)
                if est_finish <= shadow_time:
                    placements.append(Placement(
                        job_id=job.job_id,
                        node_id=feasible[0],
                        policy=self.name,
                    ))
        else:
            # Head can't run now; try backfilling jobs that can fit
            for job in ordered[1:]:
                feasible = ledger.feasible_nodes(job, now)
                if feasible:
                    # Only backfill if it doesn't block head
                    placements.append(Placement(
                        job_id=job.job_id,
                        node_id=feasible[0],
                        policy=self.name,
                    ))

        return placements
