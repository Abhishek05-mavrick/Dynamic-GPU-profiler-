"""
policies/base.py — Policy protocol and base classes (agent.md §11).

A Policy maps (pending jobs, cluster view, predictor) to a list of Placements.
Decisions are event-triggered (arrival, finish) plus a periodic tick.
Optimizers use a rolling horizon and a time box.
"""

from __future__ import annotations

from typing import List, Optional, Protocol

from ..schema.models import JobRecord, Placement
from ..ledger.state import ClusterLedger


class Predictor(Protocol):
    """Teammate's runtime / demand predictor plug-in interface."""
    def predict(self, job: JobRecord) -> dict:
        """Return {"pred_runtime_s": float, "pred_gpu_util_pct": float, "uncertainty": float}."""
        ...


class Policy(Protocol):
    """Scheduler policy protocol — same interface for every policy."""
    name: str

    def decide(
        self,
        now: float,
        pending: List[JobRecord],
        ledger: ClusterLedger,
        predictor: Optional[Predictor] = None,
    ) -> List[Placement]:
        """Map pending jobs and cluster state → a list of placements."""
        ...
