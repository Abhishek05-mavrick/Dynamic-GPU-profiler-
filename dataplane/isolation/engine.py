"""
isolation/engine.py — Hard isolation engine (agent.md §10).

Ports the logic of isolation_sim.py into a composable module.

Rules:
  - MIG / exclusive mode: hardware-isolated. Cross-tenant co-location is
    structurally impossible.  The check always passes.
  - time_slice / shard / MPS: NOT isolated.  Two different tenants on the
    same device → BLOCKED.

Lifecycle:
  1. Policy proposes a Placement.
  2. IsolationEngine.check(placement, cluster_view) → (allowed: bool, reason: str)
  3. Violations are blocked, logged, and counted.
  4. Policy receives "rejected" signal and must propose again.

Soft-penalty form: penalty() returns a float in [0, 1] so QUBO/ILP policies
can include isolation as H_isolation in the cost function while the hard check
remains as the final guard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..schema.models import JobRecord, NodeRecord, Placement, SharingMode


@dataclass
class IsolationViolation:
    placement: Placement
    reason: str
    incumbent_tenant: str
    challenger_tenant: str


class IsolationEngine:
    """
    Hard isolation checker.

    tenant_key: which field identifies a tenant ("account" → tenant_id field).
    Configurable per agent.md §10 [A].
    """

    ISOLATED_MODES = {SharingMode.MIG, SharingMode.EXCLUSIVE}
    NON_ISOLATED_MODES = {SharingMode.TIME_SLICE, SharingMode.SHARD, SharingMode.MPS}

    def __init__(self, tenant_key: str = "account"):
        self.tenant_key = tenant_key
        self._violations: List[IsolationViolation] = []
        self._violation_count: int = 0

    # ------------------------------------------------------------------
    # Hard check
    # ------------------------------------------------------------------

    def check(
        self,
        placement: Placement,
        job: JobRecord,
        node: NodeRecord,
        running_on_node: List[JobRecord],
    ) -> Tuple[bool, str]:
        """
        Return (allowed, reason).

        allowed=True  → placement may proceed
        allowed=False → placement BLOCKED; reason explains why
        """
        mode = node.sharing_mode

        if mode in self.ISOLATED_MODES:
            # Hardware-isolated: cross-tenant is structurally impossible per slot
            return True, "isolated_mode"

        if mode in self.NON_ISOLATED_MODES:
            challenger_tenant = job.tenant_id
            for resident in running_on_node:
                incumbent_tenant = resident.tenant_id
                if incumbent_tenant and challenger_tenant and incumbent_tenant != challenger_tenant:
                    reason = (
                        f"Mode={mode.value} is non-isolated; "
                        f"tenant '{challenger_tenant[:8]}…' would share device with "
                        f"tenant '{incumbent_tenant[:8]}…'"
                    )
                    violation = IsolationViolation(
                        placement=placement,
                        reason=reason,
                        incumbent_tenant=incumbent_tenant,
                        challenger_tenant=challenger_tenant,
                    )
                    self._violations.append(violation)
                    self._violation_count += 1
                    return False, reason

        return True, "ok"

    # ------------------------------------------------------------------
    # Soft penalty (for QUBO / ILP cost function)
    # ------------------------------------------------------------------

    def penalty(
        self,
        job: JobRecord,
        node: NodeRecord,
        running_on_node: List[JobRecord],
    ) -> float:
        """
        Return isolation penalty in [0, 1].
        0.0 = safe, 1.0 = certain violation.

        Used by policies that include H_isolation in their cost function.
        """
        mode = node.sharing_mode
        if mode in self.ISOLATED_MODES:
            return 0.0
        challenger = job.tenant_id
        for resident in running_on_node:
            if resident.tenant_id and challenger and resident.tenant_id != challenger:
                return 1.0
        return 0.0

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @property
    def violation_count(self) -> int:
        return self._violation_count

    def violations(self) -> List[IsolationViolation]:
        return list(self._violations)

    def summary(self) -> str:
        return (
            f"IsolationEngine: {self._violation_count} violation(s) blocked\n"
            + "\n".join(f"  [{i+1}] {v.reason}" for i, v in enumerate(self._violations[-5:]))
        )
