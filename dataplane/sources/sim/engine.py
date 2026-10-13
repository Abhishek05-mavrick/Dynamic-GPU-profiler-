"""
sources/sim/engine.py — Discrete-event simulation engine (agent.md §8.3).

Heap-based event queue.  Events ordered by (time, priority, sequence).
Progress model: each running job has `remaining_work`; its `rate` changes on
every co-location change, so finish times are recomputed dynamically.

RNG: one random.Random per component, seeded from the run seed for determinism.
Same seed + same config → byte-identical outputs (rule 9, agent.md §0).
"""

from __future__ import annotations

import heapq
import random
import time as _time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ...schema.models import (
    CanonicalEvent, ClusterSnapshot, DataOrigin, EventType,
    JobClass, JobRecord, JobStatus, NodeRecord, TelemetryRecord,
    MetricQuality, Placement, hash_tenant, now_ts,
)
from ...store.event_log import EventLog
from ...ledger.state import ClusterLedger


# ---------------------------------------------------------------------------
# Internal simulation state
# ---------------------------------------------------------------------------

@dataclass
class SimJob:
    """Simulation-internal job state (hidden true demand + phase)."""
    job_id: str
    node_id: str = ""
    remaining_work: float = 1.0   # normalised; 1.0 = full service
    rate: float = 1.0             # current progress rate (0 to 1 due to contention)
    demand_sm: float = 0.5        # hidden true SM demand (0-1)
    demand_dram: float = 0.3      # hidden true DRAM demand
    start_real_ts: float = 0.0    # simulation time when started
    record: Optional[JobRecord] = None


@dataclass(order=True)
class _HeapEntry:
    """Heap item: (ts, priority, seq, event)."""
    ts: float
    priority: int
    seq: int
    event: CanonicalEvent = field(compare=False)


_EVENT_PRIORITY = {
    EventType.NODE_FAIL:     0,
    EventType.ARRIVE:        1,
    EventType.START:         2,
    EventType.FINISH:        3,
    EventType.TICK:          4,
    EventType.RECONFIG_BEGIN: 5,
    EventType.RECONFIG_END:  6,
}


class SimEngine:
    """
    Discrete-event simulation engine.

    Usage:
        engine = SimEngine(seed=42)
        engine.load_cluster(node_configs)
        engine.load_workload(jobs)
        engine.run(until=3600.0)
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rng = random.Random(seed)
        self._heap: List[_HeapEntry] = []
        self._seq = 0
        self._now: float = 0.0
        self._running: Dict[str, SimJob] = {}    # job_id → SimJob
        self._log = EventLog(DataOrigin.SYNTHETIC_SIM)
        self._ledger = ClusterLedger()
        self._nodes: Dict[str, NodeRecord] = {}
        self._tick_interval: float = 30.0
        self._policy_fn: Optional[Callable] = None
        self._max_events: int = 10_000_000

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def load_cluster(self, nodes: List[NodeRecord]) -> None:
        for node in nodes:
            self._nodes[node.node_id] = node
            self._ledger.register_node(node)
            self._log.append_node(node)

    def set_policy(self, fn: Callable) -> None:
        """Set the scheduling policy function: fn(pending, ledger, now) → List[Placement]."""
        self._policy_fn = fn

    def load_workload(self, jobs: List[JobRecord]) -> None:
        """Schedule ARRIVE events for all jobs."""
        for job in jobs:
            self._push(EventType.ARRIVE, job.submit_ts, job_id=job.job_id, payload={"job": job})
            self._log.upsert_job(job)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, until: float = 3600.0, tick_interval: float = 30.0) -> EventLog:
        """
        Run the simulation until `until` seconds.

        Returns the populated EventLog.
        """
        self._tick_interval = tick_interval
        self._push(EventType.TICK, tick_interval)
        n = 0
        while self._heap and n < self._max_events:
            entry = heapq.heappop(self._heap)
            if entry.ts > until:
                break
            self._now = entry.ts
            self._dispatch(entry.event)
            n += 1
        self._finalize(until)
        return self._log

    def current_time(self) -> float:
        return self._now

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, ev: CanonicalEvent) -> None:
        self._log.append_event(ev)
        if ev.event_type == EventType.ARRIVE:
            self._on_arrive(ev)
        elif ev.event_type == EventType.START:
            self._on_start(ev)
        elif ev.event_type == EventType.FINISH:
            self._on_finish(ev)
        elif ev.event_type == EventType.TICK:
            self._on_tick(ev)

    def _on_arrive(self, ev: CanonicalEvent) -> None:
        job: JobRecord = ev.payload["job"]
        job.status = JobStatus.PENDING
        self._log.upsert_job(job)
        # Try to schedule immediately
        self._try_schedule()

    def _on_start(self, ev: CanonicalEvent) -> None:
        job_id = ev.job_id
        node_id = ev.node_id
        job = self._log._jobs.get(job_id)
        if job is None:
            return
        job.start_ts = ev.ts
        job.status = JobStatus.RUNNING
        job.node_id = node_id
        self._log.upsert_job(job)
        self._ledger.record_job_start(job)

        # True demand for interference model (randomised from seed)
        rng = random.Random(self.seed + hash(job_id) % 10_000)
        sim_job = SimJob(
            job_id=job_id,
            node_id=node_id,
            remaining_work=1.0,
            rate=1.0,
            demand_sm=rng.uniform(0.2, 0.9),
            demand_dram=rng.uniform(0.1, 0.7),
            start_real_ts=ev.ts,
            record=job,
        )
        self._running[job_id] = sim_job
        self._update_rates()

        # Schedule finish
        duration = (job.timelimit_s or 60.0)
        finish_ts = ev.ts + duration / sim_job.rate
        self._push(EventType.FINISH, finish_ts, job_id=job_id, node_id=node_id)

    def _on_finish(self, ev: CanonicalEvent) -> None:
        job_id = ev.job_id
        job = self._log._jobs.get(job_id)
        if job is None or job_id not in self._running:
            return
        job.end_ts = ev.ts
        job.status = JobStatus.COMPLETED
        self._log.upsert_job(job)
        self._ledger.record_job_end(job_id, ev.ts)
        del self._running[job_id]
        self._update_rates()
        # New resources free → try to schedule pending
        self._try_schedule()

    def _on_tick(self, ev: CanonicalEvent) -> None:
        # Emit cluster snapshots and synthetic telemetry
        snaps = self._ledger.cluster_snapshot(ev.ts, DataOrigin.SYNTHETIC_SIM)
        for snap in snaps:
            self._log.append_snapshot(snap)
            # Synthetic telemetry for running jobs on this node
            for sjob in self._running.values():
                if sjob.node_id == snap.node_id:
                    noise = self._rng.gauss(0, 0.02)
                    self._log.append_telemetry(TelemetryRecord(
                        ts=ev.ts,
                        node_id=snap.node_id,
                        device_id="gpu0",
                        job_id=sjob.job_id,
                        metric="gpu_util_pct",
                        value=max(0.0, min(100.0, sjob.demand_sm * 100 + noise * 10)),
                        unit="%",
                        quality=MetricQuality.SIMULATED,
                        source="sim",
                    ))
        # Next tick
        self._push(EventType.TICK, ev.ts + self._tick_interval)

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def _try_schedule(self) -> None:
        pending = self._log.pending_jobs_at(self._now)
        if not pending or not self._nodes:
            return

        if self._policy_fn is not None:
            placements = self._policy_fn(pending, self._ledger, self._now)
        else:
            placements = self._fifo_schedule(pending)

        for placement in placements:
            job = self._log._jobs.get(placement.job_id)
            if job is None:
                continue
            self._push(EventType.START, self._now, job_id=placement.job_id, node_id=placement.node_id)

    def _fifo_schedule(self, pending: List[JobRecord]) -> List[Placement]:
        """Default FIFO policy — place by submit order if a node fits."""
        placements = []
        # Sort by submit time (FIFO)
        ordered = sorted(pending, key=lambda j: j.submit_ts)
        for job in ordered:
            feasible = self._ledger.feasible_nodes(job, self._now)
            if feasible:
                node_id = feasible[0]
                placements.append(Placement(job_id=job.job_id, node_id=node_id, policy="fifo"))
        return placements

    # ------------------------------------------------------------------
    # Interference / rate model (agent.md §8.8)
    # ------------------------------------------------------------------

    def _update_rates(self) -> None:
        """
        Recompute progress rates for all running jobs based on co-location.
        rate_j = min(1, cap_r / S_r) over resources r used by j.
        Simple implementation: normalise by total SM/DRAM demand per node.
        """
        # Group by node
        by_node: Dict[str, List[SimJob]] = {}
        for sj in self._running.values():
            by_node.setdefault(sj.node_id, []).append(sj)

        for node_id, jobs_on_node in by_node.items():
            total_sm = sum(j.demand_sm for j in jobs_on_node)
            total_dram = sum(j.demand_dram for j in jobs_on_node)
            for j in jobs_on_node:
                rate_sm   = min(1.0, j.demand_sm / total_sm)  if total_sm > 0 else 1.0
                rate_dram = min(1.0, j.demand_dram / total_dram) if total_dram > 0 else 1.0
                j.rate = min(rate_sm, rate_dram)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _push(
        self,
        etype: EventType,
        ts: float,
        job_id: str = "",
        node_id: str = "",
        payload: dict = None,
    ) -> None:
        self._seq += 1
        ev = CanonicalEvent(
            event_type=etype,
            ts=ts,
            job_id=job_id,
            node_id=node_id,
            sequence=self._seq,
            payload=payload or {},
        )
        prio = _EVENT_PRIORITY.get(etype, 9)
        heapq.heappush(self._heap, _HeapEntry(ts=ts, priority=prio, seq=self._seq, event=ev))

    def _finalize(self, until: float) -> None:
        """Mark any still-running jobs as RUNNING at end of simulation."""
        for job_id, sj in self._running.items():
            job = self._log._jobs.get(job_id)
            if job:
                job.status = JobStatus.RUNNING
                self._log.upsert_job(job)
