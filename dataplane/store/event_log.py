"""
store/event_log.py — Append-only canonical event log.

In-memory implementation backed by Python lists.  Emits identical canonical
tables regardless of whether data come from the simulator, Slurm adapter, or
Alibaba replay (interface parity — rule 4 from agent.md §0).

In a production deployment this would persist to Parquet + DuckDB, but for the
modular demo we keep it pure-Python so there are no external dependencies.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional

from ..schema.models import (
    CanonicalEvent, ClusterSnapshot, JobRecord, NodeRecord, TelemetryRecord,
    DataOrigin, EventType,
)


class EventLog:
    """
    Append-only log of canonical events and associated row tables.

    Supports:
    - Appending jobs, nodes, telemetry, snapshots
    - Point-in-time queries (as_of)
    - Replay: iterate events in chronological order
    - JSON serialisation for persistence
    """

    def __init__(self, data_origin: DataOrigin = DataOrigin.SYNTHETIC_SIM):
        self.data_origin = data_origin
        self._events: List[CanonicalEvent] = []
        self._jobs: Dict[str, JobRecord] = {}          # job_id → latest record
        self._nodes: Dict[str, List[NodeRecord]] = {}   # node_id → SCD history
        self._telemetry: List[TelemetryRecord] = []
        self._snapshots: List[ClusterSnapshot] = []
        self._seq: int = 0

    # ------------------------------------------------------------------
    # Append helpers
    # ------------------------------------------------------------------

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def append_event(self, event: CanonicalEvent) -> None:
        event.sequence = self._next_seq()
        self._events.append(event)

    def upsert_job(self, job: JobRecord) -> None:
        """Insert or update a job record (last-write-wins)."""
        self._jobs[job.job_id] = job

    def append_node(self, node: NodeRecord) -> None:
        """Append a node SCD entry."""
        if node.node_id not in self._nodes:
            self._nodes[node.node_id] = []
        self._nodes[node.node_id].append(node)

    def append_telemetry(self, row: TelemetryRecord) -> None:
        self._telemetry.append(row)

    def append_snapshot(self, snap: ClusterSnapshot) -> None:
        self._snapshots.append(snap)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def jobs_at(self, as_of: float) -> List[JobRecord]:
        """All jobs whose submit_ts <= as_of."""
        return [j for j in self._jobs.values() if j.submit_ts <= as_of]

    def running_jobs_at(self, as_of: float, node_id: Optional[str] = None) -> List[JobRecord]:
        """Jobs that are running at as_of, optionally filtered by node."""
        result = []
        for j in self._jobs.values():
            if j.start_ts is None or j.start_ts > as_of:
                continue
            if j.end_ts is not None and j.end_ts <= as_of:
                continue
            if node_id and j.node_id != node_id:
                continue
            result.append(j)
        return result

    def pending_jobs_at(self, as_of: float) -> List[JobRecord]:
        """Jobs submitted but not yet started at as_of."""
        return [
            j for j in self._jobs.values()
            if j.submit_ts <= as_of and (j.start_ts is None or j.start_ts > as_of)
        ]

    def node_at(self, node_id: str, as_of: float) -> Optional[NodeRecord]:
        """Most recent NodeRecord valid at as_of."""
        history = self._nodes.get(node_id, [])
        result: Optional[NodeRecord] = None
        for record in history:
            if record.valid_from <= as_of:
                if result is None or record.valid_from > result.valid_from:
                    result = record
        return result

    def telemetry_at(
        self,
        node_id: str,
        metric: str,
        as_of: float,
        window_s: float = 60.0,
    ) -> List[TelemetryRecord]:
        """Telemetry rows for a node/metric in [as_of-window, as_of]."""
        cutoff = as_of - window_s
        return [
            t for t in self._telemetry
            if t.node_id == node_id
            and t.metric == metric
            and cutoff <= t.ts <= as_of
        ]

    def replay(self) -> List[CanonicalEvent]:
        """Return events in deterministic chronological order."""
        return sorted(self._events)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "data_origin": self.data_origin.value,
            "jobs": [self._job_to_dict(j) for j in self._jobs.values()],
            "events": [
                {
                    "type": e.event_type.value,
                    "ts": e.ts,
                    "job_id": e.job_id,
                    "node_id": e.node_id,
                    "seq": e.sequence,
                }
                for e in self._events
            ],
            "telemetry_count": len(self._telemetry),
            "snapshot_count": len(self._snapshots),
        }

    @staticmethod
    def _job_to_dict(j: JobRecord) -> dict:
        return {
            "job_id": j.job_id,
            "submit_ts": j.submit_ts,
            "start_ts": j.start_ts,
            "end_ts": j.end_ts,
            "status": j.status.value,
            "node_id": j.node_id,
            "req_gpu_frac": j.req_gpu_frac,
            "req_cpu_cores": j.req_cpu_cores,
            "req_mem_gib": j.req_mem_gib,
            "tenant_id": j.tenant_id,
            "job_class": j.job_class.value,
            "data_origin": j.data_origin.value,
        }

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "n_events": len(self._events),
            "n_jobs": len(self._jobs),
            "n_nodes": len(self._nodes),
            "n_telemetry": len(self._telemetry),
            "n_snapshots": len(self._snapshots),
        }
