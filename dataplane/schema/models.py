"""
schema/models.py — Canonical pydantic data models and enums for GSDP.

All timestamps are UTC epoch seconds (float).
All memory in GiB. All CPU in cores (float). GPU fraction in [0, 1] per full GPU.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    PENDING   = "PENDING"
    RUNNING   = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED    = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT   = "TIMEOUT"
    NODE_FAIL = "NODE_FAIL"
    OOM       = "OOM"
    OTHER     = "OTHER"


class SharingMode(str, Enum):
    EXCLUSIVE  = "exclusive"
    MIG        = "mig"
    TIME_SLICE = "time_slice"
    SHARD      = "shard"
    MPS        = "mps"


class DataOrigin(str, Enum):
    SLURM_LIVE    = "slurm_live"
    ALIBABA_REPLAY = "alibaba_replay"
    SYNTHETIC_SIM  = "synthetic_sim"


class MetricQuality(str, Enum):
    MEASURED    = "measured"
    DERIVED     = "derived"
    ESTIMATED   = "estimated"
    SIMULATED   = "simulated"
    UNAVAILABLE = "unavailable"


class EventType(str, Enum):
    ARRIVE        = "ARRIVE"
    START         = "START"
    FINISH        = "FINISH"
    TICK          = "TICK"
    RECONFIG_BEGIN = "RECONFIG_BEGIN"
    RECONFIG_END  = "RECONFIG_END"
    NODE_FAIL     = "NODE_FAIL"


class JobClass(str, Enum):
    TRAINING    = "training"
    INFERENCE   = "inference"
    INTERACTIVE = "interactive"
    UNKNOWN     = "unknown"


class BoundType(str, Enum):
    COMPUTE  = "compute"
    MEMORY   = "memory"
    BALANCED = "balanced"
    IDLE     = "idle"
    UNKNOWN  = "unknown"


# ---------------------------------------------------------------------------
# Core data models (dataclasses — lightweight, no pydantic needed)
# ---------------------------------------------------------------------------

@dataclass
class JobRecord:
    """One allocation row — fields 1–10 plus metadata."""

    # Field 1: identity
    job_id: str
    instance_id: str = ""

    # Field 2-4: timestamps (UTC epoch seconds)
    submit_ts: float = 0.0
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None

    # Field 5
    status: JobStatus = JobStatus.PENDING

    # Field 6 — node placement (multi-node in job_nodes child table)
    node_id: str = ""

    # Field 7: GPU demand
    req_gpu_frac: float = 0.0      # 1.0 = one full GPU
    req_gpu_count: int = 0
    req_mig_profile: Optional[str] = None

    # Field 8
    req_cpu_cores: float = 0.0

    # Field 9
    req_mem_gib: float = 0.0

    # Field 10
    req_gpu_type: Optional[str] = None

    # Extra scheduler metadata
    tenant_id: str = ""            # hashed at ingestion
    partition: str = ""
    qos: str = ""
    priority: int = 0
    timelimit_s: Optional[float] = None
    job_class: JobClass = JobClass.UNKNOWN

    # Provenance
    data_origin: DataOrigin = DataOrigin.SYNTHETIC_SIM
    raw: str = ""


@dataclass
class NodeRecord:
    """One node snapshot row — fields 11–15 (slowly-changing)."""

    # Field 11
    node_id: str

    # Field 12-15
    gpu_model: str = ""
    gpu_count: int = 0
    cpu_cores: float = 0.0
    mem_gib: float = 0.0

    # MIG topology
    mig_enabled: bool = False
    mig_layout: List[List[str]] = field(default_factory=list)   # per GPU → list of profiles
    sharing_mode: SharingMode = SharingMode.EXCLUSIVE

    # Validity window for SCD
    valid_from: float = 0.0
    valid_to: Optional[float] = None

    data_origin: DataOrigin = DataOrigin.SYNTHETIC_SIM


@dataclass
class TelemetryRecord:
    """One telemetry row — fields 16–18 plus extras (long format)."""

    ts: float
    node_id: str
    device_id: str = ""
    mig_instance_id: Optional[str] = None
    job_id: Optional[str] = None
    metric: str = ""
    value: Optional[float] = None
    unit: str = ""
    quality: MetricQuality = MetricQuality.UNAVAILABLE
    source: str = ""


@dataclass
class ClusterSnapshot:
    """Fields 19–22 plus measured deltas — written at each decision epoch."""

    ts: float
    node_id: str

    # Reserved (from ledger)
    avail_gpu_frac: float = 0.0     # Field 19
    avail_cpu_cores: float = 0.0    # Field 20
    avail_mem_gib: float = 0.0      # Field 21
    running_jobs: int = 0           # Field 22

    # Measured free (capacity minus profiler-measured usage)
    measured_free_gpu_frac: Optional[float] = None
    measured_free_cpu_cores: Optional[float] = None
    measured_free_mem_gib: Optional[float] = None


@dataclass
class CanonicalEvent:
    """An event in the simulation or live event log."""

    event_type: EventType
    ts: float
    job_id: str = ""
    node_id: str = ""
    sequence: int = 0          # tie-break within same ts
    payload: Dict = field(default_factory=dict)

    # Priority: lower value = higher priority (for same ts)
    PRIORITY: Dict[EventType, int] = field(default_factory=lambda: {
        EventType.ARRIVE:         1,
        EventType.START:          2,
        EventType.FINISH:         3,
        EventType.TICK:           4,
        EventType.RECONFIG_BEGIN: 5,
        EventType.RECONFIG_END:   6,
        EventType.NODE_FAIL:      0,
    })

    def __lt__(self, other: "CanonicalEvent") -> bool:
        p_self  = self.PRIORITY.get(self.event_type, 9)
        p_other = other.PRIORITY.get(other.event_type, 9)
        return (self.ts, p_self, self.sequence) < (other.ts, p_other, other.sequence)


@dataclass
class Placement:
    """A scheduler decision: place job on node (and optionally device/profile)."""

    job_id: str
    node_id: str
    device_id: str = ""
    mig_profile: Optional[str] = None
    score: float = 0.0
    policy: str = ""


@dataclass
class Decision:
    """Full decision record with candidates and isolation check outcome."""

    decision_id: str
    ts: float
    policy: str
    job_id: str
    chosen_node: str = ""
    chosen_device: str = ""
    chosen_profile: Optional[str] = None
    candidates: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    isolation_checks: Dict[str, bool] = field(default_factory=dict)
    mode: str = "shadow"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

_SALT = "gsdp-tenant-salt-2026"   # override from config in production


def hash_tenant(raw_name: str) -> str:
    """Hash a tenant name with a stable salt. Never log raw names."""
    return hashlib.sha256(f"{_SALT}:{raw_name}".encode()).hexdigest()[:16]


def now_ts() -> float:
    return time.time()
