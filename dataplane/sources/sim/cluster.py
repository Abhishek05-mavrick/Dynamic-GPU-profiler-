"""
sources/sim/cluster.py — Cluster generator for the simulator (agent.md §8.4).

Config-driven node types.  Presets: small (4 nodes), hetero (mixed GPU
generations), mig_heavy.  MIG profile tables from NVIDIA docs [M] — verify
before relying on specific combinations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from ...schema.models import DataOrigin, NodeRecord, SharingMode


# ---------------------------------------------------------------------------
# Node-type presets
# ---------------------------------------------------------------------------

@dataclass
class NodeTypeConfig:
    gpu_model: str
    gpu_count: int
    cpu_cores: float
    mem_gib: float
    mig_enabled: bool = False
    mig_layout: List[List[str]] = field(default_factory=list)
    sharing_mode: SharingMode = SharingMode.EXCLUSIVE


# MIG profile examples for A100 40 GB [M] — verify against NVIDIA MIG user guide
A100_MIG_PROFILES = [
    ["1g.5gb", "1g.5gb", "2g.10gb", "3g.20gb"],  # one GPU's worth of instances
]

# H100 80 GB MIG profiles [M]
H100_MIG_PROFILES = [
    ["1g.10gb", "1g.10gb", "2g.20gb", "4g.40gb"],
]

PRESETS: dict[str, List[NodeTypeConfig]] = {
    "small": [
        NodeTypeConfig("RTX-2050", 1, 8.0,  4.0),
        NodeTypeConfig("RTX-2050", 1, 8.0,  4.0),
        NodeTypeConfig("RTX-2050", 1, 8.0,  4.0),
        NodeTypeConfig("RTX-2050", 1, 8.0,  4.0),
    ],
    "hetero": [
        NodeTypeConfig("RTX-3090",  2, 32.0, 64.0),
        NodeTypeConfig("A100-40GB", 4, 64.0, 256.0, mig_enabled=True,
                       mig_layout=A100_MIG_PROFILES * 4,
                       sharing_mode=SharingMode.MIG),
        NodeTypeConfig("H100-80GB", 8, 96.0, 512.0, mig_enabled=True,
                       mig_layout=H100_MIG_PROFILES * 8,
                       sharing_mode=SharingMode.MIG),
    ],
    "mig_heavy": [
        NodeTypeConfig(f"A100-40GB", 4, 64.0, 256.0, mig_enabled=True,
                       mig_layout=A100_MIG_PROFILES * 4,
                       sharing_mode=SharingMode.MIG)
        for _ in range(6)
    ],
}


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class ClusterGenerator:
    """Build a list of NodeRecord objects from a preset or custom config."""

    @staticmethod
    def build(preset: str = "small") -> List[NodeRecord]:
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(PRESETS)}")
        node_types = PRESETS[preset]
        nodes = []
        for i, cfg in enumerate(node_types):
            node = NodeRecord(
                node_id=f"node_{i:03d}",
                gpu_model=cfg.gpu_model,
                gpu_count=cfg.gpu_count,
                cpu_cores=cfg.cpu_cores,
                mem_gib=cfg.mem_gib,
                mig_enabled=cfg.mig_enabled,
                mig_layout=cfg.mig_layout,
                sharing_mode=cfg.sharing_mode,
                valid_from=0.0,
                data_origin=DataOrigin.SYNTHETIC_SIM,
            )
            nodes.append(node)
        return nodes
