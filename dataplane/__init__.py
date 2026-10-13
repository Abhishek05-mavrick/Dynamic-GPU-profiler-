"""
GSDP — GPU Scheduling Data Plane
=================================
Modular data plane for the quantum-inspired, interference-aware GPU scheduler.

Modules:
    schema      — Pydantic models, canonical data types, units
    store       — Event log, DuckDB + Parquet I/O
    sources     — Sim / Slurm / Alibaba event sources
    ledger      — Cluster-state ledger (reserved vs measured)
    features    — Point-in-time feature builder
    isolation   — Hard isolation engine
    policies    — Scheduling policies (FIFO, SJF, backfill, QUBO…)
    predictors  — Runtime / demand predictor plug-ins
    eval        — Metrics, evaluation harness, reports
"""

__version__ = "0.1.0"
