"""
predictors/null.py — NullPredictor (agent.md §9).

Returns the user's submitted time limit as the runtime estimate.
Everything runs without the teammate's model when this predictor is active.
Uncertainty is always 1.0 (maximum) to signal "not calibrated".
"""

from __future__ import annotations

from ..schema.models import JobRecord


class NullPredictor:
    """Fallback predictor: returns user estimate with maximum uncertainty."""

    name = "null"

    def predict(self, job: JobRecord) -> dict:
        return {
            "pred_runtime_s": job.timelimit_s if job.timelimit_s else 60.0,
            "pred_gpu_util_pct": 50.0,   # conservative unknown [A]
            "pred_gpu_mem_gib": job.req_mem_gib,
            "uncertainty": 1.0,           # max uncertainty
        }
