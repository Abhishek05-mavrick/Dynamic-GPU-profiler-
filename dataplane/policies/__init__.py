"""policies package — all scheduling policies."""
from .base import Policy, Predictor
from .fifo import FIFOPolicy
from .sjf import SJFPolicy
from .backfill import EASYBackfillPolicy
from .interference_greedy import InterferenceGreedyPolicy

POLICIES = {
    "fifo":                FIFOPolicy,
    "sjf":                 SJFPolicy,
    "backfill":            EASYBackfillPolicy,
    "interference_greedy": InterferenceGreedyPolicy,
}

__all__ = [
    "Policy", "Predictor",
    "FIFOPolicy", "SJFPolicy", "EASYBackfillPolicy", "InterferenceGreedyPolicy",
    "POLICIES",
]
