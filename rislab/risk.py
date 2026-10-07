"""Exact fixed-sample risk bounds and admission-cost diagnostics."""
from __future__ import annotations

import numpy as np
from scipy.stats import beta, binom


def outage_upper_bound(failures, samples: int, delta: float):
    """One-sided Clopper-Pearson upper limit; confidence is 1-delta."""
    if samples < 1 or int(samples) != samples or not 0 < delta < 1:
        raise ValueError("positive integer samples and delta in (0,1) required")
    counts = np.asarray(failures)
    if np.any(~np.isfinite(counts)) or np.any(counts != np.floor(counts)) or np.any(counts < 0) or np.any(counts > samples):
        raise ValueError("failures must be integer counts in [0,samples]")
    upper = np.ones(counts.shape, dtype=float)
    mask = counts < samples
    upper[mask] = beta.ppf(1 - delta, counts[mask] + 1, samples - counts[mask])
    return upper.item() if upper.ndim == 0 else upper


def maximum_accepted_failures(samples: int, epsilon: float, delta: float) -> int:
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must be in (0,1)")
    upper = outage_upper_bound(np.arange(samples + 1), samples, delta)
    accepted = np.flatnonzero(upper <= epsilon)
    return int(accepted[-1]) if len(accepted) else -1


def single_weak_user_diagnostic(p: float, samples: int, users: int, epsilon: float, delta: float):
    """One risky user; others have probability zero and always pass if feasible.

    This is a hypothetical fixed-p binomial calculation. It is not evidence
    of a channel posterior, dynamic risk controller, or physical reliability.
    """
    if not np.isfinite(p) or not 0 <= p <= 1 or users < 1 or int(users) != users:
        raise ValueError("p in [0,1] and positive integer users required")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0,1)")
    limit = maximum_accepted_failures(samples, epsilon, delta / users)
    admission = float(binom.cdf(limit, samples, p)) if limit >= 0 else 0.0
    return {
        "hypothetical_user_outage": p, "validation_samples": samples, "users": users,
        "epsilon": epsilon, "family_delta": delta, "per_user_delta": delta / users,
        "max_accepted_failures": limit,
        "zero_failure_upper_bound": outage_upper_bound(0, samples, delta / users),
        "admission_probability": admission, "rejection_probability": 1 - admission,
        "all_slot_service_outage": 1 - admission * (1 - p),
        "all_slot_outage_increase": (1 - admission) * (1 - p),
        "service_outage_target_met": bool(1 - admission * (1 - p) <= epsilon),
    }
