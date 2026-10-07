"""Budget-limited phase refinement using observed CSI only."""
from __future__ import annotations

import time
import numpy as np

from .channel import sample_decision_scenarios
from .cuda_backend import evaluate_batch


class ScenarioScorer:
    def __init__(self, g_hat, h_hat, scenario_g, scenario_h, power, noise, backend):
        self.g_hat, self.h_hat = g_hat, h_hat
        self.gs, self.hs = scenario_g, scenario_h
        self.power, self.noise, self.backend = power, noise, backend
        self.candidates = self.evaluations = 0
        self.metadata = None

    def rates(self, phases):
        phases = np.atleast_2d(phases)
        count, scenarios = len(phases), len(self.gs)
        flat = np.repeat(phases, scenarios, axis=0)
        gs = np.tile(self.gs, (count, 1, 1))
        hs = np.tile(self.hs, (count, 1, 1))
        _, rates, self.metadata = evaluate_batch(
            hs, gs, flat, self.power, self.noise, self.backend,
            np.broadcast_to(self.h_hat, hs.shape), np.broadcast_to(self.g_hat, gs.shape))
        if self.backend == "cupy":
            rates = rates.get()
        self.candidates += count
        self.evaluations += count * scenarios
        return np.asarray(rates).reshape(count, scenarios, self.h_hat.shape[0])


def optimize_phases(g_hat, h_hat, cfg, rng, method="risk_constrained",
                    initial=None, scenario_count=4, budget=None, target=0.5,
                    risk_scope="pooled"):
    """Quantized coordinate refinement with counted candidate-scenario calls.

    Feasible risk candidates maximize expected sum rate. If none are feasible,
    candidates maximize the selected lower-tail rate. ``pooled`` applies the
    quantile to all scenario-user values; ``per_user`` takes the minimum of the
    user-wise quantiles and therefore cannot hide a weak user. All risk
    decisions use a fixed scenario pool. No truth-channel object crosses this
    interface.
    """
    methods = ("coordinate", "scenario_mean", "uncertainty_aware", "risk_constrained")
    started = time.perf_counter()
    if method not in methods:
        raise ValueError(f"unknown controller: {method}")
    if risk_scope not in ("pooled", "per_user"):
        raise ValueError("risk_scope must be 'pooled' or 'per_user'")
    budget = cfg.evaluation.matched_candidate_budget if budget is None else budget
    scenarios = 1 if method == "coordinate" else scenario_count
    if scenarios < 1 or budget < scenarios:
        raise ValueError("budget must cover the initial candidate and scenarios")
    order = rng.permutation(g_hat.shape[0])
    if method == "coordinate":
        gs, hs = g_hat[None], h_hat[None]
    else:
        gs, hs = sample_decision_scenarios(g_hat, h_hat, cfg.system.csi_error_scale,
                                          scenarios, rng)
    grid = np.linspace(-np.pi, np.pi, 2 ** cfg.system.phase_bits, endpoint=False)
    phases = np.zeros(g_hat.shape[0]) if initial is None else np.array(initial, copy=True)
    if phases.shape != (g_hat.shape[0],):
        raise ValueError("initial phase vector has wrong shape")
    # Round warm starts to the same hardware phase alphabet for all methods.
    distance = np.abs(np.angle(np.exp(1j * (phases[:, None] - grid[None]))))
    phases = grid[np.argmin(distance, axis=1)]
    scorer = ScenarioScorer(g_hat, h_hat, gs, hs, cfg.system.transmit_power,
                            cfg.system.noise_power, cfg.evaluation.backend)
    quantile = cfg.evaluation.outage_quantile

    def statistics(rates):
        user_tail = np.quantile(rates, quantile, axis=0)
        if risk_scope == "per_user":
            tail = float(np.min(user_tail))
        else:
            tail = float(np.quantile(rates, quantile))
        return float(rates.sum(axis=-1).mean()), tail, user_tail

    def rank(stats):
        mean, tail, _ = stats
        if method == "risk_constrained":
            return (int(tail >= target), mean if tail >= target else tail,
                    tail if tail >= target else mean)
        if method == "uncertainty_aware":
            return (0.65 * mean + 0.35 * tail,)
        return (mean,)

    best = statistics(scorer.rates(phases)[0])
    best_rank = rank(best)
    accepted = 0
    visits = 0
    while scorer.evaluations + scenarios <= budget:
        idx = int(order[visits % len(order)])
        visits += 1
        alternatives = grid[np.abs(np.angle(np.exp(1j * (grid - phases[idx])))) > 1e-10]
        remaining = (budget - scorer.evaluations) // scenarios
        alternatives = alternatives[:remaining]
        if len(alternatives) == 0:
            break
        trials = np.repeat(phases[None], len(alternatives), axis=0)
        trials[:, idx] = alternatives
        rates = scorer.rates(trials)
        for trial, candidate_rates in zip(trials, rates):
            stats = statistics(candidate_rates)
            if rank(stats) > best_rank:
                phases, best, best_rank = trial.copy(), stats, rank(stats)
                accepted += 1
    return phases, {
        "candidate_count": scorer.candidates,
        "rate_evaluations": scorer.evaluations,
        "budget_limit": budget,
        "scenario_count": scenarios,
        "unused_budget": budget - scorer.evaluations,
        "accepted_moves": accepted,
        "selected_p05": best[1],
        "selected_user_p05": best[2].tolist(),
        "risk_scope": risk_scope,
        "reliability_target": float(target),
        "selected_expected_sum_rate": best[0],
        "constraint_satisfied": bool(best[1] >= target) if method == "risk_constrained" else None,
        "fallback": bool(best[1] < target) if method == "risk_constrained" else None,
        "decision_seconds": time.perf_counter() - started,
        "backend_metadata": scorer.metadata,
    }
