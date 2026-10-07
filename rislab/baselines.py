from __future__ import annotations

import numpy as np

from .metrics import evaluate_phases


def random_phases(elements: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-np.pi, np.pi, size=elements)


def coordinate_search(channel, cfg, rng: np.random.Generator, iterations: int = 2) -> np.ndarray:
    """Low-cost deterministic baseline: coordinate phase refinement on estimated CSI."""
    phases = np.zeros(channel.g.shape[0])
    candidates = np.linspace(-np.pi, np.pi, 2 ** cfg.system.phase_bits, endpoint=False)
    best = evaluate_phases(channel, phases, cfg.system.transmit_power, cfg.system.noise_power,
                           use_estimate=True)["sum_rate"]
    for _ in range(iterations):
        for idx in range(len(phases)):
            local_best, local_phase = best, phases[idx]
            for candidate in candidates:
                trial = phases.copy(); trial[idx] = candidate
                score = evaluate_phases(channel, trial, cfg.system.transmit_power, cfg.system.noise_power,
                                        use_estimate=True)["sum_rate"]
                if score > local_best:
                    local_best, local_phase = score, candidate
            phases[idx], best = local_phase, local_best
    return phases


def robust_projected_search(channel, cfg, rng: np.random.Generator, samples: int = 12) -> np.ndarray:
    """Sample-and-select phases using one estimated-CSI observation."""
    elements = channel.g.shape[0]
    candidates = [random_phases(elements, rng) for _ in range(samples)]
    candidates.append(np.zeros(elements))
    scored = [(evaluate_phases(channel, p, cfg.system.transmit_power, cfg.system.noise_power,
                               use_estimate=True), p) for p in candidates]
    return max(scored, key=lambda x: x[0]["sum_rate"])[1]


def uncertainty_aware_phases(channel, cfg, rng: np.random.Generator,
                             scenarios: int = 8, candidates: int = 24) -> np.ndarray:
    """Choose phases by maximizing a lower-tail surrogate over CSI perturbations.

    This is deliberately a transparent reference controller. It provides a
    non-learning robust target for later learned-policy distillation and makes
    the uncertainty mechanism testable independently of neural architecture.
    """
    from .channel import ChannelRealization, sample_decision_scenarios
    gs, hs = sample_decision_scenarios(channel.g_hat, channel.h_hat,
                                     cfg.system.csi_error_scale, scenarios, rng)
    scenario_channels = [ChannelRealization(g, h, channel.g_hat, channel.h_hat)
                         for g, h in zip(gs, hs)]
    elements = channel.g.shape[0]
    phase_candidates = [random_phases(elements, rng) for _ in range(candidates)]
    phase_candidates.append(np.zeros(elements))
    scored = []
    for phases in phase_candidates:
        # Scenario estimates determine the decision statistic. The same
        # ground-truth channel is used for the returned phase evaluation.
        estimate_scores = [evaluate_phases(sc, phases, cfg.system.transmit_power,
                                            cfg.system.noise_power)
                           for sc in scenario_channels]
        estimate_lower = float(np.quantile([v["p05_user_rate"] for v in estimate_scores], 0.25))
        estimate_sum = float(np.mean([v["sum_rate"] for v in estimate_scores]))
        scored.append((0.65 * estimate_sum + 0.35 * estimate_lower, phases))
    return max(scored, key=lambda x: x[0])[1]


def risk_constrained_phases(channel, cfg, rng: np.random.Generator,
                             scenarios: int = 8, candidates: int = 24,
                             reliability_target: float = 0.5) -> tuple[np.ndarray, dict]:
    """Select a phase vector subject to an empirical scenario reliability target.

    The target is applied to the pooled user rates across sampled CSI scenarios;
    infeasible candidates fall back to the highest lower-tail score and are
    reported explicitly through the returned diagnostics.
    """
    from .channel import ChannelRealization, sample_decision_scenarios
    gs, hs = sample_decision_scenarios(channel.g_hat, channel.h_hat,
                                     cfg.system.csi_error_scale, scenarios, rng)
    scenario_channels = [ChannelRealization(g, h, channel.g_hat, channel.h_hat)
                         for g, h in zip(gs, hs)]
    elements = channel.g.shape[0]
    phase_candidates = [random_phases(elements, rng) for _ in range(candidates)]
    phase_candidates.append(np.zeros(elements))
    records = []
    for phases in phase_candidates:
        rates = np.concatenate([np.asarray(evaluate_phases(sc, phases,
                            cfg.system.transmit_power, cfg.system.noise_power,
                            )["rates"]) for sc in scenario_channels])
        records.append({"phases": phases, "mean": float(rates.mean()),
                        "p05": float(np.quantile(rates, 0.05)),
                        "feasible": bool(np.quantile(rates, 0.05) >= reliability_target)})
    feasible = [r for r in records if r["feasible"]]
    pool = feasible or records
    chosen = max(pool, key=(lambda r: (r["mean"], r["p05"])) if feasible
                 else (lambda r: (r["p05"], r["mean"])))
    return chosen["phases"], {"candidate_count": len(records),
                              "feasible_count": len(feasible),
                              "reliability_target": reliability_target,
                              "selected_p05": chosen["p05"],
                              "constraint_satisfied": chosen["feasible"]}
