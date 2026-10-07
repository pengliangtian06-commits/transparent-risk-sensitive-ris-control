from dataclasses import replace

import numpy as np
import pytest

from rislab.channel import ChannelRealization, estimate_trajectory, evolve_estimate, sample_channel
from rislab.config import load_config
from rislab.control import optimize_phases
from rislab.cuda_backend import evaluate_batch
from rislab.metrics import effective_channel, equal_power_beamforming, evaluate_phases, user_rates
from rislab.sweep import _evaluate, summarize_seed_blocks


def small_config():
    cfg = load_config("configs/default.yaml")
    return replace(cfg, system=replace(cfg.system, ris_elements=8, users=3),
                   evaluation=replace(cfg.evaluation, matched_candidate_budget=32,
                                      channel_realizations=4, trajectory_steps=3))


def test_temporal_mean_does_not_inflate_at_zero_error():
    ch = sample_channel(2, 4, 2, 3, 0, np.random.default_rng(4))
    for rho in (0, 0.5, 0.8, 0.95, 1):
        trajectory = estimate_trajectory(ch, 0, rho, 20, np.random.default_rng(9))
        for slot in trajectory:
            np.testing.assert_allclose(slot.g_hat, ch.truth_g, atol=1e-12)
            np.testing.assert_allclose(slot.h_hat, ch.truth_h, atol=1e-12)


def test_error_process_has_stationary_variance_and_correlation():
    rng = np.random.default_rng(19)
    truth = np.ones((40, 40), dtype=complex)
    sigma, rho = 0.2, 0.8
    error = sigma * (rng.normal(size=truth.shape) + 1j * rng.normal(size=truth.shape)) / np.sqrt(2)
    ch = ChannelRealization(truth, truth, truth + error, truth + error)
    previous, current = [], []
    for _ in range(40):
        nxt = evolve_estimate(ch, sigma, rho, rng)
        previous.append((ch.g_hat - truth).ravel())
        current.append((nxt.g_hat - truth).ravel())
        ch = nxt
    a, b = np.concatenate(previous), np.concatenate(current)
    assert abs(np.mean(b)) < 0.01
    assert np.mean(np.abs(b) ** 2) == pytest.approx(sigma ** 2, rel=0.04)
    measured = np.vdot(a, b).real / np.vdot(a, a).real
    assert measured == pytest.approx(rho, abs=0.02)


def test_truth_evaluation_keeps_nominal_estimated_beams():
    ch = sample_channel(4, 8, 3, 2, 0.6, np.random.default_rng(17))
    phases = np.zeros(8)
    nominal = effective_channel(ch, phases, True)
    truth = effective_channel(ch, phases)
    beams = equal_power_beamforming(nominal, 1)
    expected = user_rates(truth, beams, 1e-3)
    actual = evaluate_phases(ch, phases, 1, 1e-3)
    np.testing.assert_allclose(actual["rates"], expected)
    _, batch_rates, _ = evaluate_batch(ch.truth_h[None], ch.truth_g[None], phases[None],
                                     1, 1e-3, beam_h=ch.h_hat[None], beam_g=ch.g_hat[None])
    np.testing.assert_allclose(batch_rates[0], expected, atol=1e-12)
    oracle = user_rates(truth, equal_power_beamforming(truth, 1), 1e-3)
    assert not np.allclose(expected, oracle)


@pytest.mark.parametrize("method", ["coordinate", "scenario_mean", "risk_constrained"])
def test_budget_counts_executed_candidate_scenario_pairs(monkeypatch, method):
    import rislab.control as control
    original = control.evaluate_batch
    calls = []

    def counted(h, g, phases, *args, **kwargs):
        calls.append(len(phases))
        return original(h, g, phases, *args, **kwargs)

    monkeypatch.setattr(control, "evaluate_batch", counted)
    cfg = small_config()
    ch = sample_channel(4, 8, 3, 2, 0.1, np.random.default_rng(5))
    phases, d = optimize_phases(ch.g_hat, ch.h_hat, cfg, np.random.default_rng(7), method)
    assert d["rate_evaluations"] == sum(calls) == 32
    assert d["candidate_count"] * d["scenario_count"] == 32
    grid = np.linspace(-np.pi, np.pi, 2 ** cfg.system.phase_bits, endpoint=False)
    assert np.all(np.isclose(phases[:, None], grid).any(axis=1))


def test_same_channel_streams_across_methods_and_all_slots_evaluated():
    cfg = small_config()
    rows = [_evaluate(cfg, method, 7, 2)
            for method in ("coordinate", "scenario_mean", "risk_constrained")]
    assert len({r["channel_stream_sha256"] for r in rows}) == 1
    for row in rows:
        assert np.asarray(row["episode_rates"]).shape == (2, 3, 3)
        assert len(row["episode_details"]) == 6
        assert [(d["episode"], d["slot"]) for d in row["episode_details"]] == list(
            (episode, slot) for episode in range(2) for slot in range(3))


def test_infeasible_fallback_is_reported_and_monotone():
    cfg = small_config()
    ch = sample_channel(4, 8, 3, 2, 0.1, np.random.default_rng(5))
    _, short = optimize_phases(ch.g_hat, ch.h_hat, cfg, np.random.default_rng(7),
                               target=1e6, budget=4)
    _, long = optimize_phases(ch.g_hat, ch.h_hat, cfg, np.random.default_rng(7),
                              target=1e6, budget=32)
    assert long["fallback"] and not long["constraint_satisfied"]
    assert long["selected_p05"] >= short["selected_p05"]


def test_per_user_risk_scope_does_not_hide_one_weak_user(monkeypatch):
    import rislab.control as control

    cfg = small_config()
    cfg = replace(cfg, system=replace(cfg.system, users=4),
                  evaluation=replace(cfg.evaluation, matched_candidate_budget=16,
                                     channel_realizations=4))
    ch = sample_channel(4, 8, 4, 2, 0.1, np.random.default_rng(91))

    def fake_batch(h, g, phases, *args, **kwargs):
        phases = np.asarray(phases)
        rates = np.full((len(phases), 4), 8.0)
        for index, phase in enumerate(phases):
            if np.allclose(phase, 0.0):
                # One weak user in one scenario is masked by a pooled quantile
                # but remains visible to the per-user lower-tail check.
                if index % 4 == 0:
                    rates[index] = [10.0, 0.1, 10.0, 10.0]
                else:
                    rates[index] = 10.0
        return None, rates, {"backend": "test"}

    monkeypatch.setattr(control, "evaluate_batch", fake_batch)
    pooled_phases, pooled = control.optimize_phases(
        ch.g_hat, ch.h_hat, cfg, np.random.default_rng(3),
        method="risk_constrained", scenario_count=4, target=5.0,
        risk_scope="pooled")
    per_user_phases, per_user = control.optimize_phases(
        ch.g_hat, ch.h_hat, cfg, np.random.default_rng(3),
        method="risk_constrained", scenario_count=4, target=5.0,
        risk_scope="per_user")
    assert np.allclose(pooled_phases, 0.0)
    assert not np.allclose(per_user_phases, 0.0)
    assert pooled["constraint_satisfied"] is True
    assert per_user["constraint_satisfied"] is True
    assert pooled["risk_scope"] == "pooled"
    assert per_user["risk_scope"] == "per_user"
    assert per_user["selected_user_p05"][1] >= 8.0


def test_seed_block_ci_recomputes_quantile_instead_of_mean():
    row = dict(error=0.1, temporal_rho=0.8, elements=8, users=2,
               method="coordinate", outage_threshold=0.5, quantile_level=0.05,
               decision_seconds_mean=0)
    rows = [{**row, "seed": seed, "episode_rates": [[[1, 100], [1, 100]]]}
            for seed in (1, 2, 3)]
    summary = summarize_seed_blocks(rows, draws=100)[0]
    assert summary["pooled_user_p05_rate"] == 1
    assert summary["pooled_user_p05_rate_ci_low"] == 1
    assert summary["pooled_user_p05_rate_ci_high"] == 1
    assert summary["sum_rate"] == 101
    assert summary["ci_unit"] == "independent_seed_block"


def test_invalid_backend_is_rejected():
    with pytest.raises(ValueError, match="backend"):
        evaluate_batch(np.ones((1, 2, 2)), np.ones((1, 2, 2)), np.zeros((1, 2)), 1, 0.1,
                       backend="unsupported")
