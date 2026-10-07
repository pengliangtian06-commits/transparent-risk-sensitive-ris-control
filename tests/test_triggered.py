import numpy as np
import pytest

from dataclasses import replace

from rislab.channel import sample_channel
from rislab.config import load_config
from rislab.triggered import TriggerPolicy, make_controller


def _config():
    cfg = load_config("configs/default.yaml")
    return replace(
        cfg,
        system=replace(cfg.system, ris_elements=8, users=3, csi_error_scale=0.12,
                        temporal_rho=0.8, phase_bits=2),
        evaluation=replace(cfg.evaluation, matched_candidate_budget=8,
                           channel_realizations=2, backend="numpy"),
    )


def _observation(seed=7):
    cfg = _config()
    channel = sample_channel(cfg.system.bs_antennas, cfg.system.ris_elements,
                             cfg.system.users, cfg.system.rician_k,
                             cfg.system.csi_error_scale, np.random.default_rng(seed))
    return cfg, channel.g_hat, channel.h_hat


def test_initial_refresh_and_quantized_phase_alphabet():
    cfg, g_hat, h_hat = _observation()
    controller = make_controller(
        "posterior_triggered", cfg, np.random.default_rng(17),
        TriggerPolicy(refresh_period=10, scenario_count=2, uncertainty_threshold=99,
                      innovation_threshold=99),
    )
    phases, posterior_g, posterior_h, meta = controller.step(g_hat, h_hat, 0)
    grid = np.linspace(-np.pi, np.pi, 2 ** cfg.system.phase_bits, endpoint=False)
    assert phases.shape == (cfg.system.ris_elements,)
    assert posterior_g.shape == g_hat.shape
    assert posterior_h.shape == h_hat.shape
    assert meta["refreshed"] is True
    assert meta["trigger"] == "initial"
    assert meta["candidate_count"] > 0
    assert meta["rate_evaluations"] == cfg.evaluation.matched_candidate_budget
    assert np.all(np.min(np.abs(np.angle(np.exp(1j * (phases[:, None] - grid[None])))), axis=1) < 1e-10)


def test_reuse_has_zero_candidate_evaluations():
    cfg, g_hat, h_hat = _observation()
    controller = make_controller(
        "posterior_triggered", cfg, np.random.default_rng(18),
        TriggerPolicy(refresh_period=10, scenario_count=2, uncertainty_threshold=99,
                      innovation_threshold=99),
    )
    first, *_ = controller.step(g_hat, h_hat, 0)
    second, *_rest, meta = controller.step(g_hat, h_hat, 1)
    assert np.array_equal(first, second)
    assert meta["trigger"] == "reuse"
    assert meta["candidate_count"] == 0
    assert meta["rate_evaluations"] == 0


def test_period_and_uncertainty_triggers():
    cfg, g_hat, h_hat = _observation()
    periodic = make_controller(
        "posterior_triggered", cfg, np.random.default_rng(19),
        TriggerPolicy(refresh_period=2, scenario_count=2, uncertainty_threshold=99,
                      innovation_threshold=99),
    )
    periodic.step(g_hat, h_hat, 0)
    _, _, _, period_meta = periodic.step(g_hat, h_hat, 2)
    assert period_meta["trigger"] == "period"
    uncertain = make_controller(
        "posterior_triggered", cfg, np.random.default_rng(20),
        TriggerPolicy(refresh_period=99, scenario_count=2, uncertainty_threshold=0,
                      innovation_threshold=99),
    )
    uncertain.step(g_hat, h_hat, 0)
    _, _, _, uncertainty_meta = uncertain.step(g_hat, h_hat, 1)
    assert uncertainty_meta["trigger"] == "uncertainty"


def test_controller_reproducibility_and_no_truth_argument():
    cfg, g_hat, h_hat = _observation(23)
    policy = TriggerPolicy(refresh_period=3, scenario_count=2)
    c1 = make_controller("posterior_triggered", cfg, np.random.default_rng(31), policy)
    c2 = make_controller("posterior_triggered", cfg, np.random.default_rng(31), policy)
    for slot in range(3):
        out1 = c1.step(g_hat, h_hat, slot)
        out2 = c2.step(g_hat, h_hat, slot)
        np.testing.assert_allclose(out1[0], out2[0])
        assert out1[3]["trigger"] == out2[3]["trigger"]
        assert out1[3]["candidate_count"] == out2[3]["candidate_count"]


def test_risk_guard_reuses_or_escalates_causally():
    cfg, g_hat, h_hat = _observation(29)
    controller = make_controller(
        "posterior_triggered", cfg, np.random.default_rng(32),
        TriggerPolicy(refresh_period=99, uncertainty_threshold=99,
                      innovation_threshold=99, scenario_count=2,
                      guard_budget=2, guard_target=0.5),
    )
    first = controller.step(g_hat, h_hat, 0)
    second = controller.step(g_hat, h_hat, 1)
    assert first[3]["refreshed"] is True
    assert second[3]["rate_evaluations"] >= 2
    assert second[3]["trigger"] in {"guard", "reuse"}


def test_guard_can_use_smaller_scenario_pool_than_refresh():
    cfg, g_hat, h_hat = _observation(30)
    controller = make_controller(
        "posterior_triggered", cfg, np.random.default_rng(33),
        TriggerPolicy(refresh_period=99, uncertainty_threshold=99,
                      innovation_threshold=99, scenario_count=4,
                      guard_scenario_count=2, guard_budget=2, guard_target=0.5),
    )
    controller.step(g_hat, h_hat, 0)
    _, _, _, meta = controller.step(g_hat, h_hat, 1)
    if meta["trigger"] == "reuse":
        assert meta["guard_scenario_count"] == 2
        assert meta["guard_rate_evaluations"] == 2
    else:
        assert meta["trigger"] == "guard"
        assert meta["guard_scenario_count"] == 2
        assert meta["guard_rate_evaluations"] == 2


def test_adaptive_guard_escalates_then_forces_full_refresh():
    cfg, g_hat, h_hat = _observation(31)
    controller = make_controller(
        "posterior_triggered", cfg, np.random.default_rng(34),
        TriggerPolicy(
            refresh_period=99, min_refresh_gap=1, scenario_count=4,
            uncertainty_threshold=99, innovation_threshold=99,
            guard_budget=1, guard_scenario_count=1, guard_target=0.0,
            adaptive_guard=True, guard_full_refresh_after=3,
            guard_escalation_scenario_count=2, guard_escalation_budget=2,
            guard_escalation_uncertainty_threshold=0.0,
            guard_escalation_innovation_threshold=99,
            guard_release_uncertainty_threshold=0.0,
            guard_release_innovation_threshold=99,
        ),
    )
    first = controller.step(g_hat, h_hat, 0)
    second = controller.step(g_hat, h_hat, 1)
    third = controller.step(g_hat, h_hat, 2)
    assert first[3]["trigger"] == "initial"
    assert second[3]["adaptive_guard"] is True
    assert second[3]["guard_mode"] == "escalated"
    assert second[3]["guard_scenario_count"] == 2
    assert second[3]["guard_rate_evaluations"] == 2
    assert third[3]["trigger"] == "anomaly"
    assert third[3]["refreshed"] is True
def test_per_user_scope_is_propagated_to_risk_baselines():
    cfg, g_hat, h_hat = _observation(37)
    policy = TriggerPolicy(refresh_period=3, scenario_count=2, risk_scope="per_user",
                           baseline_risk_scope="per_user")
    periodic = make_controller("periodic_risk", cfg, np.random.default_rng(41), policy)
    always = make_controller("always_risk", cfg, np.random.default_rng(42), policy)
    assert periodic.risk_scope == "per_user"
    assert always.risk_scope == "per_user"
    _, _, _, periodic_meta = periodic.step(g_hat, h_hat, 0)
    _, _, _, always_meta = always.step(g_hat, h_hat, 0)
    assert periodic_meta["risk_scope"] == "per_user"
    assert always_meta["risk_scope"] == "per_user"


def test_baseline_period_can_be_frozen_independently_of_trigger_period():
    cfg, g_hat, h_hat = _observation(38)
    policy = TriggerPolicy(refresh_period=4, baseline_refresh_period=2,
                           scenario_count=2)
    periodic = make_controller("periodic_risk", cfg, np.random.default_rng(43), policy)
    assert periodic.period == 2
    periodic.step(g_hat, h_hat, 0)
    _, _, _, meta = periodic.step(g_hat, h_hat, 1)
    assert meta["trigger"] == "reuse"
    _, _, _, meta = periodic.step(g_hat, h_hat, 2)
    assert meta["trigger"] == "period"
    assert meta["reliability_target"] == cfg.evaluation.outage_threshold


def test_invalid_risk_scope_is_rejected():
    with pytest.raises(ValueError, match="risk_scope"):
        TriggerPolicy(risk_scope="user_average")
    with pytest.raises(ValueError, match="scenario_inflation"):
        TriggerPolicy(scenario_inflation=0.0)
    with pytest.raises(ValueError, match="guard_budget"):
        TriggerPolicy(scenario_count=4, guard_scenario_count=2, guard_budget=1)
    with pytest.raises(ValueError, match="baseline_refresh_period"):
        TriggerPolicy(baseline_refresh_period=0)
    with pytest.raises(ValueError, match="baseline_risk_scope"):
        TriggerPolicy(baseline_risk_scope="mean")
    with pytest.raises(ValueError, match="baseline_reliability_target"):
        TriggerPolicy(baseline_reliability_target=1.1)
