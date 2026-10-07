import numpy as np

from rislab.channel import sample_channel
from rislab.metrics import evaluate_phases
from rislab.metrics import outage_probability
from rislab.baselines import uncertainty_aware_phases
from rislab.config import load_config
from rislab.learning import evaluate_policy, train_teacher_policy


def test_channel_shapes_and_finite_metric():
    rng = np.random.default_rng(3)
    ch = sample_channel(4, 8, 3, 2.0, 0.1, rng)
    assert ch.g.shape == (8, 4)
    assert ch.h.shape == (3, 8)
    result = evaluate_phases(ch, np.zeros(8), 1.0, 1e-3)
    assert result["sum_rate"] >= 0.0
    assert np.isfinite(result["p05_user_rate"])


def test_uncertainty_aware_controller_is_reproducible():
    cfg = load_config("configs/default.yaml")
    ch = sample_channel(4, 8, 3, 2.0, 0.1, np.random.default_rng(5))
    p1 = uncertainty_aware_phases(ch, cfg, np.random.default_rng(9), scenarios=2, candidates=3)
    p2 = uncertainty_aware_phases(ch, cfg, np.random.default_rng(9), scenarios=2, candidates=3)
    assert np.allclose(p1, p2)
    assert p1.shape == (ch.g.shape[0],)


def test_learned_policy_smoke():
    cfg = load_config("configs/default.yaml")
    policy = train_teacher_policy(cfg, seed=4, episodes=4, scenarios=1, candidates=2)
    result = evaluate_policy(cfg, policy, seed=5, episodes=2)
    assert np.isfinite(result["sum_rate_mean"])
    assert result["p05_user_rate_mean"] >= 0


def test_correlated_estimate_preserves_truth_and_shapes():
    from rislab.channel import evolve_estimate, sample_channel
    rng = np.random.default_rng(11)
    ch = sample_channel(4, 8, 3, 3.0, 0.1, rng)
    nxt = evolve_estimate(ch, 0.1, 0.8, rng)
    assert nxt.g_hat.shape == ch.g_hat.shape
    assert nxt.h_hat.shape == ch.h_hat.shape
    np.testing.assert_array_equal(nxt.truth_g, ch.truth_g)
    np.testing.assert_array_equal(nxt.truth_h, ch.truth_h)


def test_outage_metric_boundary():
    assert outage_probability(np.array([0.2, 0.5, 0.7]), 0.5) == 1 / 3
