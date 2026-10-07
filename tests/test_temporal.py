import numpy as np
import pytest

from rislab.temporal import ComplexBelief, CorrelatedErrorPosterior, gaussian_update


def dense_posterior(observations, rho, r0, innovations, m0, v0):
    """Condition the entire Gaussian joint distribution, without whitening."""
    steps = len(observations)
    covariance = np.empty((steps, steps))
    marginal = r0
    for i in range(steps):
        if i:
            marginal = rho ** 2 * marginal + innovations[i - 1]
        covariance[i, i] = marginal
        for j in range(i):
            covariance[i, j] = covariance[j, i] = rho ** (i - j) * covariance[j, j]
    joint = covariance + v0 * np.ones((steps, steps))
    mean = m0 + v0 * np.ones(steps) @ np.linalg.solve(joint, observations - m0)
    variance = v0 - v0 ** 2 * np.ones(steps) @ np.linalg.solve(joint, np.ones(steps))
    return mean, variance


@pytest.mark.parametrize("rho", [0, 0.8, 0.95, 1])
def test_causal_estimator_equals_full_joint_conditioning_with_drifting_variance(rho):
    rng = np.random.default_rng(20261007)
    observations = rng.normal(size=(9, 3)) + 1j * rng.normal(size=(9, 3))
    means, variances = np.array([0.3 + 0.1j, 0, -0.2j]), np.array([0.2, 0.8, 1.2])
    r0 = np.array([0.1, 0.3, 0.5])
    q = np.tile(r0, (8, 1)) * np.array([0.3, 0.3, 0.6, 0.6, 2, 2, 0.2, 0.2])[:, None]
    estimator = CorrelatedErrorPosterior(means, variances, rho)
    for slot, observation in enumerate(observations):
        belief, _ = estimator.observe(observation, r0 if slot == 0 else q[slot - 1])
        for coefficient in range(3):
            expected_mean, expected_variance = dense_posterior(
                observations[:slot + 1, coefficient], rho, r0[coefficient],
                q[:slot, coefficient], means[coefficient], variances[coefficient])
            assert belief.mean[coefficient] == pytest.approx(expected_mean, abs=1e-11)
            assert belief.variance[coefficient] == pytest.approx(expected_variance, abs=1e-11)


def test_perfectly_correlated_repeated_observations_add_no_information():
    estimator = CorrelatedErrorPosterior(np.array([0j]), 1, 1)
    first, _ = estimator.observe(np.array([1 + 2j]), 0.3)
    for _ in range(20):
        later, nis = estimator.observe(np.array([1 + 2j]), 0)
        assert np.isnan(nis).all()
        np.testing.assert_array_equal(later.mean, first.mean)
        np.testing.assert_array_equal(later.variance, first.variance)
    naive = first
    for _ in range(20):
        naive, _ = gaussian_update(naive, np.array([1 + 2j]), 1, 0.3)
    assert naive.variance[0] < first.variance[0] / 10


def test_matched_complex_gaussian_posterior_is_calibrated_after_reported_drift():
    rng = np.random.default_rng(1701)
    count, rho, v0, r0 = 20000, 0.8, 0.7, 0.2

    def noise(v):
        return np.sqrt(v / 2) * (rng.standard_normal(count) + 1j * rng.standard_normal(count))

    mean = np.full(count, 0.4 - 0.2j)
    truth, error = mean + noise(v0), noise(r0)
    estimator = CorrelatedErrorPosterior(mean, v0, rho)
    for slot in range(12):
        q = (1 - rho ** 2) * r0 * (9 if slot >= 6 else 1)
        if slot:
            error = rho * error + noise(q)
        belief, nis = estimator.observe(truth + error, r0 if slot == 0 else q)
        standardized = np.abs(belief.mean - truth) ** 2 / belief.variance
        assert standardized.mean() == pytest.approx(1, abs=0.035)
        assert np.mean(standardized <= -np.log(0.05)) == pytest.approx(0.95, abs=0.008)
        assert nis.mean() == pytest.approx(1, abs=0.035)


def test_posterior_scenarios_have_declared_moments_and_do_not_use_truth():
    belief = ComplexBelief(np.array([1 + 2j, -1j]), np.array([0.2, 0.8]))
    scenarios = belief.sample(np.random.default_rng(13), 40000)
    np.testing.assert_allclose(scenarios.mean(axis=0), belief.mean, atol=0.015)
    np.testing.assert_allclose(np.mean(np.abs(scenarios - belief.mean) ** 2, axis=0), belief.variance, rtol=0.025)


def test_zero_error_is_exact_and_observation_copies_preserve_causality():
    estimator = CorrelatedErrorPosterior(np.zeros(3), 1, 0.8)
    observation = np.array([1j, 2, 3 - 1j])
    truth = observation.copy()
    belief, _ = estimator.observe(observation, 0)
    observation[:] = 100
    np.testing.assert_array_equal(belief.mean, truth)
    np.testing.assert_array_equal(belief.variance, np.zeros(3))
    again, _ = estimator.observe(truth, 0)
    np.testing.assert_allclose(again.mean, truth)
    with pytest.raises(ValueError, match="contradicts"):
        estimator.observe(truth + 1, 0)
