import numpy as np
import pytest

from rislab.temporal import AugmentedStatePosterior


def test_augmented_posterior_matches_dense_joint_conditioning_first_slot():
    mean = np.array([0.2 + 0.1j, -0.3 + 0.2j])
    channel_cov = np.array([[0.8, 0.1j], [-0.1j, 0.6]], dtype=complex)
    error_cov = np.array([[0.2, 0.03], [0.03, 0.15]], dtype=complex)
    cross = np.array([[0.01, 0.0], [0.0, -0.01]], dtype=complex)
    posterior = AugmentedStatePosterior(mean, channel_cov, error_cov, rho=0.7,
                                        initial_cross_covariance=cross)
    observation = np.array([0.1 + 0.3j, -0.1 + 0.1j])
    belief, nis = posterior.observe(observation)
    H = np.hstack([np.eye(2), np.eye(2)])
    state_mean = np.r_[mean, np.zeros(2, dtype=complex)]
    state_cov = np.block([[channel_cov, cross], [cross.conj().T, error_cov]])
    innovation = H @ state_cov @ H.conj().T
    gain = state_cov @ H.conj().T @ np.linalg.inv(innovation)
    expected_mean = state_mean + gain @ (observation - H @ state_mean)
    expected_cov = state_cov - gain @ H @ state_cov
    assert np.allclose(posterior.state_mean, expected_mean)
    assert np.allclose(posterior.channel_covariance, expected_cov[:2, :2])
    assert np.isfinite(nis)
    assert np.allclose(belief.mean, expected_mean[:2])


def test_temporal_prediction_uses_declared_drift_covariance_and_is_causal():
    posterior = AugmentedStatePosterior(np.zeros(1, complex), np.array([[1.0]]),
                                        np.array([[0.2]]), rho=0.8, channel_ar=0.9)
    posterior.observe(np.array([0.4 + 0.1j]))
    previous = posterior.state_mean.copy()
    posterior.observe(np.array([0.3 - 0.2j]), np.array([[0.5]]))
    assert not np.allclose(previous, posterior.state_mean)
    assert posterior.channel_covariance.shape == (1, 1)


def test_singular_innovation_reports_undefined_nis():
    posterior = AugmentedStatePosterior(np.array([0.2 + 0.1j]), np.array([[0.0]]),
                                        np.array([[0.0]]), rho=1.0)
    _, nis = posterior.observe(np.array([0.2 + 0.1j]))
    assert np.isnan(nis)


def test_cross_covariance_must_keep_joint_covariance_psd():
    with pytest.raises(ValueError):
        AugmentedStatePosterior(np.zeros(1, complex), np.array([[1.0]]),
                                np.array([[1.0]]), rho=0.5,
                                initial_cross_covariance=np.array([[2.0]]))
