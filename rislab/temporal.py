"""Causal Gaussian conditioning for a fixed channel observed with AR(1) error."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ComplexBelief:
    mean: np.ndarray
    variance: np.ndarray

    def sample(self, rng: np.random.Generator, count: int) -> np.ndarray:
        if count < 1:
            raise ValueError("count must be positive")
        noise = (rng.standard_normal((count, *self.mean.shape))
                 + 1j * rng.standard_normal((count, *self.mean.shape))) / np.sqrt(2)
        return self.mean[None] + np.sqrt(self.variance)[None] * noise


def _variance(value, shape):
    array = np.array(np.broadcast_to(np.asarray(value, dtype=float), shape), copy=True)
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError("variance must be finite and nonnegative")
    return array


def gaussian_update(belief: ComplexBelief, measurement, coefficient, noise_variance):
    """Condition x on measurement = coefficient*x + proper complex noise."""
    y = np.asarray(measurement, dtype=np.complex128)
    if y.shape != belief.mean.shape or not np.all(np.isfinite(y)):
        raise ValueError("measurement must be finite and have the prior shape")
    q = _variance(noise_variance, y.shape)
    residual = y - coefficient * belief.mean
    predictive_variance = coefficient ** 2 * belief.variance + q
    positive = predictive_variance > 0
    if np.any(~positive & (np.abs(residual) > 1e-10)):
        raise ValueError("measurement contradicts a deterministic prior/noise model")
    gain = np.divide(coefficient * belief.variance, predictive_variance,
                     out=np.zeros_like(predictive_variance), where=positive)
    mean = belief.mean + gain * residual
    # Joseph form avoids negative variances from subtractive cancellation.
    variance = (1 - gain * coefficient) ** 2 * belief.variance + gain ** 2 * q
    normalized_innovation = np.divide(np.abs(residual) ** 2, predictive_variance,
                                     out=np.full_like(predictive_variance, np.nan), where=positive)
    return ComplexBelief(mean, variance), normalized_innovation


class CorrelatedErrorPosterior:
    """Exact posterior under declared prior/noise parameters, not under mismatch.

    The first update receives Var(e0). Later updates receive Var(ut) where
    et = rho*e[t-1] + ut. No true channel or future observation is accepted.
    """

    def __init__(self, prior_mean, prior_variance, rho: float):
        if not np.isfinite(rho) or not 0 <= rho <= 1:
            raise ValueError("rho must be in [0, 1]")
        mean = np.array(prior_mean, dtype=np.complex128, copy=True)
        if not np.all(np.isfinite(mean)):
            raise ValueError("prior mean must be finite")
        self.belief = ComplexBelief(mean, _variance(prior_variance, mean.shape))
        self.rho = float(rho)
        self.previous_observation = None

    def observe(self, observation, noise_variance):
        y = np.asarray(observation, dtype=np.complex128)
        if y.shape != self.belief.mean.shape or not np.all(np.isfinite(y)):
            raise ValueError("observation must be finite and have the prior shape")
        if self.previous_observation is None:
            measurement, coefficient = y, 1.0
        else:
            # Whitening exposes an independent innovation even when qt drifts.
            measurement = y - self.rho * self.previous_observation
            coefficient = 1 - self.rho
        updated, nis = gaussian_update(self.belief, measurement, coefficient, noise_variance)
        self.belief = updated
        self.previous_observation = y.copy()
        return ComplexBelief(updated.mean.copy(), updated.variance.copy()), nis


def _covariance(value, dimension, name):
    """Broadcast a scalar/diagonal/full covariance and validate it."""
    array = np.asarray(value, dtype=np.complex128)
    if array.ndim == 0:
        result = np.eye(dimension, dtype=np.complex128) * complex(array)
    elif array.ndim == 1:
        if array.shape != (dimension,):
            raise ValueError(f"{name} has wrong diagonal shape")
        result = np.diag(array)
    elif array.shape == (dimension, dimension):
        result = np.array(array, copy=True)
    else:
        raise ValueError(f"{name} has wrong shape")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    result = (result + result.conj().T) / 2.0
    eigenvalues = np.linalg.eigvalsh(result)
    if np.min(eigenvalues) < -1e-10:
        raise ValueError(f"{name} must be positive semidefinite")
    return result


class AugmentedStatePosterior:
    """Causal dense Gaussian filter for a latent channel plus AR CSI error.

    The supplied observation is a noisy coefficient estimate z_t=x_t+e_t.
    The state is [x_t, e_t], with x_t=a*x_(t-1)+w_t and
    e_t=rho*e_(t-1)+u_t. The filter receives only observations and declared
    covariance metadata; it never accepts truth or realized errors.
    """

    def __init__(self, prior_mean, prior_covariance, initial_error_covariance,
                 rho: float, channel_ar: float = 1.0,
                 initial_cross_covariance=None):
        mean = np.asarray(prior_mean, dtype=np.complex128)
        if mean.ndim != 1 or not np.all(np.isfinite(mean)):
            raise ValueError("prior_mean must be a finite vector")
        dimension = mean.size
        if not (np.isfinite(rho) and 0 <= rho <= 1):
            raise ValueError("rho must be in [0, 1]")
        if not (np.isfinite(channel_ar) and 0 <= channel_ar <= 1):
            raise ValueError("channel_ar must be in [0, 1]")
        channel_cov = _covariance(prior_covariance, dimension, "prior_covariance")
        error_cov = _covariance(initial_error_covariance, dimension, "initial_error_covariance")
        cross = np.zeros((dimension, dimension), dtype=np.complex128) if initial_cross_covariance is None else np.asarray(initial_cross_covariance, dtype=np.complex128)
        if cross.shape != (dimension, dimension) or not np.all(np.isfinite(cross)):
            raise ValueError("initial_cross_covariance has wrong shape or is not finite")
        state_cov = np.block([[channel_cov, cross], [cross.conj().T, error_cov]])
        if np.min(np.linalg.eigvalsh((state_cov + state_cov.conj().T) / 2.0)) < -1e-10:
            raise ValueError("joint initial covariance must be positive semidefinite")
        self.dimension = dimension
        self.rho = float(rho)
        self.channel_ar = float(channel_ar)
        self.state_mean = np.concatenate([mean, np.zeros(dimension, dtype=np.complex128)])
        self.state_covariance = (state_cov + state_cov.conj().T) / 2.0
        self.channel_process_covariance = (1.0 - self.channel_ar ** 2) * channel_cov
        self._observed = False
        self._identity = np.eye(2 * dimension, dtype=np.complex128)
        self._transition = np.block([
            [self.channel_ar * np.eye(dimension), np.zeros((dimension, dimension))],
            [np.zeros((dimension, dimension)), self.rho * np.eye(dimension)],
        ]).astype(np.complex128)
        self._observation = np.concatenate([
            np.eye(dimension, dtype=np.complex128),
            np.eye(dimension, dtype=np.complex128),
        ], axis=1)

    def _predict(self, innovation_covariance):
        q_error = _covariance(innovation_covariance, self.dimension, "innovation_covariance")
        process = np.block([
            [self.channel_process_covariance, np.zeros((self.dimension, self.dimension))],
            [np.zeros((self.dimension, self.dimension)), q_error],
        ]).astype(np.complex128)
        self.state_mean = self._transition @ self.state_mean
        self.state_covariance = self._transition @ self.state_covariance @ self._transition.conj().T + process
        self.state_covariance = (self.state_covariance + self.state_covariance.conj().T) / 2.0

    def observe(self, observation, innovation_covariance=None):
        """Consume one causal LS observation and return channel belief/NIS."""
        z = np.asarray(observation, dtype=np.complex128)
        if z.shape != (self.dimension,) or not np.all(np.isfinite(z)):
            raise ValueError("observation must have the state coefficient shape")
        if self._observed:
            if innovation_covariance is None:
                raise ValueError("later observations require innovation covariance")
            self._predict(innovation_covariance)
        residual = z - self._observation @ self.state_mean
        innovation_covariance = self._observation @ self.state_covariance @ self._observation.conj().T
        innovation_covariance = (innovation_covariance + innovation_covariance.conj().T) / 2.0
        rank = np.linalg.matrix_rank(innovation_covariance, tol=1e-10)
        inverse = np.linalg.pinv(innovation_covariance, hermitian=True)
        gain = self.state_covariance @ self._observation.conj().T @ inverse
        self.state_mean = self.state_mean + gain @ residual
        correction = self._identity - gain @ self._observation
        self.state_covariance = correction @ self.state_covariance @ correction.conj().T
        self.state_covariance = (self.state_covariance + self.state_covariance.conj().T) / 2.0
        self._observed = True
        nis = np.nan if rank < self.dimension else float(np.real(residual.conj().T @ inverse @ residual))
        return self.channel_belief(), nis

    def channel_belief(self) -> ComplexBelief:
        return ComplexBelief(self.state_mean[:self.dimension].copy(),
                             np.real(np.diag(self.state_covariance[:self.dimension, :self.dimension])).copy())

    @property
    def channel_covariance(self):
        return self.state_covariance[:self.dimension, :self.dimension].copy()

    @property
    def error_covariance(self):
        return self.state_covariance[self.dimension:, self.dimension:].copy()
