from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ChannelRealization:
    """Cascaded BS-RIS-user channels and the noisy CSI available to a controller."""

    g: np.ndarray  # RIS x BS
    h: np.ndarray  # users x RIS
    g_hat: np.ndarray
    h_hat: np.ndarray
    g_truth: np.ndarray | None = None
    h_truth: np.ndarray | None = None

    @property
    def truth_g(self) -> np.ndarray:
        return self.g_truth if self.g_truth is not None else self.g

    @property
    def truth_h(self) -> np.ndarray:
        return self.h_truth if self.h_truth is not None else self.h


def perturb_estimate(channel: ChannelRealization, error_scale: float,
                     rng: np.random.Generator) -> ChannelRealization:
    """Generate an independent CSI realization for held-out uncertainty tests."""
    truth_g, truth_h = channel.truth_g, channel.truth_h
    g_err = error_scale * np.linalg.norm(truth_g) * (rng.standard_normal(truth_g.shape) + 1j * rng.standard_normal(truth_g.shape)) / np.sqrt(2.0 * truth_g.size)
    h_err = error_scale * np.linalg.norm(truth_h) * (rng.standard_normal(truth_h.shape) + 1j * rng.standard_normal(truth_h.shape)) / np.sqrt(2.0 * truth_h.size)
    estimate_g, estimate_h = truth_g + g_err, truth_h + h_err
    return ChannelRealization(truth_g, truth_h, estimate_g, estimate_h, truth_g, truth_h)


def evolve_estimate(channel: ChannelRealization, error_scale: float,
                    temporal_rho: float, rng: np.random.Generator) -> ChannelRealization:
    """Evolve zero-mean estimation errors, preserving the signal mean."""
    if not 0 <= temporal_rho <= 1:
        raise ValueError("temporal_rho must be in [0, 1]")
    rho = float(temporal_rho)
    fresh = perturb_estimate(channel, error_scale, rng)
    g_hat = (channel.truth_g + rho * (channel.g_hat - channel.truth_g)
             + np.sqrt(1.0 - rho * rho) * (fresh.g_hat - channel.truth_g))
    h_hat = (channel.truth_h + rho * (channel.h_hat - channel.truth_h)
             + np.sqrt(1.0 - rho * rho) * (fresh.h_hat - channel.truth_h))
    return ChannelRealization(channel.truth_g, channel.truth_h, g_hat, h_hat,
                              channel.truth_g, channel.truth_h)


def estimate_trajectory(channel: ChannelRealization, error_scale: float,
                        temporal_rho: float, steps: int,
                        rng: np.random.Generator) -> list[ChannelRealization]:
    """Generate an observed CSI trajectory while keeping truth fixed."""
    if steps < 1:
        raise ValueError("trajectory steps must be positive")
    trajectory = [channel]
    current = channel
    for _ in range(max(0, int(steps) - 1)):
        current = evolve_estimate(current, error_scale, temporal_rho, rng)
        trajectory.append(current)
    return trajectory


def sample_decision_scenarios(g_hat: np.ndarray, h_hat: np.ndarray,
                              error_scale: float, count: int,
                              rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Local Gaussian uncertainty around observed CSI; never reads truth.

    This is a declared design surrogate, not the posterior of the simulator's
    additive observation model. It must be calibrated on development data.
    """
    if count < 1 or error_scale < 0:
        raise ValueError("invalid scenario count or error scale")
    def draw(center):
        noise = (rng.standard_normal((count, *center.shape))
                 + 1j * rng.standard_normal((count, *center.shape)))
        return center[None] + noise * error_scale * np.linalg.norm(center) / np.sqrt(2 * center.size)
    return draw(g_hat), draw(h_hat)


def _rician(shape: tuple[int, ...], k: float, rng: np.random.Generator) -> np.ndarray:
    los = np.ones(shape, dtype=np.complex128)
    nlos = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)
    return np.sqrt(k / (k + 1.0)) * los + np.sqrt(1.0 / (k + 1.0)) * nlos


def _ula_steering(length: int, angle: float) -> np.ndarray:
    index = np.arange(length, dtype=float)
    return np.exp(1j * np.pi * index * np.sin(angle)) / np.sqrt(length)


def _geometric_rician(shape: tuple[int, int], k: float, rng: np.random.Generator) -> np.ndarray:
    """Simple ULA geometric Rician link with random LoS angles."""
    rows, cols = shape
    a_r = _ula_steering(rows, rng.uniform(-np.pi / 2, np.pi / 2))
    a_t = _ula_steering(cols, rng.uniform(-np.pi / 2, np.pi / 2))
    los = np.outer(a_r, np.conjugate(a_t))
    nlos = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2 * rows)
    return np.sqrt(k / (k + 1.0)) * los + np.sqrt(1.0 / (k + 1.0)) * nlos


def sample_channel(bs_antennas: int, ris_elements: int, users: int, rician_k: float,
                   error_scale: float, rng: np.random.Generator) -> ChannelRealization:
    # Independent per-link normalization keeps RIS-size comparisons from
    # changing received power merely because more elements were simulated.
    g = _geometric_rician((ris_elements, bs_antennas), rician_k, rng)
    h = _geometric_rician((users, ris_elements), rician_k, rng)
    g_err = error_scale * np.linalg.norm(g) * (rng.standard_normal(g.shape) + 1j * rng.standard_normal(g.shape)) / np.sqrt(2.0 * g.size)
    h_err = error_scale * np.linalg.norm(h) * (rng.standard_normal(h.shape) + 1j * rng.standard_normal(h.shape)) / np.sqrt(2.0 * h.size)
    return ChannelRealization(g=g, h=h, g_hat=g + g_err, h_hat=h + h_err,
                              g_truth=g, h_truth=h)
