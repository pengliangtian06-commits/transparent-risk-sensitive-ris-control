from __future__ import annotations

import numpy as np


def effective_channel(channel, phases: np.ndarray, use_estimate: bool = False) -> np.ndarray:
    theta = np.exp(1j * phases)
    h = channel.h_hat if use_estimate else channel.truth_h
    g = channel.g_hat if use_estimate else channel.truth_g
    return (h * theta[None, :]) @ g


def equal_power_beamforming(effective: np.ndarray, transmit_power: float) -> np.ndarray:
    users, antennas = effective.shape
    beams = np.conjugate(effective.T)
    norms = np.linalg.norm(beams, axis=0, keepdims=True) + 1e-12
    beams = beams / norms
    return beams * np.sqrt(transmit_power / max(users, 1))


def user_rates(effective: np.ndarray, beams: np.ndarray, noise_power: float) -> np.ndarray:
    gains = np.abs(effective @ beams) ** 2
    desired = np.diag(gains)
    interference = gains.sum(axis=1) - desired
    return np.log2(1.0 + desired / (noise_power + interference + 1e-12))


def evaluate_phases(channel, phases: np.ndarray, transmit_power: float, noise_power: float,
                    use_estimate: bool = False) -> dict:
    eff = effective_channel(channel, phases, use_estimate=use_estimate)
    nominal = effective_channel(channel, phases, use_estimate=True)
    beams = equal_power_beamforming(nominal, transmit_power)
    rates = user_rates(eff, beams, noise_power)
    return {"sum_rate": float(rates.sum()), "mean_user_rate": float(rates.mean()),
            "p05_user_rate": float(np.quantile(rates, 0.05)), "rates": rates.tolist()}


def outage_probability(rates: np.ndarray, threshold: float) -> float:
    """Fraction of users below a declared service-rate threshold."""
    return float(np.mean(np.asarray(rates) < threshold))
