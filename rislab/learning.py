"""Lightweight learned controller that has no framework or GPU dependency.

The policy is a random-feature regressor trained against the robust teacher
controller. It is intentionally transparent and reproducible; its purpose is
to establish the online-control interface before introducing a larger neural
policy on CUDA.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .baselines import risk_constrained_phases
from .channel import sample_channel
from .metrics import evaluate_phases


def observation(channel) -> np.ndarray:
    """Scale-invariant real observation from the available noisy CSI."""
    g, h = channel.g_hat, channel.h_hat
    parts = [np.real(g).ravel(), np.imag(g).ravel(), np.real(h).ravel(), np.imag(h).ravel()]
    x = np.concatenate(parts).astype(np.float64)
    scale = np.linalg.norm(x) + 1e-12
    return x / scale


def feature_dimension(bs_antennas: int, ris_elements: int, users: int) -> int:
    return 2 * (ris_elements * bs_antennas + users * ris_elements)


@dataclass
class RandomFeaturePolicy:
    input_dim: int
    output_dim: int
    hidden_dim: int
    seed: int
    weights: np.ndarray | None = None
    readout: np.ndarray | None = None

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.weights = rng.normal(0, 1 / np.sqrt(self.input_dim),
                                  size=(self.hidden_dim, self.input_dim))
        self.readout = np.zeros((self.output_dim, self.hidden_dim + 1))

    def _features(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate([np.tanh(self.weights @ x), np.ones(1)])

    def fit(self, observations: np.ndarray, targets: np.ndarray, ridge: float = 1e-3) -> None:
        phi = np.stack([self._features(x) for x in observations])
        gram = phi.T @ phi + ridge * np.eye(phi.shape[1])
        self.readout = np.linalg.solve(gram, phi.T @ targets).T

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.readout is None:
            raise RuntimeError("policy must be fitted before predict")
        return np.pi * np.tanh(self.readout @ self._features(x))


def train_teacher_policy(cfg, seed: int, episodes: int = 32,
                         scenarios: int = 4, candidates: int = 12) -> RandomFeaturePolicy:
    rng = np.random.default_rng(seed)
    channels = [sample_channel(cfg.system.bs_antennas, cfg.system.ris_elements,
                               cfg.system.users, cfg.system.rician_k,
                               cfg.system.csi_error_scale, rng) for _ in range(episodes)]
    xs = np.stack([observation(ch) for ch in channels])
    ys = np.stack([risk_constrained_phases(ch, cfg, rng, scenarios, candidates)[0]
                   for ch in channels])
    policy = RandomFeaturePolicy(xs.shape[1], ys.shape[1], cfg.training.hidden_dim, seed)
    policy.fit(xs, ys)
    return policy


def train_policy_split(cfg, train_seed: int, test_seed: int, train_episodes: int,
                       test_episodes: int, scenarios: int = 4,
                       candidates: int = 12) -> dict:
    """Train and evaluate with disjoint channel seeds and raw user rates."""
    policy = train_teacher_policy(cfg, train_seed, train_episodes, scenarios, candidates)
    result = evaluate_policy(cfg, policy, test_seed, test_episodes)
    result.update({"train_seed": train_seed, "test_seed": test_seed,
                   "train_episodes": train_episodes, "test_episodes": test_episodes,
                   "policy_parameters": int(policy.readout.size + policy.weights.size)})
    return result


def evaluate_policy(cfg, policy: RandomFeaturePolicy, seed: int, episodes: int = 20) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(episodes):
        ch = sample_channel(cfg.system.bs_antennas, cfg.system.ris_elements,
                            cfg.system.users, cfg.system.rician_k,
                            cfg.system.csi_error_scale, rng)
        phases = policy.predict(observation(ch))
        rows.append(evaluate_phases(ch, phases, cfg.system.transmit_power, cfg.system.noise_power))
    all_rates = np.concatenate([np.asarray(r["rates"]) for r in rows])
    return {"sum_rate_mean": float(np.mean([r["sum_rate"] for r in rows])),
            "sum_rate_std": float(np.std([r["sum_rate"] for r in rows])),
            "p05_user_rate_mean": float(np.mean([r["p05_user_rate"] for r in rows])),
            "pooled_user_p05_rate": float(np.quantile(all_rates, 0.05)),
            "outage_probability_0p5": float(np.mean(all_rates < 0.5)),
            "policy_decision_candidate_count": 1}
