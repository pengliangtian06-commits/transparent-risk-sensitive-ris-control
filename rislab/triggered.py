"""Causal posterior-triggered RIS phase refresh controllers.

The controllers in this module receive only the current noisy CSI estimate.  A
posterior filter is used to produce a causal channel estimate and uncertainty
score; the simulator truth is never passed across this interface.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

from .config import Config
from .control import optimize_phases
from .temporal import AugmentedStatePosterior


@dataclass(frozen=True)
class TriggerPolicy:
    """Trigger parameters fixed before an evaluation campaign."""

    refresh_period: int = 3
    uncertainty_threshold: float = 0.20
    innovation_threshold: float = 3.0
    min_refresh_gap: int = 1
    scenario_count: int = 4
    reliability_target: float = 0.5
    guard_budget: int = 0
    guard_target: float | None = None
    risk_scope: str = "pooled"
    scenario_inflation: float = 1.0
    guard_scenario_count: int | None = None
    # Optional adaptive guard.  The legacy policy keeps the original
    # uncertainty/innovation -> full-refresh behavior when this is False.
    # When enabled, a high posterior/innovation event escalates the cheap
    # guard, hysteresis returns it to the normal guard, and consecutive high
    # events force a full refresh.
    adaptive_guard: bool = False
    guard_escalation_scenario_count: int | None = None
    guard_escalation_budget: int | None = None
    guard_escalation_uncertainty_threshold: float | None = None
    guard_escalation_innovation_threshold: float | None = None
    guard_release_uncertainty_threshold: float | None = None
    guard_release_innovation_threshold: float | None = None
    guard_full_refresh_after: int = 2
    baseline_refresh_period: int | None = None
    baseline_risk_scope: str = "pooled"
    baseline_reliability_target: float | None = None

    def __post_init__(self):
        if self.refresh_period < 1 or self.min_refresh_gap < 1:
            raise ValueError("refresh periods must be positive")
        if not np.isfinite(self.uncertainty_threshold) or self.uncertainty_threshold < 0:
            raise ValueError("uncertainty_threshold must be finite and nonnegative")
        if not np.isfinite(self.innovation_threshold) or self.innovation_threshold < 0:
            raise ValueError("innovation_threshold must be finite and nonnegative")
        if (self.scenario_count < 1 or not 0 <= self.reliability_target <= 1):
            raise ValueError("scenario_count must be positive and target in [0, 1]")
        if self.guard_budget < 0:
            raise ValueError("guard_budget must be nonnegative")
        guard_scenarios = (self.scenario_count if self.guard_scenario_count is None
                           else self.guard_scenario_count)
        if guard_scenarios < 1:
            raise ValueError("guard_scenario_count must be positive")
        if self.guard_budget and self.guard_budget < guard_scenarios:
            raise ValueError("guard_budget must cover the guard scenario count")
        if self.guard_target is not None and not 0 <= self.guard_target <= 1:
            raise ValueError("guard_target must be in [0, 1]")
        if self.risk_scope not in ("pooled", "per_user"):
            raise ValueError("risk_scope must be 'pooled' or 'per_user'")
        if (not np.isfinite(self.scenario_inflation) or
                self.scenario_inflation <= 0):
            raise ValueError("scenario_inflation must be finite and positive")
        if (self.baseline_refresh_period is not None and
                self.baseline_refresh_period < 1):
            raise ValueError("baseline_refresh_period must be positive")
        if self.baseline_risk_scope not in ("pooled", "per_user"):
            raise ValueError("baseline_risk_scope must be 'pooled' or 'per_user'")
        if (self.baseline_reliability_target is not None and
                (not np.isfinite(self.baseline_reliability_target) or
                 not 0 <= self.baseline_reliability_target <= 1)):
            raise ValueError("baseline_reliability_target must be in [0, 1]")
        if self.guard_full_refresh_after < 1:
            raise ValueError("guard_full_refresh_after must be positive")
        for name in (
            "guard_escalation_uncertainty_threshold",
            "guard_escalation_innovation_threshold",
            "guard_release_uncertainty_threshold",
            "guard_release_innovation_threshold",
        ):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        escalation_scenarios = (
            guard_scenarios if self.guard_escalation_scenario_count is None
            else self.guard_escalation_scenario_count
        )
        if escalation_scenarios < 1:
            raise ValueError("guard_escalation_scenario_count must be positive")
        escalation_budget = (
            self.guard_budget if self.guard_escalation_budget is None
            else self.guard_escalation_budget
        )
        if (self.guard_budget > 0 or self.adaptive_guard) and escalation_budget < escalation_scenarios:
            raise ValueError("guard_escalation_budget must cover escalation scenarios")
        if self.adaptive_guard and self.guard_budget < 1:
            raise ValueError("adaptive_guard requires a positive guard_budget")


class PosteriorTriggeredController:
    """Refresh RIS phases when posterior uncertainty or innovation is high.

    The latent channel is modeled as static over an evaluation trajectory and
    the CSI error follows an AR(1) process.  The augmented posterior is
    initialized from the first observation using only its observed power and
    the declared relative error scale.  At a refresh, scenario sampling is
    centered on the posterior mean and scaled by its posterior uncertainty.
    """

    def __init__(self, cfg: Config, rng: np.random.Generator,
                 policy: TriggerPolicy | None = None):
        self.cfg = cfg
        self.rng = rng
        self.policy = policy or TriggerPolicy()
        self._filter: AugmentedStatePosterior | None = None
        self._error_variance: float | None = None
        self._last_phases: np.ndarray | None = None
        self._last_refresh = -10**9
        self._last_decision = None
        self._guard_escalated = False
        self._anomaly_streak = 0

    def reset(self):
        self._filter = None
        self._error_variance = None
        self._last_phases = None
        self._last_refresh = -10**9
        self._last_decision = None
        self._guard_escalated = False
        self._anomaly_streak = 0

    @staticmethod
    def _vectorize(g_hat, h_hat):
        g = np.asarray(g_hat, dtype=np.complex128)
        h = np.asarray(h_hat, dtype=np.complex128)
        if g.ndim != 2 or h.ndim != 2 or not np.all(np.isfinite(g)) or not np.all(np.isfinite(h)):
            raise ValueError("CSI estimates must be finite matrices")
        return np.concatenate([g.ravel(), h.ravel()]), g.shape, h.shape

    def _posterior(self, observation: np.ndarray):
        if self._filter is None:
            power = max(float(np.mean(np.abs(observation) ** 2)), 1e-8)
            error_variance = float(self.cfg.system.csi_error_scale ** 2 * power)
            prior_variance = max(power - error_variance, 1e-8)
            dimension = observation.size
            self._error_variance = error_variance
            self._filter = AugmentedStatePosterior(
                np.zeros(dimension, dtype=np.complex128),
                np.eye(dimension, dtype=np.complex128) * prior_variance,
                np.eye(dimension, dtype=np.complex128) * error_variance,
                rho=self.cfg.system.temporal_rho,
                channel_ar=1.0,
            )
            belief, nis = self._filter.observe(observation)
        else:
            q = (1.0 - self.cfg.system.temporal_rho ** 2) * float(self._error_variance)
            belief, nis = self._filter.observe(
                observation,
                np.eye(observation.size, dtype=np.complex128) * q,
            )
        denominator = np.sqrt(np.mean(np.abs(belief.mean) ** 2) + 1e-12)
        uncertainty = float(np.sqrt(np.mean(belief.variance)) / denominator)
        nis_per_dimension = float(nis / observation.size) if np.isfinite(nis) else None
        return belief, uncertainty, nis_per_dimension

    def _refresh(self, decision_g, decision_h, initial, uncertainty):
        # The posterior uncertainty is a relative coefficient standard deviation
        # and therefore supplies the scenario scale without using channel truth.
        scenario_scale = float(np.clip(
            max(uncertainty, 0.0) * self.policy.scenario_inflation, 0.0, 2.0))
        effective_cfg = replace(
            self.cfg,
            system=replace(self.cfg.system, csi_error_scale=scenario_scale),
        )
        phases, diagnostics = optimize_phases(
            decision_g,
            decision_h,
            effective_cfg,
            self.rng,
            method="risk_constrained",
            initial=initial,
            scenario_count=self.policy.scenario_count,
            target=self.policy.reliability_target,
            risk_scope=self.policy.risk_scope,
        )
        self._last_phases = np.array(phases, copy=True)
        return phases, diagnostics, scenario_scale

    def _risk_guard(self, decision_g, decision_h, uncertainty,
                    scenario_count: int | None = None,
                    budget: int | None = None):
        """Evaluate the retained action with a small causal scenario budget.

        The guard never changes the phase vector.  It only determines whether
        the retained action still satisfies the same lower-tail target used by
        the full search; a failed guard escalates to a full refresh.
        """
        effective_cfg = replace(
            self.cfg,
            system=replace(
                self.cfg.system,
                csi_error_scale=float(np.clip(
                    max(uncertainty, 0.0) * self.policy.scenario_inflation, 0.0, 2.0)),
            ),
        )
        _, diagnostics = optimize_phases(
            decision_g,
            decision_h,
            effective_cfg,
            self.rng,
            method="risk_constrained",
            initial=self._last_phases,
            scenario_count=(self.policy.scenario_count
                             if scenario_count is None and self.policy.guard_scenario_count is None
                             else (self.policy.guard_scenario_count
                                   if scenario_count is None else scenario_count)),
            budget=self.policy.guard_budget if budget is None else budget,
            target=(self.policy.guard_target
                    if self.policy.guard_target is not None
                    else self.policy.reliability_target),
            risk_scope=self.policy.risk_scope,
        )
        return diagnostics

    def step(self, g_hat, h_hat, slot: int):
        """Consume one CSI observation and return phases plus audit metadata."""
        if slot < 0 or int(slot) != slot:
            raise ValueError("slot must be a nonnegative integer")
        observation, g_shape, h_shape = self._vectorize(g_hat, h_hat)
        belief, uncertainty, nis = self._posterior(observation)
        dimension = int(np.prod(g_shape))
        posterior_g = belief.mean[:dimension].reshape(g_shape)
        posterior_h = belief.mean[dimension:].reshape(h_shape)
        posterior_refresh = self._last_phases is None
        trigger = "initial" if posterior_refresh else None
        age = slot - self._last_refresh

        anomaly_high = False
        anomaly_low = False
        if self.policy.adaptive_guard:
            high_uncertainty = (
                self.policy.guard_escalation_uncertainty_threshold
                if self.policy.guard_escalation_uncertainty_threshold is not None
                else self.policy.uncertainty_threshold
            )
            high_innovation = (
                self.policy.guard_escalation_innovation_threshold
                if self.policy.guard_escalation_innovation_threshold is not None
                else self.policy.innovation_threshold
            )
            release_uncertainty = (
                self.policy.guard_release_uncertainty_threshold
                if self.policy.guard_release_uncertainty_threshold is not None
                else high_uncertainty
            )
            release_innovation = (
                self.policy.guard_release_innovation_threshold
                if self.policy.guard_release_innovation_threshold is not None
                else high_innovation
            )
            anomaly_high = bool(
                uncertainty >= high_uncertainty
                or (nis is not None and nis >= high_innovation)
            )
            anomaly_low = bool(
                uncertainty <= release_uncertainty
                and (nis is None or nis <= release_innovation)
            )
            self._anomaly_streak = self._anomaly_streak + 1 if anomaly_high else 0
            if not self._guard_escalated and anomaly_high:
                self._guard_escalated = True
            elif self._guard_escalated and anomaly_low:
                self._guard_escalated = False
        else:
            self._anomaly_streak = 0
            self._guard_escalated = False
        guard_mode = "escalated" if self._guard_escalated else "normal"
        if not posterior_refresh and age >= self.policy.min_refresh_gap:
            if age >= self.policy.refresh_period:
                trigger = "period"
            elif self.policy.adaptive_guard:
                if self._anomaly_streak >= self.policy.guard_full_refresh_after:
                    trigger = "anomaly"
            elif uncertainty >= self.policy.uncertainty_threshold:
                trigger = "uncertainty"
            elif nis is not None and nis >= self.policy.innovation_threshold:
                trigger = "innovation"
        refreshed = trigger is not None
        if refreshed:
            phases, diagnostics, scenario_scale = self._refresh(
                posterior_g, posterior_h, self._last_phases, uncertainty)
            self._last_refresh = slot
            self._last_decision = (posterior_g.copy(), posterior_h.copy())
        else:
            phases = np.array(self._last_phases, copy=True)
            scenario_scale = float(uncertainty)
            if self.policy.guard_budget:
                guard_scenario_count = (
                    self.policy.guard_escalation_scenario_count
                    if self._guard_escalated and self.policy.guard_escalation_scenario_count is not None
                    else (self.policy.guard_scenario_count
                          if self.policy.guard_scenario_count is not None
                          else self.policy.scenario_count)
                )
                guard_budget = (
                    self.policy.guard_escalation_budget
                    if self._guard_escalated and self.policy.guard_escalation_budget is not None
                    else self.policy.guard_budget
                )
                guard = self._risk_guard(
                    posterior_g, posterior_h, uncertainty,
                    scenario_count=guard_scenario_count,
                    budget=guard_budget,
                )
                if not guard["constraint_satisfied"]:
                    phases, diagnostics, scenario_scale = self._refresh(
                        posterior_g, posterior_h, self._last_phases, uncertainty)
                    full_candidate_count = diagnostics["candidate_count"]
                    full_rate_evaluations = diagnostics["rate_evaluations"]
                    full_budget_limit = diagnostics["budget_limit"]
                    diagnostics["candidate_count"] += guard["candidate_count"]
                    diagnostics["rate_evaluations"] += guard["rate_evaluations"]
                    diagnostics["budget_limit"] = full_budget_limit + guard["budget_limit"]
                    diagnostics["unused_budget"] = (
                        diagnostics["budget_limit"] - diagnostics["rate_evaluations"])
                    diagnostics["decision_seconds"] += guard["decision_seconds"]
                    diagnostics = {
                        **diagnostics,
                        "guard_passed": False,
                        "guard_constraint_satisfied": False,
                        "guard_candidate_count": guard["candidate_count"],
                        "guard_rate_evaluations": guard["rate_evaluations"],
                        "guard_budget_limit": guard["budget_limit"],
                        "guard_scenario_count": guard["scenario_count"],
                        "guard_mode": guard_mode,
                        "full_refresh_candidate_count": full_candidate_count,
                        "full_refresh_rate_evaluations": full_rate_evaluations,
                    }
                    self._last_refresh = slot
                    self._last_decision = (posterior_g.copy(), posterior_h.copy())
                    refreshed = True
                    trigger = "guard"
                else:
                    diagnostics = guard
                    diagnostics["guard_passed"] = True
                    diagnostics["guard_constraint_satisfied"] = True
                    diagnostics["guard_candidate_count"] = guard["candidate_count"]
                    diagnostics["guard_rate_evaluations"] = guard["rate_evaluations"]
                    diagnostics["guard_budget_limit"] = guard["budget_limit"]
                    diagnostics["guard_scenario_count"] = guard["scenario_count"]
                    diagnostics["guard_mode"] = guard_mode
            else:
                diagnostics = {
                    "candidate_count": 0,
                    "rate_evaluations": 0,
                    "budget_limit": self.cfg.evaluation.matched_candidate_budget,
                    "scenario_count": 0,
                    "unused_budget": self.cfg.evaluation.matched_candidate_budget,
                    "accepted_moves": 0,
                    "selected_p05": None,
                    "selected_user_p05": None,
                    "risk_scope": self.policy.risk_scope,
                    "selected_expected_sum_rate": None,
                    "constraint_satisfied": None,
                    "reliability_target": self.policy.reliability_target,
                    "fallback": None,
                    "decision_seconds": 0.0,
                    "backend_metadata": None,
                    "guard_mode": guard_mode,
                }
        return phases, posterior_g, posterior_h, {
            **diagnostics,
            "refreshed": refreshed,
            "trigger": trigger or "reuse",
            "posterior_uncertainty": uncertainty,
            "innovation_nis_per_dimension": nis,
            "scenario_scale": scenario_scale,
            "scenario_inflation": self.policy.scenario_inflation,
            "reuse_age": slot - self._last_refresh,
            "adaptive_guard": self.policy.adaptive_guard,
            "guard_mode": guard_mode,
            "guard_escalated": self._guard_escalated,
            "guard_anomaly_high": anomaly_high,
            "guard_anomaly_low": anomaly_low,
            "guard_anomaly_streak": self._anomaly_streak,
        }


class PeriodicController:
    """Budget-matched baseline that refreshes from nominal noisy CSI periodically."""

    def __init__(self, cfg: Config, rng: np.random.Generator, period: int = 3,
                 method: str = "risk_constrained", risk_scope: str = "pooled",
                 target: float | None = None):
        if period < 1 or method not in ("risk_constrained", "coordinate"):
            raise ValueError("invalid periodic controller configuration")
        if risk_scope not in ("pooled", "per_user"):
            raise ValueError("risk_scope must be 'pooled' or 'per_user'")
        if target is not None and (not np.isfinite(target) or not 0 <= target <= 1):
            raise ValueError("target must be in [0, 1]")
        self.cfg, self.rng, self.period, self.method, self.risk_scope = (
            cfg, rng, period, method, risk_scope)
        self.target = (cfg.evaluation.outage_threshold if target is None
                       else float(target))
        self.phases = None
        self.last_refresh = -10**9

    def reset(self):
        self.phases = None
        self.last_refresh = -10**9

    def step(self, g_hat, h_hat, slot: int):
        refresh = self.phases is None or slot - self.last_refresh >= self.period
        if refresh:
            self.phases, diagnostics = optimize_phases(
                g_hat, h_hat, self.cfg, self.rng, method=self.method,
                initial=self.phases, scenario_count=self.cfg.evaluation.channel_realizations,
                target=self.target,
                risk_scope=self.risk_scope)
            self.last_refresh = slot
            trigger = "initial" if slot == 0 else "period"
        else:
            diagnostics = {
                "candidate_count": 0, "rate_evaluations": 0,
                "budget_limit": self.cfg.evaluation.matched_candidate_budget,
                "scenario_count": 0, "unused_budget": self.cfg.evaluation.matched_candidate_budget,
                "accepted_moves": 0, "selected_p05": None,
                "selected_user_p05": None, "risk_scope": self.risk_scope,
                "reliability_target": self.target,
                "selected_expected_sum_rate": None, "constraint_satisfied": None,
                "fallback": None, "decision_seconds": 0.0, "backend_metadata": None,
            }
            trigger = "reuse"
        return np.array(self.phases, copy=True), np.asarray(g_hat), np.asarray(h_hat), {
            **diagnostics, "refreshed": refresh, "trigger": trigger,
            "posterior_uncertainty": None, "innovation_nis_per_dimension": None,
            "scenario_scale": self.cfg.system.csi_error_scale,
            "reuse_age": slot - self.last_refresh,
        }


def make_controller(kind: str, cfg: Config, rng: np.random.Generator,
                    policy: TriggerPolicy | None = None):
    if kind == "posterior_triggered":
        return PosteriorTriggeredController(cfg, rng, policy)
    if kind == "periodic_risk":
        active_policy = policy or TriggerPolicy()
        return PeriodicController(cfg, rng, period=(
                                      active_policy.baseline_refresh_period
                                      if active_policy.baseline_refresh_period is not None
                                      else active_policy.refresh_period),
                                  method="risk_constrained",
                                  risk_scope=active_policy.baseline_risk_scope,
                                  target=active_policy.baseline_reliability_target)
    if kind == "always_risk":
        return PeriodicController(cfg, rng, period=1, method="risk_constrained",
                                  risk_scope=(policy or TriggerPolicy()).baseline_risk_scope,
                                  target=(policy or TriggerPolicy()).baseline_reliability_target)
    if kind == "periodic_nominal":
        active_policy = policy or TriggerPolicy()
        return PeriodicController(cfg, rng, period=(
                                      active_policy.baseline_refresh_period
                                      if active_policy.baseline_refresh_period is not None
                                      else active_policy.refresh_period),
                                  method="coordinate", risk_scope="pooled",
                                  target=active_policy.baseline_reliability_target)
    raise ValueError(f"unknown controller kind: {kind}")
