from __future__ import annotations

import csv
import hashlib
import itertools
import json
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .channel import sample_channel, estimate_trajectory
from .config import Config, load_config
from .control import optimize_phases
from .cuda_backend import evaluate_batch


def source_hash() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _cfg(base: Config, error: float, elements: int, users: int) -> Config:
    return replace(base, system=replace(base.system, csi_error_scale=error,
                                        ris_elements=elements, users=users))


def _evaluate(cfg: Config, method: str, seed: int, episodes: int) -> dict:
    if episodes < 1 or cfg.evaluation.trajectory_steps < 1:
        raise ValueError("episodes and trajectory_steps must be positive")
    if not 0 < cfg.evaluation.outage_quantile < 1:
        raise ValueError("outage_quantile must be in (0, 1)")
    # Controller RNG consumption cannot alter any subsequent channel or CSI.
    channel_rng = np.random.default_rng(seed)
    raw = []
    details = []
    stream_hash = hashlib.sha256()
    for episode in range(episodes):
        initial_channel = sample_channel(
            cfg.system.bs_antennas, cfg.system.ris_elements, cfg.system.users,
            cfg.system.rician_k, cfg.system.csi_error_scale, channel_rng)
        trajectory = estimate_trajectory(
            initial_channel, cfg.system.csi_error_scale, cfg.system.temporal_rho,
            cfg.evaluation.trajectory_steps, channel_rng)
        phases = None
        episode_rates = []
        for slot, ch in enumerate(trajectory):
            for array in (ch.truth_g, ch.truth_h, ch.g_hat, ch.h_hat):
                stream_hash.update(np.ascontiguousarray(array).tobytes())
            rng = np.random.default_rng(np.random.SeedSequence([seed, episode, slot, 173]))
            phases, diagnostics = optimize_phases(
                ch.g_hat, ch.h_hat, cfg, rng, method, initial=phases,
                scenario_count=cfg.evaluation.channel_realizations,
                target=cfg.evaluation.outage_threshold)
            _, rates, truth_meta = evaluate_batch(
                ch.truth_h[None], ch.truth_g[None], phases[None],
                cfg.system.transmit_power, cfg.system.noise_power, cfg.evaluation.backend,
                ch.h_hat[None], ch.g_hat[None])
            if cfg.evaluation.backend == "cupy":
                rates = rates.get()
            rates = np.asarray(rates)[0]
            episode_rates.append(rates.tolist())
            details.append({"episode": episode, "slot": slot, **diagnostics,
                            "phases": phases.tolist(), "truth_backend_metadata": truth_meta,
                            "realized_outage": float(np.mean(rates < cfg.evaluation.outage_threshold)),
                            "g_relative_error": float(np.linalg.norm(ch.g_hat - ch.truth_g) / np.linalg.norm(ch.truth_g)),
                            "h_relative_error": float(np.linalg.norm(ch.h_hat - ch.truth_h) / np.linalg.norm(ch.truth_h))})
        raw.append(episode_rates)
    rates = np.asarray(raw)
    constraints = [d["constraint_satisfied"] for d in details if d["constraint_satisfied"] is not None]
    return {
        "sum_rate": float(rates.sum(axis=-1).mean()),
        "pooled_user_p05_rate": float(np.quantile(rates, cfg.evaluation.outage_quantile)),
        "outage_probability": float(np.mean(rates < cfg.evaluation.outage_threshold)),
        "quantile_level": cfg.evaluation.outage_quantile,
        "decision_candidate_count": float(np.mean([d["candidate_count"] for d in details])),
        "rate_evaluations_per_decision": float(np.mean([d["rate_evaluations"] for d in details])),
        "decision_seconds_mean": float(np.mean([d["decision_seconds"] for d in details])),
        "constraint_satisfaction_rate": float(np.mean(constraints)) if constraints else None,
        "episode_rates": raw,
        "episode_details": details,
        "channel_stream_sha256": stream_hash.hexdigest(),
    }


def summarize_seed_blocks(rows: list[dict], draws: int = 1000) -> list[dict]:
    """Bootstrap independent seeds; retain all correlated slots/users in a block."""
    groups = {}
    keys = ("error", "temporal_rho", "elements", "users", "method")
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    summaries = []
    for group_id, (key, block_rows) in enumerate(sorted(groups.items())):
        block_rows = sorted(block_rows, key=lambda r: r["seed"])
        if len({r["seed"] for r in block_rows}) != len(block_rows):
            raise ValueError("duplicate seed blocks")
        blocks = np.asarray([r["episode_rates"] for r in block_rows])
        threshold = block_rows[0]["outage_threshold"]
        quantile = block_rows[0]["quantile_level"]

        def metrics(x):
            return (float(x.sum(axis=-1).mean()), float(np.quantile(x, quantile)),
                    float(np.mean(x < threshold)))

        point = metrics(blocks)
        ci = None
        if len(block_rows) >= 2:
            rng = np.random.default_rng(700001 + group_id)
            resamples = np.asarray([metrics(blocks[rng.integers(len(blocks), size=len(blocks))])
                                   for _ in range(draws)])
            ci = np.quantile(resamples, [0.025, 0.975], axis=0)
        result = dict(zip(keys, key))
        result.update({"independent_seeds": len(block_rows),
                       "slots_per_seed": int(np.prod(blocks.shape[1:-1])),
                       "ci_unit": "independent_seed_block", "bootstrap_draws": draws,
                       "mean_decision_seconds": float(np.mean([r["decision_seconds_mean"] for r in block_rows]))})
        for i, metric in enumerate(("sum_rate", "pooled_user_p05_rate", "outage_probability")):
            result[metric] = point[i]
            result[metric + "_ci_low"] = float(ci[0, i]) if ci is not None else None
            result[metric + "_ci_high"] = float(ci[1, i]) if ci is not None else None
        summaries.append(result)
    return summaries


def run_sweep(config_path: str, output_dir: str = "results/sweep", seed: int = 7,
              episodes: int = 5, seeds: list[int] | None = None,
              train_errors: tuple[float, ...] = (0.0, 0.04, 0.08, 0.12),
              test_errors: tuple[float, ...] = (0.20,),
              rhos: tuple[float, ...] = (0.0, 0.5, 0.8, 0.95),
              backend: str | None = None, elements=(16, 32, 64), users=(2, 4, 6),
              methods=("coordinate", "scenario_mean", "risk_constrained")) -> list[dict]:
    base = load_config(config_path)
    if backend is not None:
        base = replace(base, evaluation=replace(base.evaluation, backend=backend))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "sweep.json").exists() or (out / "raw.jsonl").exists():
        raise FileExistsError("select a new output directory to preserve previous results")
    seed_values = seeds if seeds is not None else [seed]
    if not seed_values or len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must be nonempty and distinct")
    config_hash = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                           stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision = None
    code_hash = source_hash()
    rows = []
    errors = tuple(dict.fromkeys((*train_errors, *test_errors)))
    conditions = list(itertools.product(errors, rhos, elements, users, methods, seed_values))
    (out / "manifest.json").write_text(json.dumps({
        "purpose": "development; historical error scales have already been inspected",
        "source_sha256": code_hash, "git_revision": revision,
        "config_sha256": config_hash, "base_configuration": asdict(base),
        "planned_rows": len(conditions), "seeds": seed_values,
        "errors": errors, "rhos": rhos, "elements": elements, "users": users,
        "methods": methods, "episodes": episodes,
    }, indent=2), encoding="utf-8")
    with (out / "raw.jsonl").open("w", encoding="utf-8") as raw_file:
        for n, (error, rho, n_elements, n_users, method, run_seed) in enumerate(conditions, 1):
            cfg = _cfg(base, error, n_elements, n_users)
            cfg = replace(cfg, system=replace(cfg.system, temporal_rho=rho))
            effective = asdict(cfg)
            effective_hash = hashlib.sha256(json.dumps(effective, sort_keys=True).encode()).hexdigest()
            started = time.perf_counter()
            metrics = _evaluate(cfg, method, run_seed, episodes)
            row = {"error": error, "elements": n_elements, "users": n_users,
                   "method": method, "seed": run_seed, "episodes": episodes,
                   "error_split": "observed_reference" if error in test_errors else "development",
                   "backend": cfg.evaluation.backend, "trajectory_steps": cfg.evaluation.trajectory_steps,
                   "outage_threshold": cfg.evaluation.outage_threshold,
                   "budget_limit": cfg.evaluation.matched_candidate_budget,
                   "config_sha256": config_hash, "effective_config_sha256": effective_hash,
                   "effective_configuration": effective, "git_revision": revision,
                   "source_sha256": code_hash, "temporal_rho": rho, **metrics,
                   "runtime_seconds": time.perf_counter() - started}
            rows.append(row)
            raw_file.write(json.dumps(row) + "\n")
            raw_file.flush()
            if n % 10 == 0 or n == len(conditions):
                print(f"sweep progress {n}/{len(conditions)}", flush=True)
    summaries = summarize_seed_blocks(rows)
    (out / "sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    with (out / "sweep.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    return rows
