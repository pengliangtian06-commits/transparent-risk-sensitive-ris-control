from __future__ import annotations

import argparse
import json
import csv
import hashlib
from pathlib import Path
import time
import numpy as np

from .baselines import (coordinate_search, random_phases, robust_projected_search,
                        uncertainty_aware_phases, risk_constrained_phases)
from .channel import sample_channel
from .config import load_config
from .metrics import evaluate_phases
from .learning import evaluate_policy, train_teacher_policy, train_policy_split
from .sweep import run_sweep


def _config_hash(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _episode(cfg, rng, method):
    ch = sample_channel(cfg.system.bs_antennas, cfg.system.ris_elements, cfg.system.users,
                        cfg.system.rician_k, cfg.system.csi_error_scale, rng)
    if method == "random": phases = random_phases(cfg.system.ris_elements, rng)
    elif method == "coordinate": phases = coordinate_search(ch, cfg, rng)
    elif method == "robust": phases = robust_projected_search(ch, cfg, rng)
    elif method == "uncertainty_aware": phases = uncertainty_aware_phases(ch, cfg, rng)
    elif method == "risk_constrained": phases = risk_constrained_phases(ch, cfg, rng)[0]
    else: raise ValueError(method)
    return evaluate_phases(ch, phases, cfg.system.transmit_power, cfg.system.noise_power)


def run_benchmark(cfg, seed: int, episodes: int) -> dict:
    channel_rng = np.random.default_rng(seed)
    channels = [sample_channel(cfg.system.bs_antennas, cfg.system.ris_elements, cfg.system.users,
                               cfg.system.rician_k, cfg.system.csi_error_scale, channel_rng)
                for _ in range(episodes)]
    results = {}
    for method in ("random", "coordinate", "robust", "uncertainty_aware", "risk_constrained"):
        rows = []
        for idx, ch in enumerate(channels):
            rng = np.random.default_rng(seed * 1_000_003 + idx * 101 + sum(method.encode("utf-8")))
            if method == "random": phases = random_phases(ch.g.shape[0], rng)
            elif method == "coordinate": phases = coordinate_search(ch, cfg, rng)
            elif method == "robust": phases = robust_projected_search(ch, cfg, rng)
            elif method == "uncertainty_aware": phases = uncertainty_aware_phases(ch, cfg, rng)
            else: phases = risk_constrained_phases(ch, cfg, rng)[0]
            rows.append(evaluate_phases(ch, phases, cfg.system.transmit_power, cfg.system.noise_power))
        results[method] = {"sum_rate_mean": float(np.mean([r["sum_rate"] for r in rows])),
                           "sum_rate_std": float(np.std([r["sum_rate"] for r in rows])),
                           "p05_user_rate_mean": float(np.mean([r["p05_user_rate"] for r in rows]))}
    return {"seed": seed, "episodes": episodes, "methods": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["smoke", "benchmark", "train", "campaign", "sweep"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--backend", choices=["numpy", "cupy"], default=None)
    parser.add_argument("--output-dir", default="results/sweep")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.backend is not None:
        from dataclasses import replace
        cfg = replace(cfg, evaluation=replace(cfg.evaluation, backend=args.backend))
    Path("results").mkdir(exist_ok=True)
    if args.command == "sweep":
        rows = run_sweep(args.config, seed=args.seed, episodes=args.episodes or 5,
                         seeds=args.seeds, backend=args.backend, output_dir=args.output_dir)
        print(json.dumps({"status": "ok", "rows": len(rows), "output": args.output_dir}, indent=2))
        return
    start = time.perf_counter()
    if args.command == "smoke":
        payload = {"status": "ok", "benchmark": run_benchmark(cfg, args.seed, 1)}
        out = Path("results/smoke.json")
    elif args.command in ("benchmark", "campaign"):
        payload = run_benchmark(cfg, args.seed, args.episodes or cfg.evaluation.episodes)
        out = Path("results/benchmark.json" if args.command == "benchmark" else "results/campaign.json")
    else:
        train_n = max(4, min(args.steps or cfg.training.steps, 64))
        payload = {"status": "trained", "steps": args.steps or cfg.training.steps,
                   "seed": args.seed,
                   "test": train_policy_split(cfg, args.seed, args.seed + 1,
                                               train_n, cfg.evaluation.episodes,
                                               scenarios=3, candidates=8)}
        out = Path("results/train_smoke.json")
    payload["config_sha256"] = _config_hash(args.config)
    payload["config_path"] = args.config
    payload["runtime_seconds"] = time.perf_counter() - start
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.command in ("benchmark", "campaign"):
        rows = [{"method": method, **metrics} for method, metrics in payload["methods"].items()]
        csv_path = out.with_suffix(".csv")
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
