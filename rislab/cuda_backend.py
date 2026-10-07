"""Optional CuPy backend for batched RIS phase evaluation on Jetson CUDA."""
from __future__ import annotations

import numpy as np
import time


def available() -> bool:
    try:
        import cupy  # noqa: F401
        return True
    except Exception:
        return False


def benchmark_vector_add(size: int = 1_000_000) -> dict:
    """Run a real CUDA operation and return auditable device metadata."""
    import cupy as cp
    x = cp.ones(size, dtype=cp.float32)
    z = x + x
    cp.cuda.Stream.null.synchronize()
    return {"backend": "cupy", "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "compute_capability": cp.cuda.Device().compute_capability,
            "size": size, "mean": float(cp.mean(z).get())}


def batch_effective_channels(h, g, phases):
    """Evaluate cascaded channels for batches; arrays are copied to GPU."""
    import cupy as cp
    h_gpu, g_gpu = cp.asarray(h), cp.asarray(g)
    theta = cp.exp(1j * cp.asarray(phases))
    return cp.einsum("bnr,brm->bnm", h_gpu * theta[:, None, :], g_gpu)


def batch_sum_rates(h, g, phases, transmit_power: float, noise_power: float,
                    beam_h=None, beam_g=None):
    """Compute equal-power matched-filter rates for a batch on CUDA."""
    import cupy as cp
    eff = batch_effective_channels(h, g, phases)
    nominal = eff if beam_h is None else batch_effective_channels(beam_h, beam_g, phases)
    beams = cp.conj(cp.transpose(nominal, (0, 2, 1)))
    norms = cp.linalg.norm(beams, axis=1, keepdims=True) + 1e-12
    beams = beams / norms * cp.sqrt(transmit_power / eff.shape[1])
    gains = cp.abs(cp.matmul(eff, beams)) ** 2
    desired = cp.diagonal(gains, axis1=1, axis2=2)
    interference = cp.sum(gains, axis=2) - desired
    rates = cp.log2(1 + desired / (noise_power + interference + 1e-12))
    return cp.sum(rates, axis=1), rates


def batch_sum_rates_numpy(h, g, phases, transmit_power: float, noise_power: float,
                          beam_h=None, beam_g=None):
    """Reference implementation matching :func:`batch_sum_rates` semantics."""
    h = np.asarray(h)
    g = np.asarray(g)
    phases = np.asarray(phases)
    theta = np.exp(1j * phases)
    eff = np.einsum("bnr,brm->bnm", h * theta[:, None, :], g)
    nominal = eff if beam_h is None else np.einsum(
        "bnr,brm->bnm", np.asarray(beam_h) * theta[:, None, :], np.asarray(beam_g))
    beams = np.conjugate(np.transpose(nominal, (0, 2, 1)))
    norms = np.linalg.norm(beams, axis=1, keepdims=True) + 1e-12
    beams = beams / norms * np.sqrt(transmit_power / eff.shape[1])
    gains = np.abs(np.matmul(eff, beams)) ** 2
    desired = np.diagonal(gains, axis1=1, axis2=2)
    interference = np.sum(gains, axis=2) - desired
    rates = np.log2(1 + desired / (noise_power + interference + 1e-12))
    return np.sum(rates, axis=1), rates


def evaluate_batch(h, g, phases, transmit_power: float, noise_power: float,
                   backend: str = "numpy", beam_h=None, beam_g=None):
    """Evaluate a phase batch and return backend metadata.

    The CuPy path is deliberately opt-in so CPU reference runs remain
    deterministic and available on machines without CUDA.
    """
    if (beam_h is None) != (beam_g is None):
        raise ValueError("provide both nominal beam channels")
    wall_start = time.perf_counter()
    if backend == "cupy":
        import cupy as cp
        start = cp.cuda.Event(); end = cp.cuda.Event()
        start.record()
        sums, rates = batch_sum_rates(h, g, phases, transmit_power, noise_power, beam_h, beam_g)
        end.record(); end.synchronize()
        return sums, rates, {"backend": "cupy", "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
                             "compute_capability": cp.cuda.Device().compute_capability,
                             "elapsed_ms": float(cp.cuda.get_elapsed_time(start, end)),
                             "wall_ms": (time.perf_counter() - wall_start) * 1000}
    if backend != "numpy":
        raise ValueError("backend must be numpy or cupy")
    sums, rates = batch_sum_rates_numpy(h, g, phases, transmit_power, noise_power, beam_h, beam_g)
    return sums, rates, {"backend": "numpy", "elapsed_ms": (time.perf_counter() - wall_start) * 1000}


def compare_backends(h, g, phases, transmit_power: float, noise_power: float,
                      beam_h=None, beam_g=None) -> dict:
    """Compare CPU reference and CUDA outputs on identical inputs."""
    cpu_sum, cpu_rates, cpu_meta = evaluate_batch(h, g, phases, transmit_power, noise_power,
                                                "numpy", beam_h, beam_g)
    gpu_sum, gpu_rates, meta = evaluate_batch(h, g, phases, transmit_power, noise_power,
                                             "cupy", beam_h, beam_g)
    gpu_sum = gpu_sum.get()
    gpu_rates = gpu_rates.get()
    return {"max_sum_abs_error": float(np.max(np.abs(cpu_sum - gpu_sum))),
            "max_rate_abs_error": float(np.max(np.abs(cpu_rates - gpu_rates))),
            "cpu_elapsed_ms": cpu_meta["elapsed_ms"], "gpu_metadata": meta}
