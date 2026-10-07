"""Quantized RIS pilot construction and linear pilot observation utilities."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _next_power_of_two(value: int) -> int:
    if value < 1:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


def _hadamard(order: int) -> np.ndarray:
    """Return a Sylvester Hadamard matrix with entries in {-1, +1}."""
    matrix = np.array([[1.0]])
    while matrix.shape[0] < order:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix


@dataclass(frozen=True)
class PilotDesign:
    ris_elements: int
    phase_bits: int
    pilot_symbols: int
    phases: np.ndarray  # P x N, in the configured hardware alphabet
    operator: np.ndarray  # P x (N+1): direct column followed by RIS columns
    rank: int
    condition_number: float
    overhead_symbols: int


def build_hadamard_pilots(ris_elements: int, phase_bits: int,
                          include_direct: bool = True) -> PilotDesign:
    """Build a full-column-rank pilot operator implementable by b-bit RIS.

    The first operator column is the direct-link/all-ones column. Remaining
    columns are RIS phase patterns. Hadamard signs map to phases 0 and pi, so
    the construction is valid for any phase_bits >= 1.
    """
    if ris_elements < 1 or phase_bits < 1:
        raise ValueError("ris_elements and phase_bits must be positive")
    columns = ris_elements + (1 if include_direct else 0)
    pilot_symbols = _next_power_of_two(columns)
    hadamard = _hadamard(pilot_symbols)
    if include_direct:
        operator = hadamard[:, :columns].astype(np.complex128)
        ris_signs = operator[:, 1:]
    else:
        operator = hadamard[:, :ris_elements].astype(np.complex128)
        ris_signs = operator
    phases = np.where(np.real(ris_signs) < 0, np.pi, 0.0)
    rank = int(np.linalg.matrix_rank(operator))
    if rank != columns:
        raise ValueError("pilot operator is rank deficient")
    condition_number = float(np.linalg.cond(operator))
    return PilotDesign(
        ris_elements=ris_elements,
        phase_bits=phase_bits,
        pilot_symbols=pilot_symbols,
        phases=phases,
        operator=operator,
        rank=rank,
        condition_number=condition_number,
        overhead_symbols=pilot_symbols,
    )


def observe_pilots(design: PilotDesign, coefficients: np.ndarray,
                   noise_variance: float, rng: np.random.Generator) -> np.ndarray:
    """Generate Y=A X+W for a coefficient matrix X of shape (N+1, M)."""
    x = np.asarray(coefficients, dtype=np.complex128)
    expected = design.operator.shape[1]
    if x.ndim != 2 or x.shape[0] != expected:
        raise ValueError("coefficients must have shape (N+1, BS antennas)")
    if not np.isfinite(noise_variance) or noise_variance < 0:
        raise ValueError("noise_variance must be finite and nonnegative")
    noise = (rng.standard_normal((design.pilot_symbols, x.shape[1]))
             + 1j * rng.standard_normal((design.pilot_symbols, x.shape[1])))
    noise *= np.sqrt(noise_variance / 2.0)
    return design.operator @ x + noise


def least_squares_observation(design: PilotDesign, measurements: np.ndarray,
                              noise_variance: float) -> tuple[np.ndarray, np.ndarray]:
    """Return X-hat and per-antenna coefficient covariance."""
    y = np.asarray(measurements, dtype=np.complex128)
    if y.ndim != 2 or y.shape[0] != design.pilot_symbols:
        raise ValueError("measurements must have shape (pilot_symbols, BS antennas)")
    if not np.isfinite(noise_variance) or noise_variance < 0:
        raise ValueError("noise_variance must be finite and nonnegative")
    pseudo_inverse = np.linalg.pinv(design.operator)
    estimate = pseudo_inverse @ y
    covariance = noise_variance * (pseudo_inverse @ pseudo_inverse.conj().T)
    covariance = (covariance + covariance.conj().T) / 2.0
    return estimate, covariance
