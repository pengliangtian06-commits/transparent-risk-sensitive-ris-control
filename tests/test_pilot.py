import numpy as np
import pytest

from rislab.pilot import build_hadamard_pilots, least_squares_observation, observe_pilots


@pytest.mark.parametrize("elements", [1, 3, 16, 32, 64])
def test_hadamard_pilots_are_quantized_and_full_rank(elements):
    design = build_hadamard_pilots(elements, phase_bits=1)
    assert design.pilot_symbols >= elements + 1
    assert design.rank == elements + 1
    assert design.condition_number == pytest.approx(1.0)
    assert np.all(np.isin(design.phases, [0.0, np.pi]))
    assert np.allclose(design.operator[:, 0], 1.0)


def test_ls_covariance_and_recovery_without_noise():
    design = build_hadamard_pilots(4, phase_bits=2)
    rng = np.random.default_rng(19)
    coefficients = rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3))
    observations = observe_pilots(design, coefficients, 0.0, rng)
    estimate, covariance = least_squares_observation(design, observations, 0.0)
    assert np.allclose(estimate, coefficients)
    assert np.allclose(covariance, 0.0)


def test_pilot_input_validation():
    with pytest.raises(ValueError):
        build_hadamard_pilots(4, phase_bits=0)
    design = build_hadamard_pilots(4, phase_bits=1)
    with pytest.raises(ValueError):
        observe_pilots(design, np.zeros((4, 2)), 0.1, np.random.default_rng(1))
