import numpy as np
import pytest

from rislab.cuda_backend import available, batch_effective_channels, batch_sum_rates_numpy, evaluate_batch


def test_cuda_backend_optional_contract():
    if not available():
        return
    h = np.ones((2, 3, 4), dtype=np.complex64)
    g = np.ones((2, 4, 2), dtype=np.complex64)
    phases = np.zeros((2, 4), dtype=np.float32)
    out = batch_effective_channels(h, g, phases)
    assert out.shape == (2, 3, 2)


def test_numpy_batch_rate_reference_contract():
    rng = np.random.default_rng(3)
    batch, users, elements, antennas = 4, 2, 5, 3
    h = rng.normal(size=(batch, users, elements)) + 1j * rng.normal(size=(batch, users, elements))
    g = rng.normal(size=(batch, elements, antennas)) + 1j * rng.normal(size=(batch, elements, antennas))
    phases = rng.uniform(-np.pi, np.pi, size=(batch, elements))
    sums, rates = batch_sum_rates_numpy(h, g, phases, 1.0, 0.1)
    assert sums.shape == (batch,)
    assert rates.shape == (batch, users)
    assert np.all(np.isfinite(rates))
    np.testing.assert_allclose(sums, rates.sum(axis=1), rtol=1e-12, atol=1e-12)
    sums2, rates2, meta = evaluate_batch(h, g, phases, 1.0, 0.1)
    np.testing.assert_allclose(sums, sums2)
    np.testing.assert_allclose(rates, rates2)
    assert meta["backend"] == "numpy"


def test_cuda_matches_reference_with_fixed_estimated_beams():
    pytest.importorskip("cupy")
    from rislab.cuda_backend import compare_backends
    rng = np.random.default_rng(29)
    h = rng.normal(size=(12, 3, 8)) + 1j * rng.normal(size=(12, 3, 8))
    g = rng.normal(size=(12, 8, 4)) + 1j * rng.normal(size=(12, 8, 4))
    phases = rng.uniform(-np.pi, np.pi, (12, 8))
    beam_h = h + 0.2 * rng.normal(size=h.shape)
    beam_g = g + 0.2 * rng.normal(size=g.shape)
    result = compare_backends(h, g, phases, 1, 0.01, beam_h, beam_g)
    assert result["max_sum_abs_error"] < 1e-10
    assert result["max_rate_abs_error"] < 1e-10
