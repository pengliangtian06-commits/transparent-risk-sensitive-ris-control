import numpy as np
import pytest
from scipy.stats import binom

from rislab.risk import maximum_accepted_failures, outage_upper_bound, single_weak_user_diagnostic


def test_zero_event_bound_is_nonzero_and_requires_sufficient_samples():
    delta = 0.05 / 4
    assert outage_upper_bound(0, 64, delta) == pytest.approx(1 - delta ** (1 / 64))
    assert outage_upper_bound(0, 4, delta) > 0.6
    assert maximum_accepted_failures(64, 0.1, delta) == 1
    assert maximum_accepted_failures(4, 0.1, delta) == -1


@pytest.mark.parametrize("samples", [8, 64, 512])
def test_one_sided_coverage_by_exact_binomial_enumeration(samples):
    counts = np.arange(samples + 1)
    delta = 0.0125
    bounds = outage_upper_bound(counts, samples, delta)
    assert np.all(np.diff(bounds) >= 0)
    for true_p in (0, 0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.9, 1):
        noncoverage = binom.pmf(counts, samples, true_p)[bounds < true_p].sum()
        assert noncoverage <= delta + 1e-12


def test_refusal_cannot_be_removed_from_service_outage_accounting():
    row = single_weak_user_diagnostic(0.01, 64, 4, 0.1, 0.05)
    assert row["admission_probability"] == pytest.approx(binom.pmf(0, 64, 0.01) + binom.pmf(1, 64, 0.01))
    assert row["all_slot_service_outage"] > 0.14
    assert not row["service_outage_target_met"]
    adequate = single_weak_user_diagnostic(0.01, 512, 4, 0.1, 0.05)
    assert adequate["service_outage_target_met"]
