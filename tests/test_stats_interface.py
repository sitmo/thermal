"""Pytest module to test stats interfaces."""
import numpy as np
import pytest

from thermal import SampleHist, anderson_test, ks_test


def test_ks_test_interface():
    """Testing the ks_test interface."""
    x = np.random.normal(size=10)

    s = np.random.normal(size=(10, 4))
    ans = anderson_test(x, s)

    s = [np.random.normal(size=10), np.random.normal(size=10), np.random.normal(size=10)]
    ans = anderson_test(x, s)

    eng = SampleHist()
    eng.fit(x)
    ans = anderson_test(x, eng)

    return ans


def test_anderson_test_interface():
    """Testing the ks_test interface."""
    x = np.random.normal(size=10)

    s = np.random.normal(size=(10, 4))
    ans = ks_test(x, s)

    s = [np.random.normal(size=10), np.random.normal(size=10), np.random.normal(size=10)]
    ans = ks_test(x, s)

    eng = SampleHist()
    eng.fit(x)
    ans = ks_test(x, eng)

    with pytest.raises(Exception):
        ans = ks_test(x, "banana")

    return ans
