"""Two sample test to test if two sets of samples are comming from the same distribution."""

import numpy as np
from scipy.stats import anderson_ksamp, ks_2samp

from thermal.sample._interface import _SampleInterface


def _surrogates_generator(surrogates, num_tests=100):
    if isinstance(surrogates, _SampleInterface):
        for _ in range(num_tests):
            s = surrogates.sample()
            yield s
    elif isinstance(surrogates, list):
        for s in surrogates:
            yield s
    elif isinstance(surrogates, np.ndarray):
        for i in range(surrogates.shape[1]):
            s = surrogates[:, i]
            yield s
    else:
        raise ValueError('Unknown surrogates argument type.')


def _generic_test(original, surrogates, tester, num_tests=1000):
    p_values = []
    for s in _surrogates_generator(surrogates, num_tests):
        p_values.append(tester(original, s))
    return p_values


"""
def _generic_adversarial_test(original, surrogates, tester, tester2=None, num_tests=100, num_repeats=4):
    inner_p_values = []
    cross_p_values = []
    hn = int(len(original) / 2)
    for s in _surrogates_generator(surrogates, num_tests):
        for r in range(num_repeats):
            np.random.shuffle(original)
            np.random.shuffle(s)
            inner_p_values.append(tester(original[:hn], original[hn : 2 * hn]))
            cross_p_values.append(tester(original[:hn], s[:hn]))
    if tester2 is None:
        tester2 = tester
    return tester2(inner_p_values, cross_p_values), inner_p_values, cross_p_values
"""


def _ks_test_func(a, b):
    t = ks_2samp(a, b)
    return t.pvalue


def ks_test(original, surrogates, *, num_tests=1000):
    """Run multiple two-sample Kolmogorov–Smirnov test.

    This test is used to test the similarity between the distribution of original data samples
    v.s. multiple set of generated surrogate data samples.

    Parameters
    ----------
    original : Orignal set samples
    surrogates : Sets of samples, or a fitted resampler opject that will be used to generate surrogate samples.
    num_tests : (optional) number of tests to do.

    Returns
    -------
    Array of p-values.

    """
    return _generic_test(original, surrogates, _ks_test_func, num_tests=num_tests)


def _anderson_test_sig_func(a, b):
    statistic, critical_values, significance_level = anderson_ksamp([a, b])
    return significance_level


"""
def _anderson_test_stat_func(a, b):
    statistic, critical_values, significance_level = anderson_ksamp([a, b])
    return statistic
"""


def anderson_test(original, surrogates, *, num_tests=1000):
    """Run multiple two-sample Anderson-Darling test between the original and surrogate samples.

    Tests the null hypothesis that multiple surrogate sets and a set of original samples are drawn
    from the same population without having to specify the distribution function of that population.
    For each test it returns an approximate significance level at which the null hypothesis for the
    provided samples can be rejected. The value is floored / capped at 0.1% / 25%.

    Parameters
    ----------
    original : Orignal set samples
    surrogates : Sets of samples, or a fitted resampler opject.
    num_tests : (optional) number of tests to do.

    Returns
    -------
    Array of significance level values.

    """
    return _generic_test(original, surrogates, _anderson_test_sig_func, num_tests)
