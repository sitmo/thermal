"""Pytest module to test interfaces."""

import numpy as np
import pytest

import thermal as th


def test_fitting_sampling():
    """Testing samplers."""
    x = np.random.normal(size=100)

    resamplers = [
        th.SampleHist(replace=False),
        th.SampleHist(replace=True),
        th.SampleKde(),
        th.SampleKde(cv=2),
        th.SampleGmm(bayesian=False),
        th.SampleGmm(bayesian=False, prune=False),
        th.SampleGmm(bayesian=True),
        th.Sample(amplitude='gmm', n_components=5),
        th.Sample(amplitude='kde'),
        th.Sample(amplitude='hist'),
    ]

    ans = []
    for eng in resamplers:
        eng.fit(x)
        ans.append(eng.sample())
        ans.append(eng.sample(5))
        ans.append(eng.sample(200))
        ans.append(eng.fit(x).sample())
        ans.append(str(eng))
        ans.append(eng.n_samples_)

    with pytest.raises(Exception):
        th.SampleKde().sample()

    with pytest.raises(Exception):
        th.Sample(amplitude='banaan')

    return ans
