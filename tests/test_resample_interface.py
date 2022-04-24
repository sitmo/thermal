"""Pytest module to test interfaces."""
import numpy as np

from thermal import ResampleGmm, ResampleHist, ResampleKde


def test_resample_gmm_interface():
    """Testing the ResampleGmm interface."""
    x = np.random.normal(size=10)

    eng = ResampleGmm()
    eng.fit(x)
    y = eng.resample()
    y = eng.resample(5)

    eng = ResampleGmm(3)
    eng.fit(x)
    y = eng.resample()
    y = eng.resample(5)

    eng = ResampleGmm().fit(x)
    y = eng.resample()
    y = eng.resample(5)

    y = ResampleGmm().fit(x).resample()
    y = ResampleGmm(3).fit(x).resample()
    y = ResampleGmm().fit(x).resample(5)
    return y


def test_resample_hist_interface():
    """Testing the ResampleHist inerface."""
    x = np.random.normal(size=10)

    eng = ResampleHist()
    eng.fit(x)
    y = eng.resample(replace=False)
    y = eng.resample(5)
    y = eng.resample(5, replace=False)

    eng = ResampleHist().fit(x)
    y = eng.resample()

    y = ResampleGmm().fit(x).resample()
    return y


def test_resample_kde_interface():
    """Testing the ResampleKde interface."""
    x = np.random.normal(size=10)

    eng = ResampleKde()
    eng.fit(x)
    y = eng.resample()
    y = eng.resample(5)

    eng = ResampleKde().fit(x)
    y = eng.resample()

    y = ResampleKde().fit(x).resample()
    return y
