"""Autocorrelation."""
import numpy as np


def autocorr(x):
    """Autocorrelation.

    Parameters
    ----------
    x : array
        Array of value

    Returns
    -------
    Vector of autocorreltions.
    """
    mean = x.mean()
    var = np.var(x)
    xp = x - mean
    corr = np.correlate(xp, xp, 'full')[len(x) - 1 :] / var / len(x)

    return corr
