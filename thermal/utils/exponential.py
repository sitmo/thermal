"""Exponential distribution routines."""
import numpy as np
from scipy.stats import expon


def exponential_pdf(x, lambda_):
    """Probability density function (pdf) of the Exponential ddistribution.

    Parameters
    ----------
    x : array, number
        Locations where to evaluate the pdf.
    lambda_ : number
         rate parameter of the exponential distribution.

    Returns
    -------
        pdfs.
    """
    return expon.pdf(x, scale=1.0 / lambda_)


def exponential_fit_from_median(x, axis=None):
    """Estimate the rate parameter lambda using the median of a set of samples.

    Parameters
    ----------
    x : array
    axis : int (None)

    Returns
    -------
        lambda : number, array
            rate parameter estimates.
    """
    lambda_ = np.log(2) / np.median(x, axis=axis)
    return lambda_


def exponential_fit_from_mean(x, axis=None):
    """Estimate the rate parameter lambda using the mean of a set of samples.

    Parameters
    ----------
    x : array
    axis : int (None)

    Returns
    -------
        lambda : number, array
            rate parameter estimates.
    """
    lambda_ = 1.0 / np.mean(x, axis=axis)
    return lambda_


def exponential_fit_from_quantile(x, q=0.95, axis=None):
    """Estimate the rate parameter lambda using the quantile of a set of samples.

    Parameters
    ----------
    x : array
    q : number [0..1]. Quantile.
    axis : int (None)

    Returns
    -------
        lambda : number, array
            rate parameter estimates.
    """
    level = np.quantile(x, q, axis=axis)
    lambda_ = -np.log(1 - q) / level
    return lambda_


def exponential_kmax_icdf(lambda_, k, p=0.95):
    """Inverse Cumulative distribution function (inv CDF) of the maximum of k exponential distributed variates.

    Computes the level that the max of k exponential distributed variates will
    breach with probability p.

    Parameters
    ----------
    lambda_ : number > 0, rate parameter of the exponential distribution.
    k : int >= 1, the number of variantes of which we compute the max.
    p : number, 0 < level < 1, the p-value.

    Returns
    -------
        level
    """
    p1 = p ** (1 / k)
    scale = 1.0 / lambda_
    level = expon.ppf(p1, scale=scale)
    return level
