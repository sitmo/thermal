"""Hurst exponent analysis routines."""

from math import gamma

import numpy as np


def hurst(x, method=2):
    """Compute the Hurst Exponent of the time series.

    Parameters
    ----------
    x : array
        time series
    method : int
        Hust exponent calculation method {0,1,2,3}, default=2.

    Returns
    -------
    h_est : float
        Hurst exponent
    lengths : array
        time scale lengths
    rescaled_ranges : array
        rescale ranges
    base_stats : array
        Theoretical white noise R/S statistic
    """
    lengths = []
    rescaled_ranges = []
    base_stats = []
    samples = []

    n = 4  # pattern length

    while True:
        num_patterns = int(len(x) / n)
        if num_patterns < 2:
            break

        # reshape into [num_patterns, pattern_length]
        patterns = x[: num_patterns * n].reshape(num_patterns, n).copy()

        # subtract sample means from each sample
        patterns -= np.mean(patterns, axis=1, keepdims=True)

        # compute the std estimate for each sample
        pattern_stds = (np.sum(patterns**2, axis=1, keepdims=True) / (n - 1)) ** 0.5

        # compute sample cumulatives
        patterns = np.cumsum(patterns, axis=1)

        # compute the min-max ranges of the cumulatives
        pattern_ranges = np.max(patterns, axis=1, keepdims=True) - np.min(patterns, axis=1, keepdims=True)

        # average rescales ranges
        average_rescaled_ranges = np.mean(pattern_ranges / pattern_stds)

        # correction by Anis and Lloyd (1976); Peters (1994)
        # https://arxiv.org/pdf/1805.08931.pdf
        i = np.arange(1, n)  # 1, 2, ... n-1
        ers = np.sum(np.sqrt((n - i) / i))
        if n <= 340:
            ers *= gamma((n - 1) / 2) / np.sqrt(np.pi) / gamma(n / 2)
        else:
            ers *= 1 / np.sqrt(np.pi * n / 2)
        # ers *= (n-0.5) / n #  additional correction by Peters (1994) .. doesn not seem to work, makes it worse

        # store results
        lengths.append(n)
        rescaled_ranges.append(average_rescaled_ranges)
        base_stats.append(ers)
        samples.append(num_patterns)

        n *= 2

    lengths = np.array(lengths)
    rescaled_ranges = np.array(rescaled_ranges)
    base_stats = np.array(base_stats)

    # Calculate Hurst exponent based on slope
    h_est = 0.5

    if method == 0:
        # Naive, Hurst
        log_n = np.log(lengths)
        log_h = np.log(rescaled_ranges)
        reg = np.polyfit(log_n, log_h, 1)
        h_est = reg[0]
    if method == 1:
        # https://w3.ual.es/personal/jetrini/objetos/HURST.pdf
        log_n = np.log(lengths)
        log_h = np.log(rescaled_ranges) - np.log(base_stats) + np.log(lengths) / 2
        reg = np.polyfit(log_n, log_h, 1)
        h_est = reg[0]
    elif method == 2:
        # https://arxiv.org/pdf/1805.08931.pdf
        log_n = np.log(lengths)
        rsal = rescaled_ranges - base_stats + np.sqrt(0.5 * np.pi * lengths)
        log_h = np.log(rsal)
        reg = np.polyfit(log_n, log_h, 1)
        h_est = reg[0]
    elif method == 3:
        # https://en.wikipedia.org/wiki/Hurst_exponent
        # doesn't seem to work well
        dev = rescaled_ranges - base_stats
        reg = np.polyfit(np.array(lengths), dev, 1)
        h_est = 0.5 + reg[0]

    return h_est, lengths, rescaled_ranges, base_stats
