import numpy as np
from scipy import signal
from scipy.stats import expon

from thermal.utils.exponential import exponential_fit_from_median


def spectral_signal_noise_split(x, *, deg=1, signal_prob=0.95, tol=1e-4, max_itt=20, hard=False, window='bh'):
    freq = np.fft.rfftfreq(len(x))
    if window == 'bh':
        window_ = signal.blackmanharris(len(x))
        x_fft = np.fft.rfft(x * window_)
    else:
        x_fft = np.fft.rfft(x)

    # we can only handle positive freqs
    f_ = freq[freq > 0]
    x_ = np.abs(x_fft[freq > 0]) ** 2

    lambda_ = exponential_fit_from_median(x_)
    p_signal = np.zeros_like(x_)

    for r in range(max_itt):

        p_signal_new = expon.cdf(x_, scale=1 / lambda_) ** len(x_)

        # force hard signal/ noise classification?
        if hard:
            p_signal_new = 1.0 * (p_signal_new > signal_prob)

        # Check stop condition
        p_change = np.max(np.abs(p_signal - p_signal_new))
        p_signal = p_signal_new
        if p_change < tol:
            break

        # Fit polynomial
        coefs = np.polynomial.polynomial.polyfit(np.log(f_), np.log(x_), deg=deg, w=1 - p_signal)
        lambda_ = np.exp(-np.polynomial.polynomial.polyval(np.log(f_), coefs))

    # final signal / noise classification is binary (hard)

    signal_mask = np.ones(len(x_fft))
    signal_mask[freq > 0] = p_signal > signal_prob

    noise_amp = np.zeros(len(x_fft))
    noise_amp[freq > 0] = np.abs(lambda_**-0.5)

    return freq, x_fft, signal_mask, noise_amp
