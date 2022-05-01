"""Power Noise distributions."""

import numpy as np
from scipy import stats


class PowerNoiseDist:
    def __init__(self, signal_amp=0, noise_amp=1):
        self.signal_amp_ = signal_amp
        self.noise_amp_ = noise_amp
        self._update()

    def _update(self):
        self.nc = (self.signal_amp_ / self.noise_amp_) ** 2
        self.scale = self.noise_amp_**2

    def pdf(self, x):
        return stats.ncx2.pdf(x, df=2, nc=self.nc, scale=self.scale)

    def fit(self, samples):
        mu = np.mean(samples)
        var = np.var(samples, ddof=1)

        b = 4 * mu**2 / var
        b = max(b, 4)
        nc = ((b - 4) + (b * (b - 4)) ** 0.5) / 2
        scale = mu / (2 + nc)
        noise_amp = scale**0.5
        signal_amp = (nc**0.5) * noise_amp

        self.signal_amp_, self.noise_amp_ = signal_amp, noise_amp
        self._update()

        return self

    @property
    def mean(self):
        return (2 + self.nc) * self.scale

    @property
    def var(self):
        return 2 * (2 + 2 * self.nc) * self.scale**2

    @property
    def std(self):
        return self.var**0.5

    @property
    def signal(self):
        return self.signal_amp_

    @property
    def noise(self):
        return self.noise_amp_

    @property
    def params(self):
        return self.signal_amp_, self.noise_amp_
