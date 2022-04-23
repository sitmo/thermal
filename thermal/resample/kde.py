import numpy as np


class ResampleKde:
    def __init__(self, x=None):
        self.x = None
        self.mu = 0.0
        self.sigma = 1.0
        self.kernel_width = 0.1
        if x is not None:
            self.fit(x)

    def fit(self, x):
        self.size = len(x)
        self.x = x
        self.mu = np.mean(x)
        self.sigma = np.std(x, ddof=1)
        self.kernel_width = 1.06 * self.sigma * len(x) ** -0.2
        pass

    def resample(self, size=None):
        if size is None:
            size = self.size
        ans = np.random.choice(self.x, size=size, replace=True) + self.kernel_width * np.random.normal(size=size)
        ans = self.mu + (ans - self.mu) * self.sigma * (self.sigma**2 + self.kernel_width**2) ** -0.5
        return ans


def resample_kde(x, size=None):
    eng = ResampleKde(x)
    return eng.resample(size=size)
