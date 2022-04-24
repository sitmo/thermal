import numpy as np

from thermal.resample._interface import ResampleInterface


class ResampleKde(ResampleInterface):
    def __init__(self, x=None):
        self.x_ = None
        self.mu_ = 0.0
        self.sigma_ = 1.0
        self.kernel_width_ = 0.1
        if x is not None:
            self.fit(x)

    def fit(self, x, **kwargs):
        self.size_ = len(x)
        self.x_ = x
        self.mu_ = np.mean(x)
        self.sigma__ = np.std(x, ddof=1)
        self.kernel_width_ = 1.06 * self.sigma_ * len(x) ** -0.2
        return self

    def resample(self, size=None, **kwargs):
        if size is None:
            size = self.size_
        ans = np.random.choice(self.x_, size=size, replace=True) + self.kernel_width_ * np.random.normal(size=size)
        ans = self.mu_ + (ans - self.mu_) * self.sigma_ * (self.sigma_**2 + self.kernel_width_**2) ** -0.5
        return ans
