import numpy as np

from thermal.resample._interface import ResampleInterface


class ResampleHist(ResampleInterface):
    def __init__(self):
        self.size_ = 1
        self.x_ = None

    def fit(self, x, **kwargs):
        self.x_ = x
        self.size_ = len(x)
        return self

    def resample(self, size=None, *, replace=True, **kwargs):
        if size is None:
            size = self.size_
        ans = np.random.choice(self.x_, size=size, replace=replace)
        return ans
