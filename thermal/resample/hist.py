import numpy as np


class ResampleHist:
    def __init__(self, x=None):
        self.x = None
        if x is not None:
            self.fit(x)

    def fit(self, x):
        self.x = x
        self.size = len(x)

    def resample(self, size=None, *, replace=True):
        if size is None:
            size = self.size
        ans = np.random.choice(self.x, size=size, replace=replace)
        return ans


def resample_hist(x, size=None, *, replace=True):
    eng = ResampleHist(x)
    return eng.resample(size=size, replace=replace)
