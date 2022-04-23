from sklearn.mixture import GaussianMixture


class ResampleGmm:
    def __init__(self, x=None, *, n_components=3):
        self.size = len(x)
        self.n_components = n_components
        if x is not None:
            self.fit(x, n_components=n_components)

    def fit(self, x, *, n_components=3):
        self.n_components = n_components
        self.gmm = GaussianMixture(n_components=2).fit(x.reshape(-1, 1))

    def resample(self, size=None):
        if size is None:
            size = self.size
        samples = self.gmm.sample(n_samples=size)
        return samples[0].flatten()


def resample_gmm(x, size=None, *, n_components=3):
    eng = ResampleGmm(x, n_components=n_components)
    return eng.resample(size=size)
