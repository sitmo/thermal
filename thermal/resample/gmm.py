"""Gaussian Mixture Model based fitting and Resampling."""

# Author: Thijs van den Berg <thijs@sitmo.com>
# License: MIT

from sklearn.mixture import GaussianMixture

from thermal.resample._interface import _ResampleInterface


class ResampleGmm(_ResampleInterface):
    """Resample using Gaussian Mixtures.

    Parameters
    ----------
    n_components : int, default=3
        The number of mixture components.

    Attributes
    ----------
    size_ : int
        The default number of samples to generate.
    gmm_ : sklearn.mixture.GaussianMixture object.
        The Gaussian Mixture model.

    Example
    -------
    ::

        import numpy as np
        import thermal as th

        x = np.random.normal(size=10)
        s = th.ResampleGmm(3).fit(x).resample()

        s
        >>> array([ 0.01212549,  0.04772549,  0.08693959, ..., -0.00519905,
           -0.00908192,  0.00756048])

    """

    def __init__(self, n_components=3):
        self.size_ = 1
        self.gmm_ = None
        self.n_components_ = n_components

    def fit(self, x, **kwargs):
        """Estimate model parameters of the Gaussian Mixtures Resampler.

        Parameters
        ----------
        x : array-like of shape (n_samples)
            List of data points.

        Returns
        -------
        self : object
            The fitted Gaussian Mixtures Resampler.
        """
        self.size_ = len(x)
        self.gmm_ = GaussianMixture(n_components=self.n_components_).fit(x.reshape(-1, 1))
        return self

    def resample(self, size=None, **kwargs):
        """Generate random samples.

        Parameters
        ----------
        size : int, default=None
            Number of samples to generate. When omitted the number of samples will be the same as
            the number of samples used to fit.

        Returns
        -------
        X : array, shape (n_samples)
            Randomly generated sample.
        """
        if self.gmm_ is None:
            raise ValueError(
                "This ResampleGmm instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this resampler. "
            )
        if size is None:
            size = self.size_
        samples = self.gmm_.sample(n_samples=size)
        return samples[0].flatten()
