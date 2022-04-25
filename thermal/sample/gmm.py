"""Gaussian Mixture Model based fitting and Resampling."""

# Author: Thijs van den Berg <thijs@sitmo.com>
# License: MIT
import numpy as np
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

from thermal.sample._interface import _SampleInterface

"""

lowest_bic = np.infty
bic = []
n_components_range = range(1, 7)
cv_types = ["spherical", "tied", "diag", "full"]
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(
            n_components=n_components, covariance_type=cv_type
        )
        gmm.fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

"""


class SampleGmm(_SampleInterface):
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
        s = th.ResampleGmm(3).fit(x).sample()

        s
        >>> array([ 0.01212549,  0.04772549,  0.08693959, ..., -0.00519905,
           -0.00908192,  0.00756048])

    """

    def __init__(self, n_components=7, *, prune=True, bayesian=False, **kwargs):
        super(SampleGmm, self).__init__(**kwargs)

        self.gmm_ = None
        self.bayesian_ = bayesian
        self.prune_ = prune
        self.n_components_ = n_components
        self.bic_ = None

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
        self.n_samples_ = len(x)

        lowest_bic = np.infty
        best_gmm = None
        best_n_components = None
        bic = []
        if self.bayesian_:
            self.gmm_ = BayesianGaussianMixture(n_components=self.n_components_, max_iter=1000).fit(x.reshape(-1, 1))
        else:
            if self.prune_:
                n_components_range = range(1, self.n_components_ + 1)
            else:
                n_components_range = [self.n_components_]

            for n_components in n_components_range:
                gmm = GaussianMixture(n_components=n_components, max_iter=1000).fit(x.reshape(-1, 1))
                bic.append(gmm.bic(x.reshape(-1, 1)))
                if bic[-1] < lowest_bic:
                    lowest_bic = bic[-1]
                    best_gmm = gmm
                    best_n_components = n_components

            self.gmm_ = best_gmm
            self.n_components_ = best_n_components
            self.bic_ = bic
        return self

    def sample(self, n_samples=None, **kwargs):
        """Generate random samples.

        Parameters
        ----------
        n_samples : int, default=None
            Number of samples to generate. When omitted the number of samples will be the same as
            the number of samples used to fit.

        Returns
        -------
        X : array, shape (n_samples)
            Randomly generated sample.
        """
        self._check_fitted()
        if n_samples is None:
            n_samples = self.n_samples_
        samples = self.gmm_.sample(n_samples=n_samples)
        return samples[0].flatten()

    def __str__(self):
        """Make a user friendly string representation of a class."""
        return f'ResampleGmm(bayesian={self.bayesian_}, n_components={self.n_components_})'
