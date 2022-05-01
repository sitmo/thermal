"""Kernel Density based fitting and Resampling."""

# Author: Thijs van den Berg <thijs@sitmo.com>
# License: MIT

import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KernelDensity

from thermal.sample._interface import _SampleInterface


def _est_kernel_width_1(x):
    return 1.06 * np.std(x, ddof=1) * len(x) ** -0.2


def _est_kernel_width_2(x):
    a = np.percentile(x, 75) - np.percentile(x, 25)
    s = np.std(x, ddof=1)
    return 0.9 * min(s, a / 1.34) * len(x) ** -0.2


class SampleKde(_SampleInterface):
    """Resample using Kernel Density estimate.

    Attributes
    ----------
    kernel_width_ : number
        The estimated Kernel width.
    """

    def __init__(self, cv=None, *args, **kwargs):
        super(SampleKde, self).__init__(*args, **kwargs)
        self.x_ = None
        self.mu_ = 0.0
        self.sigma_ = 1.0
        self.kernel_width_ = 0.0
        self.cv_ = cv

    def fit(self, x, **kwargs):
        """Estimate model parameters of the Kernel Density Resampler.

        Parameters
        ----------
        x : array-like of shape (n_samples)
            List of data points.

        Returns
        -------
        self : object
            The fitted Kernel Density Resampler.
        """
        self.n_samples_ = len(x)
        self.x_ = x
        self.kernel_width_ = _est_kernel_width_2(x)

        if self.cv_ is not None:
            bandwidths = self.kernel_width_ * np.logspace(-1, 1, 101)
            grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bandwidths}, cv=KFold(self.cv_))
            grid.fit(x.reshape(-1, 1))
            self.kernel_width_ = grid.best_params_['bandwidth']

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
         x : array, shape (n_samples)
             Randomly generated sample.
        """
        self._check_fitted()
        if n_samples is None:
            n_samples = self.n_samples_
        ans = np.random.choice(self.x_, size=n_samples, replace=True) + self.kernel_width_ * np.random.normal(
            size=n_samples
        )
        return ans

    def __str__(self):
        """Make a user friendly string representation of a class."""
        if self.cv_ is not None:
            return f'ResampleKde(cv={self.cv_})'
        else:
            return 'ResampleKde()'
