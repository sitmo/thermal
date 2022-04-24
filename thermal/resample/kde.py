"""Kernel Density based fitting and Resampling."""

# Author: Thijs van den Berg <thijs@sitmo.com>
# License: MIT

import numpy as np

from thermal.resample._interface import _ResampleInterface


class ResampleKde(_ResampleInterface):
    """Resample using Kernel Density estimate.

    Attributes
    ----------
    kernel_width_ : number
        The estimated Kernel width.
    """

    def __init__(self, x=None):
        self.size_ = 0
        self.x_ = None
        self.mu_ = 0.0
        self.sigma_ = 1.0
        self.kernel_width_ = 0.0
        if x is not None:
            self.fit(x)

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
        self.size_ = len(x)
        self.x_ = x
        self.mu_ = np.mean(x)
        self.sigma_ = np.std(x, ddof=1)
        self.kernel_width_ = 1.06 * self.sigma_ * len(x) ** -0.2
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
         x : array, shape (n_samples)
             Randomly generated sample.
        """
        if size is None:
            size = self.size_
        ans = np.random.choice(self.x_, size=size, replace=True) + self.kernel_width_ * np.random.normal(size=size)
        # ans = self.mu_ + (ans - self.mu_) * self.sigma_ * (self.sigma_**2 + self.kernel_width_**2) ** -0.5
        return ans
