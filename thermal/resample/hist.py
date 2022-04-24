"""Historical sampling based resampling."""

# Author: Thijs van den Berg <thijs@sitmo.com>
# License: MIT

import numpy as np

from thermal.resample._interface import _ResampleInterface


class ResampleHist(_ResampleInterface):
    """Resample using Historical sampling, either with- or without- replacement."""

    def __init__(self):
        self.size_ = 1
        self.x_ = None

    def fit(self, x, **kwargs):
        """Provide a set of samples that will be used to reample.

        Parameters
        ----------
        x : array-like of shape (n_samples)
            List of data points.

        Returns
        -------
        self : object
            The fitted Historical Resampler.
        """
        self.x_ = x
        self.size_ = len(x)
        return self

    def resample(self, size=None, *, replace=True, **kwargs):
        """Generate random samples.

        Parameters
        ----------
        size : int, default=None
            Number of samples to generate. When omitted the number of samples will be the same as
            the number of samples used to fit.
        replace : boolean, default=True
            Sample with- or without- replacement.

        Returns
        -------
        X : array, shape (n_samples)
            Randomly drawen sample.
        """
        if size is None:
            size = self.size_
        ans = np.random.choice(self.x_, size=size, replace=replace)
        return ans
