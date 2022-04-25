"""Historical sampling based resampling."""

# Author: Thijs van den Berg <thijs@sitmo.com>
# License: MIT

import numpy as np

from thermal.sample._interface import _SampleInterface


class SampleHist(_SampleInterface):
    """Sample using Historical sampling, either with- or without- replacement.

    Parameters
    ----------
    replace : boolean, default=True
        Sample with- or without-  replacement.
    """

    def __init__(self, replace=True, *args, **kwargs):
        super(SampleHist, self).__init__(*args, **kwargs)
        self.x_ = None
        self.replace_ = replace

    def fit(self, x, *args, **kwargs):
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
        self.n_samples_ = len(x)
        return self

    def sample(self, n_samples=None, *args, **kwargs):
        """Generate random samples.

        Parameters
        ----------
        n_samples : int, default=None
            Number of samples to generate. When omitted the number of samples will be the same as
            the number of samples used to fit.
        replace : boolean, default=True
            Sample with- or without- replacement.

        Returns
        -------
        X : array, shape (n_samples)
            Randomly drawen sample.
        """
        self._check_fitted()
        if n_samples is None:
            n_samples = self.n_samples_
        if n_samples > len(self.x_):
            ans = []
            remaining = n_samples
            while remaining > 0:
                chunk_size = min(len(self.x_), remaining)
                ans.append(np.random.choice(self.x_, size=chunk_size, replace=self.replace_))
                remaining -= chunk_size
            ans = np.concatenate(ans)
        else:
            ans = np.random.choice(self.x_, size=n_samples, replace=self.replace_)
        return ans

    def __str__(self):
        """Make a user friendly string representation of a class."""
        return f'ResampleHist(replace={self.replace_})'
