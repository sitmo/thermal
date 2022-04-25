"""Gaussian Mixture Model based fitting and Resampling."""

# Author: Thijs van den Berg <thijs@sitmo.com>
# License: MIT

from thermal.sample._interface import _SampleInterface
from thermal.sample.gmm import SampleGmm
from thermal.sample.hist import SampleHist
from thermal.sample.kde import SampleKde


class Sample(_SampleInterface):
    """Generic resampling algorithm interface.

    Parameters
    ----------
    amplitude : string, default='gmm'
        The amplitude density resampling method.
    n_components : int, default=3
        The number of mixture components used when amplitude=='gmm.

    Attributes
    ----------
    size_ : int
        The default number of samples to generate.

    Example
    -------
    ::

        import numpy as np
        import thermal as th

        x = np.random.normal(size=10)
        s = th.Resample().fit(x).sample()

        s
        >>> array([ 0.01212549,  0.04772549,  0.08693959, ..., -0.00519905,
           -0.00908192,  0.00756048])

    """

    def __init__(self, amplitude='gmm', *, n_components=7, **kwargs):
        super(Sample, self).__init__(**kwargs)

        self.amplitude_ = amplitude

        if amplitude == 'gmm':
            self.amp_ = SampleGmm(n_components)
        elif amplitude == 'kde':
            self.amp_ = SampleKde()
        elif amplitude == 'hist':
            self.amp_ = SampleHist()
        else:
            raise ValueError(f'Unknown amplitude_method "{amplitude}".')

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
        self.amp_.fit(x, **kwargs)
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
        return self.amp_.sample(n_samples=n_samples, **kwargs)

    def __str__(self):
        """Make a user friendly string representation of a class."""
        return f'Resample:(amplitude={str(self.amp_)})'
