"""Resample interface definition."""
from abc import ABC, abstractmethod


class _SampleInterface(ABC):
    def __init__(self, *args, **kwargs):
        self.n_samples_ = 0
        self.x_ = None

    @abstractmethod
    def fit(self, x, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def sample(self, n_samples=None, *args, **kwargs):
        raise NotImplementedError

    def _check_fitted(self):
        if self.n_samples_ == 0:
            raise ValueError('This sampler instance is not yet fitted.')
