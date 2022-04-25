"""Resample interface definition."""
from abc import ABC, abstractmethod


class _ResampleInterface(ABC):
    def __init__(self, *args, **kwargs):
        self.size_ = -1
        self.x_ = None

    @abstractmethod
    def fit(self, x, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def resample(self, size=None, *args, **kwargs):
        raise NotImplementedError

    def _check_fitted(self):
        if self.size_ == -1:
            raise ValueError('This resampler instance is not yet fitted.')
