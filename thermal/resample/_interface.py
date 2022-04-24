"""Resample interface definition."""
from abc import ABC, abstractmethod


class _ResampleInterface(ABC):
    @abstractmethod
    def fit(self, x, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def resample(self, size=None, *args, **kwargs):
        raise NotImplementedError
