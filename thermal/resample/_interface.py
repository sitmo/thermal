from abc import ABC, abstractmethod


class ResampleInterface(ABC):
    @abstractmethod
    def fit(self, x, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def resample(self, size=None, *args, **kwargs):
        raise NotImplementedError
