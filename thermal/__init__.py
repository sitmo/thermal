"""Top-level package for thermal."""

__author__ = """Thijs van den Berg"""
__email__ = 'thijs@sitmo.com'
__version__ = '0.3.0'

from .resample import ResampleGmm, ResampleHist, ResampleKde, resample_gmm, resample_hist, resample_kde
