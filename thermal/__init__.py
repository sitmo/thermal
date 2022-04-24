"""Thermal Modules."""

__author__ = """Thijs van den Berg"""
__email__ = 'thijs@sitmo.com'
__version__ = '0.4.0'

from thermal.resample import ResampleGmm, ResampleHist, ResampleKde
from thermal.stats import anderson_test, ks_test
