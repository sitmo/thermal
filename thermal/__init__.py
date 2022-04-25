"""Thermal Modules."""

__author__ = """Thijs van den Berg"""
__email__ = 'thijs@sitmo.com'
__version__ = '0.6.1'

from thermal.resample import Resample, ResampleGmm, ResampleHist, ResampleKde
from thermal.stats import anderson_test, ks_test
