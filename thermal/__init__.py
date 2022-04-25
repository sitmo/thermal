"""Thermal Modules."""

__author__ = """Thijs van den Berg"""
__email__ = 'thijs@sitmo.com'
__version__ = '0.6.3'

from thermal.sample import Sample, SampleGmm, SampleHist, SampleKde
from thermal.stats import anderson_test, ks_test
