"""The stats module implements statistical test routines."""

from thermal.stats.autocorr import autocorr
from thermal.stats.rangescale import hurst
from thermal.stats.samples_tests import anderson_test, ks_test
