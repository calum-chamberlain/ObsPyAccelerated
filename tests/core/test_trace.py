"""
Tests for trace methods.
"""


import unittest

from obspy import read
import numpy as np

import obspyacc  # Unused import


class TestTrace(unittest.TestCase):
    def test_resample(self):
        st = read()
        for samp_rate in (5, 10, 12, 25):
            for tr in st:
                obspy_resampled = tr.copy()._obspy_resample(samp_rate)
                accelerated_resample = tr.copy().resample(samp_rate)
                self.assertTrue(np.allclose(
                    obspy_resampled.data, accelerated_resample.data))
                self.assertEqual(obspy_resampled.stats.starttime,
                                 accelerated_resample.stats.starttime)
                self.assertEqual(obspy_resampled.stats.sampling_rate,
                                 accelerated_resample.stats.sampling_rate)


if __name__ == "__main__":
    unittest.main()
