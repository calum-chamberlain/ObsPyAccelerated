"""
Tests for trace methods.
"""


import unittest

from obspy import read
import numpy as np

import obspyacc  # Unused import


# TODO use Pytest fixtures for a few different streams
class TestTrace(unittest.TestCase):
    def test_resample(self):
        st = read().detrend()
        for samp_rate in (100, 75, 65, 50, 25, 10):
            for tr in st:
                obspy_resampled = tr.copy()._obspy_resample(samp_rate)
                for target in ("CPU", "GPU"):
                    tr.target = target
                    accelerated_resample = tr.copy().resample(samp_rate)
                    assert np.allclose(
                        obspy_resampled.data,
                        accelerated_resample.data), f"Result differs for {samp_rate}"
                    assert obspy_resampled.stats.starttime == accelerated_resample.stats.starttime
                    assert obspy_resampled.stats.sampling_rate == accelerated_resample.stats.sampling_rate


if __name__ == "__main__":
    unittest.main()
