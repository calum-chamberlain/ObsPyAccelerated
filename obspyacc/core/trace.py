"""
Methods to monkey-patch the obspy.core.Trace object.
"""

import numpy as np

import obspy
from obspy.core.util.decorator import skip_if_no_data

from obspyacc import gpulib
from obspyacc.helpers.patcher import obspy_docs


@obspy_docs(method_or_function=obspy.Trace.resample)
@skip_if_no_data
def gpu_resample(
    self: obspy.Trace,
    sampling_rate: float,
    window: str,
    no_filter: bool,
    strict_length: bool
) -> obspy.Trace:
    """
    GPU accelerated frequency domain resampling.

    {obspy_docs}
    """
    from scipy.signal import get_window
    from scipy.fftpack import rfft, irfft
    factor = self.stats.sampling_rate / float(sampling_rate)
    # check if end time changes and this is not explicitly allowed
    if strict_length:
        if len(self.data) % factor != 0.0:
            msg = "End time of trace would change and strict_length=True."
            raise ValueError(msg)
    # do automatic lowpass filtering
    if not no_filter:
        # be sure filter still behaves good
        if factor > 16:
            msg = "Automatic filter design is unstable for resampling " + \
                  "factors (current sampling rate/new sampling rate) " + \
                  "above 16. Manual resampling is necessary."
            raise ArithmeticError(msg)
        freq = self.stats.sampling_rate * 0.5 / float(factor)
        self.filter('lowpass_cheby_2', freq=freq, maxorder=12)

    # resample in the frequency domain. Make sure the byteorder is native.
    x = rfft(self.data.newbyteorder("="))
    # Cast the value to be inserted to the same dtype as the array to avoid
    # issues with numpy rule 'safe'.
    x = np.insert(x, 1, x.dtype.type(0))
    if self.stats.npts % 2 == 0:
        x = np.append(x, [0])
    x_r = x[::2]
    x_i = x[1::2]

    if window is not None:
        if callable(window):
            large_w = window(np.fft.fftfreq(self.stats.npts))
        elif isinstance(window, np.ndarray):
            if window.shape != (self.stats.npts,):
                msg = "Window has the wrong shape. Window length must " + \
                      "equal the number of points."
                raise ValueError(msg)
            large_w = window
        else:
            large_w = np.fft.ifftshift(get_window(native_str(window),
                                                  self.stats.npts))
        x_r *= large_w[:self.stats.npts // 2 + 1]
        x_i *= large_w[:self.stats.npts // 2 + 1]

    # interpolate
    num = int(self.stats.npts / factor)
    df = 1.0 / (self.stats.npts * self.stats.delta)
    d_large_f = 1.0 / num * sampling_rate
    f = df * np.arange(0, self.stats.npts // 2 + 1, dtype=np.int32)
    n_large_f = num // 2 + 1
    large_f = d_large_f * np.arange(0, n_large_f, dtype=np.int32)
    large_y = np.zeros((2 * n_large_f))
    large_y[::2] = np.interp(large_f, f, x_r)
    large_y[1::2] = np.interp(large_f, f, x_i)

    large_y = np.delete(large_y, 1)
    if num % 2 == 0:
        large_y = np.delete(large_y, -1)
    self.data = irfft(large_y) * (float(num) / float(self.stats.npts))
    self.stats.sampling_rate = sampling_rate

    return self

# -------------------- MONKEYPATCH METHODS -------------------------


obspy.Trace.resample = gpu_resample


if __name__ == "__main__":
    import doctest

    doctest.testmod()
