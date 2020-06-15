"""
Methods to monkey-patch the obspy.core.Trace object.
"""

from typing import Union, Callable

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
    window: Union[str, Callable, np.ndarray] = "hanning",
    no_filter: bool = True,
    strict_length: bool = False
) -> obspy.Trace:
    """
    GPU accelerated frequency domain resampling.

    {obspy_docs}
    """
    from future.utils import native_str
    from scipy.signal import get_window

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

    # Move data to the GPU
    gpu_data = gpulib.asarray(self.data.newbyteorder("="))
    # resample in the frequency domain. Make sure the byteorder is native.
    x = gpulib.fft.rfft(gpu_data)
    x_r = x.real
    x_i = x.imag

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
        large_w = gpulib.asarray(large_w)
        x_r *= large_w[:self.stats.npts // 2 + 1]
        x_i *= large_w[:self.stats.npts // 2 + 1]

    # interpolate - no cuda interpolation, move to cpu
    x_r, x_i = gpulib.asnumpy(x_r), gpulib.asnumpy(x_i)
    num = int(self.stats.npts / factor)
    df = 1.0 / (self.stats.npts * self.stats.delta)
    d_large_f = 1.0 / num * sampling_rate
    f = df * np.arange(0, self.stats.npts // 2 + 1, dtype=np.int32)
    n_large_f = num // 2 + 1
    large_f = d_large_f * np.arange(0, n_large_f, dtype=np.int32)
    large_y = np.zeros((2 * n_large_f))
    x_r = np.interp(large_f, f, x_r)
    x_i = np.interp(large_f, f, x_i)
    large_y = np.vectorize(complex)(x_r, x_i)

    # IFFT on the gpu.
    self.data = gpulib.asnumpy(
        gpulib.fft.irfft(gpulib.asarray(large_y)) *
        (float(num) / float(self.stats.npts)))
    self.stats.sampling_rate = sampling_rate

    return self

# -------------------- MONKEYPATCH METHODS -------------------------


setattr(obspy.Trace, "_obspy_resample", obspy.Trace.resample)
obspy.Trace.resample = gpu_resample


if __name__ == "__main__":
    import doctest

    doctest.testmod()
