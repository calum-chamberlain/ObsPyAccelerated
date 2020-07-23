"""
Methods to monkey-patch the obspy.core.Trace object.
"""

from typing import Union, Callable

import numpy as np

import obspy
from obspy.core.util.decorator import skip_if_no_data

from obspyacc import signal, fftlib, gpulib, HAS_GPU
from obspyacc.helpers.patcher import obspy_docs


# --------------- DATA SYNCHRONISATION --------------
def set_data(
    self: obspy.Trace,
    data: Union[np.ndarray, gpulib.ndarray],
):
    self._data = data
    self._gpu_data = gpulib.asarray(data)


def get_data(self: obspy.Trace):
    return self._data


setattr(obspy.Trace, "data", property(fget=get_data, fset=set_data))

# TODO: It would be great to have a way to not copy between VRAM and RAM all the time.

# --------------- NEW METHODS ------------------------


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

    num_samples = int(self.stats.npts / factor)
    if HAS_GPU:
        data = self._gpu_data
    else:
        data = self.data
    data = signal.resample(x=data, num=num_samples, window=window)
    if num_samples % 2 == 0:  # Hack to give the same result as obspy.
        data = fftlib.irfft(fftlib.rfft(data), n=num_samples)
    if HAS_GPU:
        self.data = data.get()
    else:
        self.data = data
    self.stats.sampling_rate = sampling_rate

    return self

# -------------------- MONKEYPATCH METHODS -------------------------


setattr(obspy.Trace, "_obspy_resample", obspy.Trace.resample)
obspy.Trace.resample = gpu_resample


if __name__ == "__main__":
    import doctest

    doctest.testmod()
