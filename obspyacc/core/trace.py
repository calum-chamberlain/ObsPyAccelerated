"""
Methods to monkey-patch the obspy.core.Trace object.
"""

from typing import Union, Callable

import numpy as np
import time

import obspy
from obspy.core.util.decorator import skip_if_no_data

from obspyacc import signal, fftlib, gpulib, HAS_GPU
from obspyacc.helpers.patcher import obspy_docs


# --------------- DATA SYNCHRONISATION --------------
# TODO: At the moment, you can break this by changing the underlying arrays in-place
setattr(obspy.Trace, "_gpu_data_update_time", 0)
setattr(obspy.Trace, "_data_update_time", 0)


def _set_data(
    self: obspy.Trace,
    data: np.ndarray,
):
    self._data = data
    self._data_update_time = time.time()


def _get_data(self: obspy.Trace):
    if self._gpu_data_update_time > self._data_update_time:
        print("Synchronising gpu data to cpu")
        # Synchronise
        self.data = gpulib.asnumpy(self._gpu_data)
        self._data_update_time = self._gpu_data_update_time
    return self._data


setattr(obspy.Trace, "data", property(fget=_get_data, fset=_set_data))


def _set_gpu_data(
    self: obspy.Trace,
    data: gpulib.array,
):
    self.__gpu_data = data
    self._gpu_data_update_time = time.time()


def _get_gpu_data(self: obspy.Trace):
    if self._data_update_time > self._gpu_data_update_time:
        print("Synchronising cpu data to gpu")
        # Synchronise
        self._gpu_data = gpulib.asarray(self.data)
        self._gpu_data_update_time = self._data_update_time
    return self.__gpu_data


setattr(obspy.Trace, "_gpu_data",
        property(fget=_get_gpu_data, fset=_set_gpu_data))


# --------------- GPU MEMORY MANAGEMENT --------------

def release_vram(self: obspy.Trace):
    mempool = gpulib.get_default_memory_pool()
    pinned_mempool = gpulib.get_default_pinned_memory_pool()
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()


setattr(obspy.Trace, "release_vram", release_vram)


def delete(self: obspy.Trace):
    # Clean up
    self.release_vram()


setattr(obspy.Trace, "__del__", delete)

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
        self._gpu_data = data.get()  # synchronisation taken care of elsewhere
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
