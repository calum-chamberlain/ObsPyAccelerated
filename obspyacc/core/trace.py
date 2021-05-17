"""
Methods to monkey-patch the obspy.core.Trace object.
"""

from typing import Union, Callable

import cupy
import numpy as np
from scipy.fft import next_fast_len

import obspy
from obspy.core.util.decorator import skip_if_no_data

from obspyacc import gpulib, HAS_GPU
from obspyacc.signal.resample import (
    cupy_resample, numba_resample, prep_for_resample, get_resample_window)
from obspyacc.helpers.patcher import obspy_docs


# --------------- DATA SYNCHRONISATION --------------
# TODO: At the moment, you can break this by changing the underlying arrays
#  in-place - this also keeps using GPU memory when it isn't needed...
#  Trade-off between hosting on VRAM for speed and VRAM usage...
# setattr(obspy.Trace, "_gpu_data_update_time", 0)
# setattr(obspy.Trace, "_data_update_time", 0)
#
#
# def _set_data(
#     self: obspy.Trace,
#     data: np.ndarray,
# ):
#     self._data = data
#     self._data_update_time = time.time()
#
#
# def _get_data(self: obspy.Trace):
#     if self._gpu_data_update_time > self._data_update_time:
#         # print("Synchronising gpu data to cpu")
#         # Synchronise
#         self.data = gpulib.asnumpy(self._gpu_data)
#         self._data_update_time = self._gpu_data_update_time
#     try:
#         return self._data
#     except AttributeError:
#         raise AttributeError("Ensure you imported obspyacc before code")
#
#
# setattr(obspy.Trace, "data", property(fget=_get_data, fset=_set_data))
#
#
# def _set_gpu_data(
#     self: obspy.Trace,
#     data: gpulib.array,
# ):
#     self.__gpu_data = data
#     self._gpu_data_update_time = time.time()
#
#
# def _get_gpu_data(self: obspy.Trace):
#     if self._data_update_time > self._gpu_data_update_time:
#         # print("Synchronising cpu data to gpu")
#         # Synchronise
#         self._gpu_data = gpulib.asarray(self.data)
#         self._gpu_data_update_time = self._data_update_time
#     try:
#         return self.__gpu_data
#     except AttributeError:
#         raise AttributeError("Ensure you imported obspyacc before code")
#
#
# setattr(obspy.Trace, "_gpu_data",
#         property(fget=_get_gpu_data, fset=_set_gpu_data))

# -------------------- GPU DATA ACCESS -----------------

def _set_gpu_data(
    self: obspy.Trace,
    data: gpulib.array
):
    # Move data to RAM
    self.data = gpulib.asnumpy(data)


def _get_gpu_data(
    self: obspy.Trace
):
    # Move data to GPU
    return gpulib.asarray(self.data)


if HAS_GPU:
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

# --------------- GPU METHODS ------------------------


@obspy_docs(method_or_function=obspy.Trace.resample)
@skip_if_no_data
def trace_resample(
    self: obspy.Trace,
    sampling_rate: float,
    window: Union[str, Callable, np.ndarray] = "hanning",
    no_filter: bool = True,
    strict_length: bool = False,
    target: str = "GPU",
    *args, **kwargs
) -> obspy.Trace:
    """
    GPU accelerated frequency domain resampling.

    {obspy_docs}
    """
    assert target.upper() in ("CPU", "GPU"), f"Target {target} not supported"
    self, factor = prep_for_resample(
        trace=self, sampling_rate=sampling_rate, no_filter=no_filter,
        strict_length=strict_length)

    # Get the window to convolve and stabalise resampling
    fftlen = next_fast_len(self.stats.npts)
    large_w = get_resample_window(window=window, npts=fftlen)

    # Do the resampling!
    if target.upper() == "GPU" and HAS_GPU:
        data = self._gpu_data
        # Move the window to the GPU
        if large_w is not None:
            large_w = cupy.asarray(large_w)
        resample_func = cupy_resample
    else:
        data = self.data
        resample_func = numba_resample

    data = resample_func(
        data=data, npts_in=self.stats.npts,
        fftlen=fftlen, delta_in=self.stats.delta,
        factor=factor, sampling_rate_out=sampling_rate, large_w=large_w)
    if target.upper() == "GPU" and HAS_GPU:
        self._gpu_data = data.get()  # synchronisation taken care of elsewhere
    else:
        self.data = data
    self.stats.sampling_rate = sampling_rate

    return self

# -------------------- MONKEYPATCH METHODS -------------------------

setattr(obspy.Trace, "_obspy_resample", obspy.Trace.resample)
obspy.Trace.resample = trace_resample


if __name__ == "__main__":
    import doctest

    doctest.testmod()
