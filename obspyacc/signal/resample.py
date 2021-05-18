""" Frequency domain resampling. """
from typing import Iterable
from functools import lru_cache
from collections import deque

from concurrent.futures import ThreadPoolExecutor as Executor
# from concurrent.futures import ProcessPoolExecutor as Executor
from multiprocessing import cpu_count

import numpy as np
import scipy.fft as fftlib

from obspyacc import HAS_GPU
if HAS_GPU:
    import cupy as cp
    from cupy import fft
    import cupy.cuda.memory
else:
    import numpy as cp
    from scipy import fft

from numba import njit, objmode

from obspyacc.helpers.profilers import measure_time, Timer


# --------------------------- HELPER FUNCS ----------------------------------

# @measure_time
@lru_cache
def get_resample_window(window, npts):
    from scipy.signal import get_window
    if window is not None:
        if callable(window):
            large_w = window(np.fft.fftfreq(npts))
        elif isinstance(window, np.ndarray):
            if window.shape != (npts,):
                msg = "Window has the wrong shape. Window length must " + \
                        "equal the number of points."
                raise ValueError(msg)
            large_w = window
        else:
            large_w = np.fft.ifftshift(get_window(window, npts))
    else:
        large_w = None
    return large_w


# @measure_time
def prep_for_resample(trace, sampling_rate, strict_length, no_filter):
    factor = trace.stats.sampling_rate / float(sampling_rate)
    # check if end time changes and this is not explicitly allowed
    if strict_length:
        if len(trace.data) % factor != 0.0:
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
        freq = trace.stats.sampling_rate * 0.5 / float(factor)
        trace.filter('lowpass_cheby_2', freq=freq, maxorder=12)
    return trace, factor


# ------------------------------ RESAMPLERS ---------------------------------

# @measure_time
def cupy_resample(
    data: np.ndarray,
    npts_in: int,
    fftlen: int,
    delta_in: float,
    factor: float,
    sampling_rate_out: float,
    large_w: np.ndarray = None,
):
    x = fft.rfft(data, n=fftlen)
    x_r, x_i = cp.real(x), cp.imag(x)

    if large_w is not None:
        x_r *= large_w[:fftlen // 2 + 1]
        x_i *= large_w[:fftlen // 2 + 1]

    # interpolate
    num = int(npts_in / factor)
    df = 1.0 / (npts_in * delta_in)
    d_large_f = 1.0 / num * sampling_rate_out
    f = df * cp.arange(0, fftlen // 2 + 1, dtype=cp.int32)
    n_large_f = num // 2 + 1
    large_f = d_large_f * cp.arange(0, n_large_f, dtype=cp.int32)

    y = np.interp(large_f, f, x_r) + (1j * np.interp(large_f, f, x_i))

    data = fft.irfft(y, n=int(fftlen / factor))[0:num] * (float(num) / float(fftlen))
    return data


# @measure_time
@njit(fastmath=True)
def numba_resample(
    data: np.ndarray,
    npts_in: int,
    fftlen: int,
    delta_in: float,
    factor: float,
    sampling_rate_out: float,
    large_w: np.ndarray,
):
    x = np.empty_like(data, dtype=np.complex128)
    with objmode(x='complex128[:]'):
        x = fftlib.rfft(data, n=fftlen)
    # x = fftlib.rfft(data, n=fftlen)
    x_r, x_i = np.real(x), np.imag(x)

    if large_w is not None:
        x_r *= large_w[:fftlen // 2 + 1]
        x_i *= large_w[:fftlen // 2 + 1]

    # interpolate
    num = int(npts_in / factor)
    df = 1.0 / (npts_in * delta_in)
    d_large_f = 1.0 / num * sampling_rate_out
    f = df * np.arange(0, fftlen // 2 + 1, dtype=np.int32)
    n_large_f = num // 2 + 1
    large_f = d_large_f * np.arange(0, n_large_f, dtype=np.int32)

    y = np.interp(large_f, f, x_r) + (1j * np.interp(large_f, f, x_i))

    data_out = np.empty_like(y, dtype=np.float64)
    with objmode(data_out='float64[:]'):
        # data_out = fftlib.irfft(y)
        data_out = fftlib.irfft(y, n=int(fftlen / factor))[0:num]
    data_out *= (float(num) / float(fftlen))
    return data_out


def multi_resample(
    data: Iterable[np.ndarray],
    npts_in: Iterable[int],
    fftlen: Iterable[int],
    delta_in: Iterable[float],
    factor: Iterable[float],
    sampling_rate_out: Iterable[float],
    large_w: Iterable[np.ndarray],
    n_traces: int,
    target: str = "CPU",
    max_workers: int = None,
    chunksize: int = None,
):
    max_workers = min(max_workers or cpu_count(), n_traces)
    if target.upper() == "CPU":
        chunksize = chunksize or n_traces // max_workers
    else:
        chunksize = chunksize or n_traces

    assert target.upper() in ("CPU", "GPU"), f"Target {target} not supported"

    params = (dict(
        data=_data, npts_in=_npts_in, fftlen=_fftlen, delta_in=_delta_in,
        factor=_factor, sampling_rate_out=_sampling_rate_out,
        large_w=_large_w)
        for _data, _npts_in, _fftlen, _delta_in, _factor, _sampling_rate_out, _large_w
        in zip(data, npts_in, fftlen, delta_in, factor, sampling_rate_out, large_w))

    if target.upper() == "CPU":
        if max_workers > 1:
            data_out = []
            with Executor(max_workers=max_workers) as executor:
                futures = executor.map(
                    lambda param: numba_resample(**param), params,
                    chunksize=chunksize)
                for future in futures:
                    data_out.append(future)
        else:
            data_out = [numba_resample(**param) for param in params]
    else:
        data_out, futures, i = [], deque([]), 0
        _mem_limited = False
        while i < n_traces:
            if not _mem_limited:
                # Only get the next one if the previous one succeded
                param = next(params)
            try:
                futures.append(cupy_resample(**param))
                _mem_limited = False
                # Make sure we reset this if we previously hit the limit
            except cupy.cuda.memory.OutOfMemoryError:
                print("Hit memory limit of GPU")
                chunksize = len(futures) - 1
                _mem_limited = True  # Set to rerun previous chunk
            while len(futures) >= chunksize:  # Cope with change to chunksize on memory error
                # Empty one from the queue
                data_out.append(futures.popleft().get())
            if not _mem_limited:
                i += 1
            else:
                print("Hit memory limit, re-running")
        while len(futures):
            data_out.append(futures.popleft().get())
        assert len(data_out) == n_traces

    return data_out


if __name__ == "__main__":
    import doctest

    doctest.testmod()
