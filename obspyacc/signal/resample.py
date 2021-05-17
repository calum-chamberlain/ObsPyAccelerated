""" Frequency domain resampling. """

from typing import List
from functools import lru_cache

import numpy as np

from obspyacc import HAS_GPU
if HAS_GPU:
    import cupy as cp
    from cupy import fft
else:
    import numpy as cp
    from scipy import fft

from numba import njit, objmode


# --------------------------- HELPER FUNCS ----------------------------------

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

def cupy_resample(
    data: np.ndarray,
    npts_in: int,
    delta_in: float,
    factor: float,
    sampling_rate_out: float,
    large_w: np.ndarray = None,
):
    x = fft.rfft(data)
    x_r, x_i = cp.real(x), cp.imag(x)

    if large_w is not None:
        x_r *= large_w[:npts_in // 2 + 1]
        x_i *= large_w[:npts_in // 2 + 1]

    # interpolate
    num = int(npts_in / factor)
    df = 1.0 / (npts_in * delta_in)
    d_large_f = 1.0 / num * sampling_rate_out
    f = df * cp.arange(0, npts_in // 2 + 1, dtype=cp.int32)
    n_large_f = num // 2 + 1
    large_f = d_large_f * cp.arange(0, n_large_f, dtype=cp.int32)

    y_r = cp.interp(large_f, f, x_r)
    y_i = cp.interp(large_f, f, x_i)
    y = cp.empty_like(y_r, dtype=complex)
    y.real = y_r
    y.imag = y_i

    data = fft.irfft(y) * (float(num) / float(npts_in))
    return data


@njit
def numba_resample(
    data: np.ndarray,
    npts_in: int,
    delta_in: float,
    factor: float,
    sampling_rate_out: float,
    large_w: np.ndarray,
):
    x = np.empty_like(data, dtype=np.complex128)
    with objmode(x='complex128[:]'):
        x = np.fft.rfft(data)
    x_r, x_i = np.real(x), np.imag(x)

    if large_w is not None:
        x_r *= large_w[:npts_in // 2 + 1]
        x_i *= large_w[:npts_in // 2 + 1]

    # interpolate
    num = int(npts_in / factor)
    df = 1.0 / (npts_in * delta_in)
    d_large_f = 1.0 / num * sampling_rate_out
    f = df * np.arange(0, npts_in // 2 + 1, dtype=np.int32)
    n_large_f = num // 2 + 1
    large_f = d_large_f * np.arange(0, n_large_f, dtype=np.int32)

    y_r = np.interp(large_f, f, x_r)
    y_i = np.interp(large_f, f, x_i)
    y = np.empty_like(y_r, dtype=np.complex128)
    for i in range(y.shape[0]):
        y[i] = y_r[i] + (y_i[i] * 1j)

    data_out = np.empty_like(y, dtype=np.float64)
    with objmode(data_out='float64[:]'):
        data_out = np.fft.irfft(y)

    data_out *= (float(num) / float(npts_in))
    return data_out


def multi_resample(
    data: List[np.ndarray],
    npts_in: List[int],
    delta_in: List[float],
    factor: List[float],
    sampling_rate_out: List[float],
    large_w: List[np.ndarray],
    target: str = "CPU",
    max_workers: int = None
):
    from concurrent.futures import ThreadPoolExecutor
    from multiprocessing import cpu_count

    max_workers = min(max_workers or cpu_count(), len(data))

    assert target.upper() in ("CPU", "GPU"), f"Target {target} not supported"

    if target.upper() == "CPU":
        resample_func = numba_resample
    else:
        resample_func = cupy_resample

    if target.upper() == "CPU":
        data_out = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(
                resample_func, data=data[i], npts_in=npts_in[i],
                delta_in=delta_in[i], factor=factor[i],
                sampling_rate_out=sampling_rate_out[i],
                large_w=large_w[i])
                for i in range(len(npts_in))]
            for future in futures:
                data_out.append(future.result())
    else:
        futures = [resample_func(
            data=data[i], npts_in=npts_in[i], delta_in=delta_in[i],
            factor=factor[i], sampling_rate_out=sampling_rate_out[i],
            large_w=large_w[i]) for i in range(len(npts_in))]
        data_out = [f.get() for f in futures]  # Get the results form the GPU

    return data_out


if __name__ == "__main__":
    import doctest

    doctest.testmod()