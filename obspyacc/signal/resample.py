""" Frequency domain resampling. """

from obspyacc import HAS_GPU
if HAS_GPU:
    import cupy as np
    from cupy import fft
else:
    import numpy as np
    from scipy import fft
# from numba import njit, objmode


def resample(
    data: np.ndarray,
    delta_in: float,
    factor: float,
    sampling_rate_out: float,
    large_w: np.ndarray = None,
):
    npts_in = data.shape[0]
    x = fft.rfft(data)
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
    y = np.empty_like(y_r, dtype=complex)
    y.real = y_r
    y.imag = y_i

    data = fft.irfft(y) * (float(num) / float(npts_in))
    return data


# @njit
# def numba_resamp(
#     data: np.ndarray,
#     delta_in: float,
#     factor: float,
#     sampling_rate_out: float,
#     large_w: np.ndarray = None,
# ):
#     npts_in = data.shape[0]
#     x = np.zeros_like(data, dtype=np.complex128)
#     with objmode(x='complex128[:]'):
#         x = fft.rfft(data)
#     x_r, x_i = np.real(x), np.imag(x)
#
#     if large_w is not None:
#         x_r *= large_w[:npts_in // 2 + 1]
#         x_i *= large_w[:npts_in // 2 + 1]
#
#     # interpolate
#     num = int(npts_in / factor)
#     df = 1.0 / (npts_in * delta_in)
#     d_large_f = 1.0 / num * sampling_rate_out
#     f = df * np.arange(0, npts_in // 2 + 1, dtype=np.int32)
#     n_large_f = num // 2 + 1
#     large_f = d_large_f * np.arange(0, n_large_f, dtype=np.int32)
#
#     y_r = np.interp(large_f, f, x_r)
#     y_i = np.interp(large_f, f, x_i)
#     y = np.empty_like(y_r, dtype=np.complex128)
#     for i in range(y.shape[0]):
#         y[i] = y_r[i] + (y_i[i] * 1j)
#
#     data = np.zeros_like(y, dtype=np.float64)
#     with objmode(data='float64[:]'):
#         data = fft.irfft(y)
#     data *= (float(num) / float(npts_in))
#     return data

