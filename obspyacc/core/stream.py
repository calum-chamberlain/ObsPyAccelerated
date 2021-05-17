"""
Methods for monkey patching ObsPy Stream methods.
"""

from typing import Union, Callable
import warnings

import numpy as np
from scipy.fft import next_fast_len

import obspy
from obspy.core.util.decorator import skip_if_no_data

from obspyacc import HAS_GPU
from obspyacc.signal.resample import (
    prep_for_resample, get_resample_window, multi_resample)
from obspyacc.helpers.patcher import obspy_docs

if HAS_GPU:
    import cupy as cp
else:
    import numpy as cp

# ---------------------- METHODS -------------------------------


def _obspy_resample(
    stream: obspy.core.Stream,
    sampling_rate: float,
    window: Union[str, Callable, np.ndarray] = "hanning",
    no_filter: bool = True,
    strict_length: bool = False,
) -> obspy.core.Stream:
    for tr in stream:
        tr._obspy_resample(
            sampling_rate=sampling_rate, window=window, no_filter=no_filter,
            strict_length=strict_length)
    return stream


@obspy_docs(method_or_function=obspy.Stream.resample)
@skip_if_no_data
def stream_resample(
    stream: obspy.core.Stream,
    sampling_rate: float,
    window: Union[str, Callable, np.ndarray] = "hanning",
    no_filter: bool = True,
    strict_length: bool = False,
    max_workers: int = None,
) -> obspy.core.Stream:
    """
    Accelerated frequency domain resampling.

    {obspy_docs}
    Additional Parameters
    ---------------------
    target:
        Either CPU or GPU to target running on the CPU or GPU

    """
    target = {tr.target or "CPU" for tr in stream}
    if len(target) > 1:
        warnings.warn("Multiple targets found in traces, defaulting to CPU")
        target = {"CPU"}
    target = target.pop()
    assert target.upper() in ("CPU", "GPU"), f"Target {target} not supported"
    if isinstance(stream, obspy.core.Trace):
        traces = [stream]
    else:
        traces = stream.traces

    factors, large_ws, data_in, npts_in, delta_in, sampling_rate_out, fftlens = (
        [], [], [], [], [], [], [])
    for trace in traces:
        # Prep the data
        trace, factor = prep_for_resample(
            trace, sampling_rate=sampling_rate, no_filter=no_filter,
            strict_length=strict_length)


        # Get the window first
        fftlen = next_fast_len(trace.stats.npts)
        large_w = get_resample_window(window=window, npts=fftlen)

        if target.upper() == "GPU" and HAS_GPU:
            data = trace._gpu_data
            # Move the window to the GPU
            if large_w is not None:
                large_w = cp.asarray(large_w)
        else:
            data = trace.data

        factors.append(factor)
        large_ws.append(large_w)
        data_in.append(data)
        npts_in.append(trace.stats.npts)
        delta_in.append(trace.stats.delta)
        sampling_rate_out.append(sampling_rate)
        fftlens.append(fftlen)

    try:
        data_out = multi_resample(
            data=data_in, npts_in=npts_in, fftlen=fftlens, delta_in=delta_in,
            factor=factors, sampling_rate_out=sampling_rate_out,
            large_w=large_ws, target=target, max_workers=max_workers)
    except Exception as e:
        print(e)
        for tr in stream:
            tr.resample(sampling_rate=sampling_rate, window=window,
                        target=target, no_filter=no_filter,
                        strict_length=strict_length)

    for i, trace in enumerate(traces):
        if target.upper() == "GPU" and HAS_GPU:
            trace._gpu_data = data_out[i]
        else:
            trace.data = data_out[i]
        trace.stats.sampling_rate = sampling_rate
    stream.traces = traces

    return stream

# ----------------------------- MONKEY PATCH ------------------------------

# Because we have overloaded the lower trace.resample method this still calls
# the accelerated methods.
# setattr(obspy.Stream, "_obspy_resample", obspy.Stream.resample)
obspy.Stream._obspy_resample = _obspy_resample
obspy.Stream.resample = stream_resample


if __name__ == "__main__":
    import doctest

    doctest.testmod()
