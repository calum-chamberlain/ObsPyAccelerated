"""
Methods for monkey patching ObsPy Stream methods.
"""

from typing import Union, Callable

import numpy as np

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

@obspy_docs(method_or_function=obspy.Stream.resample)
@skip_if_no_data
def stream_resample(
    stream: obspy.core.Stream,
    sampling_rate: float,
    window: Union[str, Callable, np.ndarray] = "hanning",
    no_filter: bool = True,
    strict_length: bool = False,
    target: str = "CPU",
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
    assert target.upper() in ("CPU", "GPU"), f"Target {target} not supported"
    if isinstance(stream, obspy.core.Trace):
        traces = [stream]
    else:
        traces = stream.traces

    factors, large_ws, data_in, npts_in, delta_in, sampling_rate_out = (
        [], [], [], [], [], [])
    for trace in traces:
        # Prep the data
        trace, factor = prep_for_resample(
            trace, sampling_rate=sampling_rate, no_filter=no_filter,
            strict_length=strict_length)


        # Get the window first
        large_w = get_resample_window(window=window, npts=trace.stats.npts)

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

    data_out = multi_resample(
        data=data_in, npts_in=npts_in, delta_in=delta_in,
        factor=factors, sampling_rate_out=sampling_rate_out,
        large_w=large_ws, target=target, max_workers=max_workers)

    for i, trace in enumerate(traces):
        if target.upper() == "GPU" and HAS_GPU:
            trace._gpu_data = data_out[i]
        else:
            trace.data = data_out[i]
        trace.stats.sampling_rate = sampling_rate
    stream.traces = traces

    return stream

# ----------------------------- MONKEY PATCH ------------------------------

setattr(obspy.Stream, "_obspy_resample", obspy.Stream.resample)
obspy.Stream.resample = stream_resample


if __name__ == "__main__":
    import doctest

    doctest.testmod()
