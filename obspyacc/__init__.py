"""
ObsPyAccelerated: CUDA accelerated ObsPy
"""

__version__ = "0.0.1a"

HAS_GPU = True

try:
    import cupy as gpulib
except Exception as e:
    print(f"Could not import cupy due to {e}")
    import numpy as gpulib
    HAS_GPU = False

try:
    import cupyx.scipy.fftpack as fftlib
except Exception as e:
    print(f"Could not import accelerated fft due to {e}")
    # TODO: We could import pyfftw here.
    import numpy.fft as fftlib
    HAS_GPU = False


# Import monkey patches
from .core.stream import stream_resample
from .core.trace import trace_resample
