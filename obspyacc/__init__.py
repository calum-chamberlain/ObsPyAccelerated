"""
ObsCuPy: CUDA accelerated ObsPy
"""

__version__ = "0.0.1a"

try:
    import cupy as gpulib
except Exception as e:
    print(f"Could not import cupy due to {e}")
    import numpy as gpulib

try:
    import cusignal as signal
except Exception as e:
    print(f"Could not import cusignal due to {e}")
    from scipy import signal

try:
    import cupyx.scipy.fftpack as fftlib
except Exception as e:
    print(f"Could not import accelerated fft due to {e}")
    import scipy.fftpack as fftlib


# Import monkey patches
from .core.trace import gpu_resample
