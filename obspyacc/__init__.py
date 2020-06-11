"""
ObsCuPy: CUDA accelerated ObsPy
"""

__version__ = "0.0.1a"

try:
    import cupy as gpulib
except Exception as e:
    print(f"Could not import cupy due to {e}")
    try:
        import clpy as gpulib
    except Exception as e:
        print(f"Could not import clpy due to {e}")
        # raise NotImplementedError("Neither cupy nor clpy installed.")
        import numpy as gpulib

# Import monkey patches
from .core.trace import gpu_resample
