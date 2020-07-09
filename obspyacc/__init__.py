"""
ObsCuPy: CUDA accelerated ObsPy
"""

__version__ = "0.0.1a"

try:
    import cupy as gpulib
except Exception as e:
    print(f"Could not import cupy due to {e}")
    raise e

# Import monkey patches
from .core.trace import gpu_resample
