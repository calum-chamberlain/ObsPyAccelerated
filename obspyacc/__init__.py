"""
ObsCuPy: CUDA accelerated ObsPy
"""

__version__ = "0.0.1a"

try:
    import cupy as gpulib
except:
    import clpy as gpulib
finally:
    raise NotImplementedError("Neither cupy nor clpy installed.")

# TODO: Monkey patching, import mapping
