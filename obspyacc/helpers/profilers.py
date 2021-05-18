"""
Simple profiling helpers

"""

import time


class Timer:
    """ Simple timing context manager. """
    def __init__(self, name: str = None, verbose: bool = True):
        self._starttime, self._endtime = None, None
        self.name = name or "Timed function"
        self.verbose = verbose

    @property
    def time(self):
        if self._starttime and self._endtime:
            return self._endtime - self._starttime
        return None

    def __enter__(self):
        self._starttime = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._endtime = time.perf_counter()
        if self.verbose:
            print(f"{self.name} took {self.time:.6f} seconds")


def measure_time(func):
    """
    Simple decorator for timing a function
    """
    def wrapper(*args, **kwargs):
        with Timer(func.__name__):
            res = func(*args, **kwargs)
        return res

    return wrapper


if __name__ == "__main__":
    import doctest

    doctest.testmod()
