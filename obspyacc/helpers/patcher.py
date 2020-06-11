"""
Helpers for the obspyacc package
"""

import textwrap

from typing import Callable


def obspy_docs(method_or_function: Callable, replace_doc: str = "obspy_docs"):
    """
    Decorator for adding obspy docs to obspyacc.

    Will replace {replace_doc} in the wrapped functions doc-string with the
    obspy doc-string.

    Parameters
    ----------
    method_or_function:
        Obspy method or function to be monkey-patched
    replace_doc:
        String to insert obspy docs at.
    """

    def _wrap(func):
        nonlocal method_or_function

        docstring = func.__doc__
        # Replace {obspy_doc}
        _replace = "{%s}" % replace_doc
        lines = [_ for _ in docstring.split("\n") if _replace in _]
        for line in lines:
            # determine number of spaces used before matching character
            spaces = line.split(_replace)[0]
            # ensure only spaces precede search value
            assert set(spaces) == {" "}
            new = {replace_doc: textwrap.indent(
                textwrap.dedent(method_or_function.__doc__), spaces)}
            docstring = docstring.replace(line, line.format(**new))
        func.__doc__ = docstring
        return func

    return _wrap


if __name__ == "__main__":
    import doctest

    doctest.testmod()
