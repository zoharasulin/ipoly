from sys import exc_info
from IPython import get_ipython
from typing import Type

_ipython = get_ipython()
_SHOW_TRACEBACK = True
_TRACEBACK_FUNCTION = _ipython.showtraceback


def _hide_traceback(*args, **kwargs):
    global _SHOW_TRACEBACK, _TRACEBACK_FUNCTION
    if _SHOW_TRACEBACK:
        return _TRACEBACK_FUNCTION(*args, **kwargs)
    else:
        _SHOW_TRACEBACK = True
        error_type, value, _ = exc_info()
        value.__cause__ = None  # suppress chained exceptions
        return _ipython._showtraceback(
            error_type,
            value,
            _ipython.InteractiveTB.get_exception_only(error_type, value),
        )


_ipython.showtraceback = _hide_traceback


def raiser(
    message: str,
    exception_type: Type[BaseException] = Exception,
    traceback: bool = True,
):
    global _SHOW_TRACEBACK
    _SHOW_TRACEBACK = traceback
    raise exception_type(message)
