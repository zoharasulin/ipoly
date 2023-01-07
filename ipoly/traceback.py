from sys import exc_info
from IPython import get_ipython
from typing import Type
import sys

_ipython = get_ipython()
_SHOW_TRACEBACK = True
if _ipython:
    _TRACEBACK_FUNCTION = _ipython.showtraceback


def _ipython_hide_traceback(*args, **kwargs):
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


def _python_exception_handler(
    exception_type, exception, traceback, debug_hook=sys.excepthook
):
    global _SHOW_TRACEBACK
    if _SHOW_TRACEBACK:
        debug_hook(exception_type, exception, traceback)
    else:
        _SHOW_TRACEBACK = True
        print("%s: %s" % (exception_type.__name__, exception))


if _ipython:
    _ipython.showtraceback = _ipython_hide_traceback
else:
    sys.excepthook = _python_exception_handler


def raiser(
    message: str,
    type: Type[BaseException] = Exception,
    traceback: bool = False,
):
    """Raise an exception with or without the traceback.

    Args:
        message : The message of the exception.
        type : The type of the exception raised.
        traceback : The traceback is printed if False.

    """

    global _SHOW_TRACEBACK
    _SHOW_TRACEBACK = traceback
    raise type(message)
