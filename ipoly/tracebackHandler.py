from sys import exc_info
from typing import Any

from IPython import InteractiveShell

from ipoly.logger import Logger


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class TracebackHandler(Logger):
    """
    Abstract class aiming to deactivate the Traceback if the error is deliberately raised with the '_raiser' method.
    This class implements the Logger class.
    The Traceback is restored if an error is not raised by the '_raiser' method.
    Each public method of the class implementing 'TracebackHandler' should be decorated by the '_traceback' method to
    take advantage of its features.

    :param debug: bool, default False
    :param kwargs:
    """
    def __init__(self, debug=False, **kwargs):
        _trace = None
        _ipython = None
        _hide_traceback = None
        self._debug = debug
        if not self._debug:
            self._ipython = self._get_ipython()

            def hide_traceback(**_kwargs):
                error_type, value, tb = exc_info()
                value.__cause__ = None  # suppress chained exceptions
                return self._ipython._showtraceback(
                    error_type,
                    value,
                    self._ipython.InteractiveTB.get_exception_only(error_type, value),
                )

            self.hide_traceback = hide_traceback
            if self._ipython is not None:
                self._trace = self._ipython.showtraceback

        super().__init__(**kwargs)

    def _raiser(self, exception: str) -> None:
        self._log(exception, log="CRITICAL")
        if not self._debug and (self._ipython is not None):
            self._ipython.showtraceback = self.hide_traceback
        raise Error(exception)

    @staticmethod
    def _traceback(func) -> Any:
        """
        Decorator method to deactivate temporarily the Traceback and restore it at the end of the decorated method.
        """
        def wrapper(self, *args, **kwargs):
            if not self._DEBUG and self._ipython is not None:
                self._ipython.showtraceback = self._trace
            try:
                return func(self, *args, **kwargs)
            except Error as e:
                raise e
            except BaseException as e:
                self._log(e, log="CRITICAL")
                raise e

        return wrapper

    @staticmethod
    def _get_ipython() -> InteractiveShell:
        from IPython import get_ipython
        return get_ipython()
