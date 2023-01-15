from abc import abstractmethod
from logging.handlers import RotatingFileHandler
from os import remove
from threading import Thread
from time import sleep
from traceback import extract_stack, walk_stack
import logging
from typing import Literal

available_log_levels = Literal[
    "PROD", "NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
]


class Logger:
    """Abstract class to implement a configurable logger to a specific class

    The logger automatically creates a log file named like the class implementing this class

    :param debug: bool, default False

    :param log: str, default "PROD"
        Level of the logger among :
        PROD: Deactivate the logger
        NOTSET: Level 0
        DEBUG: Level 10
        INFO: Level 20
        WARNING: Level 30
        ERROR: Level 40
        CRITICAL: Level 50
    :param log_stdout: bool, default False
        Print the logs if enabled, however in both cases save the logs in the log file
    :param keep_log: bool, default False
        Overwrite the previous log file if False
    """

    @abstractmethod
    def _raiser(self, exception: str):
        raise NotImplementedError

    def __init__(
        self,
        debug=False,
        log: available_log_levels = "PROD",
        log_stdout=False,
        keep_log=False,
    ):
        self._DEBUG = debug
        self._LOG = log
        self._log_stdout = log_stdout
        self._keep_log = keep_log

        if self._LOG != "PROD":
            frame = None
            for frame, _ in walk_stack(None):
                variable_names = frame.f_code.co_varnames
                if variable_names == ():
                    break
                if frame.f_locals[variable_names[0]] not in (self, self.__class__):
                    break
                    # if the frame is inside a method of this instance,
                    # the first argument usually contains either the instance
                    # or its class, we want to find the first frame, where
                    # this is not the case
            else:
                self._raiser("No suitable outer frame found.")
            self._outer_frame = frame
            self.creation_module = frame.f_globals["__name__"]
            (
                self.creation_file,
                self.creation_line,
                self.creation_function,
                self.creation_text,
            ) = extract_stack(frame, 1)[0]
            self._creation_name = self.creation_text.split("=")[0].strip()
            super().__init__()
            Thread(target=self._check_existence_after_creation).start()
            try:
                if not keep_log:
                    remove(self._creation_name + ".log")
            except OSError:
                pass
            self._logger = logging.getLogger()
            self._logger.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                "%(asctime)s :: %(levelname)s :: %(message)s", "%d: %H:%M:%S"
            )
            file_handler = RotatingFileHandler(
                self._creation_name + ".log",
                "a",
                1000000,
                1,
                encoding="utf-8",
            )
            log_type = {
                "NOTSET": logging.NOTSET,
                "DEBUG": logging.DEBUG,
                "INFO": logging.INFO,
                "WARNING": logging.WARNING,
                "ERROR": logging.ERROR,
                "CRITICAL": logging.CRITICAL,
            }[self._LOG]
            file_handler.setLevel(log_type)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)
            # creation of a second handler which redirect every log of self.logging on the console
            if log_stdout:
                stream_handler = logging.StreamHandler()
                stream_handler.setLevel(log_type)
                self._logger.addHandler(stream_handler)
            self._log("Initialization of a new object of this instance", log="INFO")

    def _log(self, message, condition: bool = True, log="DEBUG"):
        """
        Create a log message
        :param message: str
            Content of the log message
        :param condition: bool, default True
            Condition to create the log
        :param log: str
            Log level (PROD, NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL)
        :return: None
        """

        if condition and self._LOG != "PROD":
            logging_levels = {
                "NOTSET": logging.NOTSET,
                "DEBUG": logging.DEBUG,
                "INFO": logging.INFO,
                "WARNING": logging.WARNING,
                "ERROR": logging.ERROR,
                "CRITICAL": logging.CRITICAL,
            }[log]
            self._logger.log(logging_levels, message)
            logging.shutdown()

    def _check_existence_after_creation(self):
        """
        Check if the log file has been created, else raise an error
        :return: None
        """

        while self._outer_frame.f_lineno == self.creation_line:
            sleep(0.01)
        # this is executed as soon as the line number changes
        # now we can be sure the instance was actually created

        def error():
            self._raiser(
                "The creation name hasn't be found, the "
                "probable cause may be that you declared "
                "several variables at the same time."
            )

        nameparts = self._creation_name.split(".")
        var = None
        try:
            var = self._outer_frame.f_locals[nameparts[0]]
        except KeyError:
            error()
        finally:
            del self._outer_frame
        # make sure we have no permanent inter frame reference
        # which could hinder garbage collection
        try:
            for name in nameparts[1:]:
                var = getattr(var, name)
        except AttributeError:
            error()
        if var is not self:
            error()
