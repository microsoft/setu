# Adapted from
# https://github.com/skypilot-org/skypilot/blob/86dc0f6283a335e4aa37b3c10716f90999f48ab6/sky/sky_logging.py
"""Logging configuration for Setu."""

import logging
import os

from rich.logging import RichHandler


class NewLineFormatter(logging.Formatter):
    """Adds logging prefix to newlines to align multi-line messages."""

    def __init__(self, fmt, datefmt=None):
        logging.Formatter.__init__(self, fmt, datefmt)

    def format(self, record):
        msg = logging.Formatter.format(self, record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg


_root_logger = logging.getLogger("setu")
_default_handler = None


def _setup_logger():
    _root_logger.setLevel(logging.DEBUG)
    global _default_handler
    if _default_handler is None:
        _default_handler = RichHandler(
            rich_tracebacks=False,
            show_time=True,
            omit_repeated_times=False,
            show_level=True,
            show_path=False,
        )

        log_level = os.getenv("SETU_LOG_LEVEL", "INFO")
        _default_handler.setLevel(log_level)
        _root_logger.addHandler(_default_handler)
    # Setting this will avoid the message
    # being propagated to the parent logger.
    _root_logger.propagate = False


# The logger is initialized when the module is imported.
# This is thread-safe as the module is only imported once,
# guaranteed by the Python GIL.
_setup_logger()


def init_logger(name: str):
    return logging.getLogger(name)
