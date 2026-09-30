"""Console logging for the command-line entry points.

Library code logs through ``logging.getLogger(__name__)`` and never configures handlers. Each
entry point (``visin-fusion`` and the stage modules) calls :func:`configure_logging` once, which
sends bare messages to stdout so job logs read as they always have. Setting
``VISIN_FUSION_LOG_LEVEL`` (for example ``WARNING``) changes the threshold.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys


class _ConsoleFormatter(logging.Formatter):
    """Bare messages, except that warnings and errors name their level."""

    def format(self, record: logging.LogRecord) -> str:
        """Prefix records of WARNING and above with their level name."""
        message = super().format(record)
        return f"{record.levelname}: {message}" if record.levelno >= logging.WARNING else message


def configure_logging(level: int | str | None = None) -> None:
    """Route log records to stdout unless the root logger already has handlers.

    A host application or test harness that installed its own handlers keeps them; only the
    threshold is applied. Calling this twice adds no second handler. When the ``visin`` package is
    installed, its run and delivery messages go to stdout as well.
    """
    threshold = level or os.environ.get("VISIN_FUSION_LOG_LEVEL") or logging.INFO
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_ConsoleFormatter("%(message)s"))
        root.addHandler(handler)
    root.setLevel(threshold)
    if importlib.util.find_spec("visin") is not None:
        import visin

        visin.enable_console_logging(level=threshold, stream=sys.stdout)
