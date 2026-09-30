"""visin_fusion/logging_setup.py: bare messages on stdout, levels named, no duplicate handlers."""

import contextlib
import logging

from visin_fusion.logging_setup import configure_logging


@contextlib.contextmanager
def bare_root():
    root = logging.getLogger()
    saved, level = root.handlers[:], root.level
    root.handlers = []
    try:
        yield root
    finally:
        root.handlers, root.level = saved, level


def test_info_is_printed_bare_and_warnings_name_their_level(capsys):
    with bare_root():
        configure_logging()
        logger = logging.getLogger("visin_fusion.test")
        logger.info("Training")
        logger.warning("No annotation")
    assert capsys.readouterr().out.splitlines() == ["Training", "WARNING: No annotation"]


def test_configuring_twice_adds_one_handler():
    with bare_root() as root:
        configure_logging()
        configure_logging()
        assert len(root.handlers) == 1


def test_existing_handlers_are_kept():
    with bare_root() as root:
        handler = logging.NullHandler()
        root.addHandler(handler)
        configure_logging()
        assert root.handlers == [handler]


def test_environment_sets_the_threshold(monkeypatch):
    monkeypatch.setenv("VISIN_FUSION_LOG_LEVEL", "ERROR")
    with bare_root() as root:
        configure_logging()
        assert root.level == logging.ERROR
