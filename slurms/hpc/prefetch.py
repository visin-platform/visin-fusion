"""Build the config's model once, so its pretrained backbone is in the cache before a node with no internet needs it."""

from __future__ import annotations

import logging
import sys

from visin_fusion.config.config import prepare_config
from visin_fusion.logging_setup import configure_logging
from visin_fusion.models.registry import from_config

logger = logging.getLogger(__name__)


def main() -> None:
    """Download the backbone for ``<template-config> <dataset-root>`` into ``HF_HOME`` and ``TORCH_HOME``."""
    configure_logging()
    template, root = sys.argv[1:3]
    config = prepare_config({"extends": template, "Dataset": {"dataset_root": root}}, ".")
    model = from_config(config)
    logger.info("Built %s; any pretrained weights are now cached", type(model).__name__)


if __name__ == "__main__":
    main()
