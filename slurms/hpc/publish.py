"""Publish a run's best checkpoint to the Hugging Face Hub and link it to the run in Visin."""

from __future__ import annotations

import argparse
import logging
import sys

from visin_fusion.config.config import load_config
from visin_fusion.hub import push_best_checkpoint
from visin_fusion.integrations.visin import attach_to_training
from visin_fusion.logging_setup import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Upload the best checkpoint under the config's ``Log.logdir`` to ``--repo``; exit 1 when that fails."""
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--repo", required=True, help="the Hub model repo, org/name")
    parser.add_argument("--public", action="store_true", help="create the repo public (default: private)")
    args = parser.parse_args()

    config = load_config(args.config)
    config["General"]["hub_repo"] = args.repo
    config["General"]["hub_private"] = not args.public
    with attach_to_training(config["Log"]["logdir"]):
        commit = push_best_checkpoint(config)
    if not commit:
        logger.error("The upload to %s did not complete", args.repo)
        sys.exit(1)
    logger.info("https://huggingface.co/%s/tree/%s", args.repo, commit)


if __name__ == "__main__":
    main()
