"""Reporting to Visin, through the ``visin`` package.

Every script in this project reports through the functions here:

    engine/stages/train       start_training_run()     the run; the training engine logs each epoch to it
    engine/stages/test        report_test_results()    a test result on the tested checkpoint's epoch
    engine/stages/benchmark   report_benchmark()       a benchmark on the measured checkpoint's epoch
    engine/stages/visualize   attach_to_training() and visualization_uploader.queue_visualizations()

Configuration comes from exported variables, ``VISIN_ENV_FILE``, or the caller's ``.env``:

    VISIN_TOKEN    a pipeline key (the project's Settings -> Pipeline keys in Visin)
    VISIN_URL      the Visin API; defaults to this project's deployment, below
    VISIN_MODE     ``offline`` on a node with no route to Visin; send later with ``visin sync``
    VISIN_ENV_FILE optional path to an env file outside the installed package

With no ``VISIN_TOKEN`` nothing is reported and every script runs as before.
``visin check --write`` tests the setup from the command line.

Training creates the run. The test, benchmark and visualization scripts run
afterwards as separate processes and report into that run, finding it by the
training UUID in the epoch logs; they never change its status, so a crashed
test script cannot mark a finished training as failed.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from visin_fusion.integrations.settings import load_environment

try:
    import visin
except ImportError as exc:  # pragma: no cover - an environment problem, stated plainly
    raise ImportError("visin-fusion reports to Visin through the visin package: pip install visin") from exc

from visin_fusion.engine.epoch_ids import (
    training_uuid_for_epoch,
    training_uuid_for_run,
)
from visin_fusion.utils.helpers import get_training_uuid_from_logs

logger = logging.getLogger(__name__)

# The deployment this project reports to. The visin package has no default
# address, on purpose; this project keeps one, so a job that sets only
# VISIN_TOKEN goes on reporting where it always has.
DEFAULT_VISIN_URL = "https://vision-api.visin.eu"

_environment_ready = False


def prepare_environment() -> None:
    """Load the caller's key file and default ``VISIN_URL``, once, before the first report.

    Importing this module changes nothing; the functions below that talk to Visin call this first, so the
    environment is set up when a stage starts reporting, not when something imports the integration.
    """
    global _environment_ready
    if _environment_ready:
        return
    load_environment()
    if os.getenv("VISIN_TOKEN") and not (os.getenv("VISIN_URL") or os.getenv("VISIN_API_URL")):
        os.environ["VISIN_URL"] = DEFAULT_VISIN_URL
    _environment_ready = True


# The UUID an epoch of a run always has. Checkpoint file names carry it
# (epoch_{n}_{uuid}.pth), which is how the later scripts find the epoch.
epoch_uuid_for = visin.epoch_uuid_for


def start_training_run(config: dict[str, Any], *, model: str) -> tuple[visin.Run, str]:
    """Register this training with Visin, or pick up the run it is resuming.

    Returns the run and the training UUID. The UUID is this project's own
    record (epoch logs and checkpoint names carry it), so it exists whether or
    not Visin is configured.

    - A fresh training gets a new UUID and a new run, with the config attached.
    - A resumed one (General.resume_training) takes its UUID from the epoch logs
      in Log.logdir and reports into the same run; nothing is created twice.
    - General.create_new_training or General.transfer_learning forces a fresh run,
      and so does General.reset_lr: its epochs start again from 0, and a run
      keeps the first values recorded for an epoch number.
    """
    prepare_environment()
    training_uuid, resumed_locally = training_uuid_for_run(config)

    dataset = config["Dataset"]["name"]
    run = visin.init(
        config.get("Summary") or f"Training {model} on {dataset} dataset",
        training_uuid=training_uuid,
        model=model,
        dataset=dataset,
        tags=config.get("tags") or None,
    )
    # Visin knows whether it already had the run; offline, only the logs do.
    resumed = run.resumed if run.resumed is not None else resumed_locally
    if not resumed:
        run.log_config(config, name=model)
    logger.info("Training UUID: %s (%s run; Visin: %s)", training_uuid, ("resumed" if resumed else "new"), run.mode)
    return run, training_uuid


def attach_to_training(log_dir: str, *, epoch_uuid: str | None = None) -> visin.Run:
    """The run a post-training script reports into, found through the epoch logs.

    With ``epoch_uuid`` (the tested checkpoint's), the run that epoch belongs
    to; otherwise the run of the newest epoch log. Reporting only: the run's
    status stays whatever training left it at. Returns a run that reports
    nothing when the logs name no training.
    """
    prepare_environment()
    training_uuid = training_uuid_for_epoch(log_dir, epoch_uuid) if epoch_uuid else None
    training_uuid = training_uuid or get_training_uuid_from_logs(log_dir)[0]
    if not training_uuid:
        logger.info("Visin: no training UUID in the epoch logs under %s; not reporting", log_dir)
        return visin.Run.disabled()
    return visin.Run.attach(training_uuid, mark_status=False)


def report_test_results(
    config: dict[str, Any],
    epoch: int,
    epoch_uuid: str | None,
    test_results: dict[str, Any],
    *,
    test_uuid: str | None = None,
) -> None:
    """Send one checkpoint's test results, as a test result on its epoch.

    ``test_uuid`` is the id the local results file records, so a repeat of the
    same upload is recognised rather than stored twice.
    """
    with attach_to_training(config["Log"]["logdir"], epoch_uuid=epoch_uuid) as run:
        run.log_test_results(epoch, test_results, epoch_uuid=epoch_uuid, test_uuid=test_uuid)


def report_benchmark(
    results: list[dict[str, Any]],
    system_info: dict[str, Any],
    *,
    training_uuid: str | None,
    epoch: int | None = None,
    epoch_uuid: str | None = None,
) -> None:
    """Send benchmark measurements, linked to the run and checkpoint they measured."""
    if not training_uuid:
        logger.info("Visin: the benchmark names no training run; not reporting it")
        return
    prepare_environment()
    with visin.Run.attach(training_uuid, mark_status=False) as run:
        run.log_benchmark(results, system_info, epoch=epoch, epoch_uuid=epoch_uuid)
