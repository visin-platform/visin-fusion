"""Reporting to Visin, through the ``visin`` package.

Every script in this project reports through the functions here:

    stages/train/*.py       start_training_run()     the run; the training engine logs each epoch to it
    stages/test/*.py        report_test_results()    a test result on the tested checkpoint's epoch
    stages/benchmark/*.py    report_benchmark()       a benchmark on the measured checkpoint's epoch
    stages/visualize/*.py   attach_to_training() and visualization_uploader.queue_visualizations()

Configuration comes from the environment, or from ``integrations/.env``:

    VISIN_TOKEN    a project token (the project's Settings -> API Tokens in Visin)
    VISIN_URL      the Visin API; defaults to this project's deployment, below
    VISIN_MODE     ``offline`` on a node with no route to Visin; send later with ``visin sync``

With no ``VISIN_TOKEN`` nothing is reported and every script runs as before.
``visin check --write`` tests the setup from the command line.

Training creates the run. The test, benchmark and visualization scripts run
afterwards as separate processes and report into that run, finding it by the
training UUID in the epoch logs; they never change its status, so a crashed
test script cannot mark a finished training as failed.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import sys
import uuid
from typing import Any

from dotenv import load_dotenv

try:
    import visin
except ImportError as exc:  # pragma: no cover - an environment problem, stated plainly
    raise ImportError(
        "visin-fusion reports to Visin through the visin package: pip install visin"
    ) from exc

from utils.helpers import get_model_path, get_training_uuid_from_logs

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# The deployment this project reports to. The visin package has no default
# address, on purpose; this project keeps one, so a job that sets only
# VISIN_TOKEN goes on reporting where it always has.
DEFAULT_VISIN_URL = "https://vision-api.visin.eu"
if os.getenv("VISIN_TOKEN") and not (os.getenv("VISIN_URL") or os.getenv("VISIN_API_URL")):
    os.environ["VISIN_URL"] = DEFAULT_VISIN_URL

# This project does not configure logging; show visin's lines like the scripts'
# own prints: which run, what was sent, what failed.
if not logging.getLogger().handlers:
    visin.enable_console_logging(stream=sys.stdout)

# The UUID an epoch of a run always has. Checkpoint file names carry it
# (epoch_{n}_{uuid}.pth), which is how the later scripts find the epoch.
epoch_uuid_for = visin.epoch_uuid_for

_CHECKPOINT_NAME = re.compile(r"epoch_(\d+)_([0-9a-fA-F-]{36})\.pth$")


def parse_checkpoint_name(path: str) -> tuple[int | None, str | None]:
    """The epoch number and epoch UUID in a checkpoint's file name, if it has them."""
    match = _CHECKPOINT_NAME.search(os.path.basename(path))
    if not match:
        return None, None
    return int(match.group(1)), match.group(2)


def training_uuid_for_epoch(log_dir: str, epoch_uuid: str) -> str | None:
    """The training an epoch belongs to, from its log file (logs/.../epochs/epoch_{n}_{uuid}.json)."""
    for path in glob.glob(os.path.join(log_dir, "epochs", f"epoch_*_{epoch_uuid}.json")):
        try:
            with open(path) as handle:
                training_uuid = json.load(handle).get("training_uuid")
        except (OSError, ValueError):
            continue
        if training_uuid:
            return training_uuid
    return None


def training_uuid_for_checkpoint(checkpoint_path: str) -> str | None:
    """The training a checkpoint belongs to, from its epoch's log beside the checkpoints directory."""
    _, epoch_uuid = parse_checkpoint_name(checkpoint_path)
    if not epoch_uuid:
        return None
    log_dir = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_path)))
    return training_uuid_for_epoch(log_dir, epoch_uuid)


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
    general = config["General"]
    fresh = general.get("create_new_training") or general.get("transfer_learning") or general.get("reset_lr")
    training_uuid = None
    if general.get("resume_training") and not fresh:
        # The run of the checkpoint being resumed, which the logs directory may
        # share with an earlier run of the same config.
        checkpoint = get_model_path(config)
        training_uuid = training_uuid_for_checkpoint(checkpoint) if checkpoint else None
        training_uuid = training_uuid or get_training_uuid_from_logs(config["Log"]["logdir"])[0]
        if not training_uuid:
            print("No earlier run in the epoch logs; starting a new one")
    resumed_locally = training_uuid is not None
    training_uuid = training_uuid or str(uuid.uuid4())

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
    print(f"Training UUID: {training_uuid} ({'resumed' if resumed else 'new'} run; Visin: {run.mode})")
    return run, training_uuid


def attach_to_training(log_dir: str, *, epoch_uuid: str | None = None) -> visin.Run:
    """The run a post-training script reports into, found through the epoch logs.

    With ``epoch_uuid`` (the tested checkpoint's), the run that epoch belongs
    to; otherwise the run of the newest epoch log. Reporting only: the run's
    status stays whatever training left it at. Returns a run that reports
    nothing when the logs name no training.
    """
    training_uuid = training_uuid_for_epoch(log_dir, epoch_uuid) if epoch_uuid else None
    training_uuid = training_uuid or get_training_uuid_from_logs(log_dir)[0]
    if not training_uuid:
        print(f"Visin: no training UUID in the epoch logs under {log_dir}; not reporting")
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
        print("Visin: the benchmark names no training run; not reporting it")
        return
    with visin.Run.attach(training_uuid, mark_status=False) as run:
        run.log_benchmark(results, system_info, epoch=epoch, epoch_uuid=epoch_uuid)
