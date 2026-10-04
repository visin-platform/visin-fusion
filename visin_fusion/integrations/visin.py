"""Reporting to Visin, through the ``visin`` package.

Every script in this project reports through the functions here:

    engine/stages/train       start_training_run()     the run; the training engine logs each epoch to it
    engine/stages/test        report_evaluation()      the tested checkpoint's results on a Visin suite, if one is set
                              report_test_results()    else a test result (an evaluation with no suite) on its epoch
    engine/stages/benchmark   report_benchmark()       a benchmark on the measured checkpoint's epoch
    engine/stages/visualize   attach_to_training() and visualization_uploader.queue_visualizations()

Configuration comes from exported variables, ``VISIN_ENV_FILE``, or the caller's ``.env``:

    VISIN_TOKEN    a pipeline key (the project's Settings -> Pipeline keys in Visin)
    VISIN_URL      the Visin API address (required when VISIN_TOKEN is set)
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

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from visin_fusion.integrations.settings import load_environment

try:
    import visin
except ImportError as exc:  # pragma: no cover - an environment problem, stated plainly
    raise ImportError("visin-fusion reports to Visin through the visin package: pip install visin") from exc

from visin_fusion._version import __version__
from visin_fusion.engine.epoch_ids import (
    training_uuid_for_epoch,
    training_uuid_for_run,
)
from visin_fusion.providers import HUB
from visin_fusion.utils.helpers import get_training_uuid_from_logs

logger = logging.getLogger(__name__)

_environment_ready = False


def prepare_environment() -> None:
    """Load the caller's key file and require an explicit API address before reporting.

    Importing this module changes nothing; the functions below that talk to Visin call this first, so the
    environment is set up when a stage starts reporting, not when something imports the integration.
    """
    global _environment_ready
    if _environment_ready:
        return
    load_environment()
    if os.getenv("VISIN_TOKEN") and not (os.getenv("VISIN_URL") or os.getenv("VISIN_API_URL")):
        raise ValueError(
            "VISIN_TOKEN requires VISIN_URL (or VISIN_API_URL); set it in VISIN_ENV_FILE or the environment"
        )
    _environment_ready = True


# The UUID an epoch of a run always has. Checkpoint file names carry it
# (epoch_{n}_{uuid}.pth), which is how the later scripts find the epoch.
epoch_uuid_for = visin.epoch_uuid_for


HUB_SNAPSHOT = re.compile(r"datasets--(?P<org>[^/\\]+)--(?P<name>[^/\\]+)[/\\]snapshots[/\\](?P<commit>[0-9a-f]{40})")


def _dataset_reference(root: str | None, name: str) -> str | dict[str, Any]:
    """Record where the data came from: a Hub snapshot's repo and commit, or a Visin download's marker.

    A ``hf:`` root resolves to a folder in the Hub cache, whose path names the repo and the exact commit it
    holds. A Visin download leaves a marker with the dataset's id and revision. Anything else, or a marker
    that cannot be read, falls back to the dataset label.
    """
    if not root:
        return name
    snapshot = HUB_SNAPSHOT.search(str(root))
    if snapshot:
        return {"source": HUB, "name": f"{snapshot['org']}/{snapshot['name']}", "revision": snapshot["commit"]}
    for directory in (Path(root), *Path(root).parents):
        marker = directory / ".visin-dataset.json"
        if marker.is_file():
            try:
                data = json.loads(marker.read_text())
                if not isinstance(data, dict) or any(
                    not isinstance(data.get(field), str) or not data[field].strip() for field in ("id", "name")
                ):
                    raise ValueError("dataset marker needs an id and name")
                if data.get("revision") is not None and not isinstance(data["revision"], str):
                    raise ValueError("dataset marker revision must be a string")
            except (OSError, ValueError) as exc:
                logger.warning("Cannot read dataset provenance from %s (%s); using label %s", marker, exc, name)
                return name
            return {
                "source": "visin",
                "id": data["id"],
                "name": data["name"],
                **({"revision": data["revision"]} if data.get("revision") else {}),
            }
    return name


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
        dataset=_dataset_reference(config["Dataset"].get("dataset_root"), dataset),
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


def report_evaluation(
    config: dict[str, Any],
    *,
    epoch: int,
    epoch_uuid: str | None,
    results: dict[str, Any],
    test_uuid: str | None,
    checkpoint_path: str | None,
    sample_counts: dict[str, int] | None,
    splits: dict[str, str] | None = None,
) -> bool:
    """Record a tested checkpoint's results as an evaluation on ``General.suite``, when one is set.

    A test result says what an epoch scored; this says what the checkpoint scored on a suite, which is what Visin
    can rank, and Visin keeps one record of a test, so the caller sends a plain test result only when this returns
    false. The checkpoint is named by the digest of its weights and the evaluation by ``test_uuid``,
    so repeating the upload is harmless. ``sample_counts`` is how many frames each test set scored: a suite pins
    those, and a test set that was skipped because it listed no frames leaves the result unranked, with the reason
    on the evaluation.

    To be ranked the evaluation also says what ran. ``General.suite_file`` is the suite file the test stage was run
    against: Visin computes its digest and it is sent as the protocol that ran (``suite`` defaults to the file's
    ``slug@version``). ``splits`` maps each test set to its frame-list file, and the digest of those lists is sent as
    the data that was scored, so a changed test set is refused as a different measurement even when the frame counts
    agree. Without ``suite_file`` the result is still ranked, as reported rather than observed, and the log says so.

    Returns whether an evaluation was recorded or kept for ``visin sync``. Refused evaluations (an unknown suite, no
    access) are logged and do not fail the test stage: the results are already saved locally, and the caller sends
    them as a plain test result instead. Needs a ``visin`` package that has ``evaluate``.
    """
    general = config.get("General") or {}
    suite_file = general.get("suite_file")
    suite = general.get("suite") or (_suite_of(suite_file) if suite_file else None)
    if not suite:
        return False
    if not hasattr(visin, "evaluate"):
        logger.warning("General.suite is set, but this visin cannot record evaluations; pip install -U visin")
        return False
    if not checkpoint_path:
        logger.warning("General.suite is set, but the tested checkpoint is not known; no evaluation recorded")
        return False
    if not suite_file:
        logger.warning(
            "General.suite_file is not set: the evaluation cannot say which protocol or data ran, so it is ranked "
            "as reported rather than observed"
        )
    prepare_environment()
    training_uuid = training_uuid_for_epoch(config["Log"]["logdir"], epoch_uuid) if epoch_uuid else None
    try:
        evaluation = visin.evaluate(
            results,
            suite=suite,
            checkpoint=visin.local_checkpoint(checkpoint_path, f"{config['CLI']['backbone']}-epoch-{epoch}"),
            sample_counts=sample_counts,
            run=training_uuid,
            epoch=epoch,
            epoch_uuid=epoch_uuid,
            evaluator={"package": "visin-fusion", "version": __version__},
            data=_scored_data(config, suite_file, splits),
            protocol=suite_file or None,
            uuid=test_uuid,
        )
    except visin.VisinError as exc:
        logger.warning("Visin did not record the evaluation on %s: %s", suite, exc)
        return False
    logger.info("Visin evaluation on %s: %s", suite, evaluation.verdict or "kept to send with `visin sync`")
    return bool(evaluation.stored or evaluation.queued)


def _read_suite(path: str) -> dict[str, Any] | None:
    """The suite file as a dict, or ``None`` when it cannot be read or this visin cannot read suites."""
    if not hasattr(visin, "load_suite"):
        return None
    try:
        return visin.load_suite(path)
    except visin.VisinError as exc:
        logger.warning("General.suite_file %s could not be read: %s", path, exc)
        return None


def _suite_of(path: str) -> str | None:
    """``slug@version`` of a suite file, or ``None`` when it cannot be read."""
    loaded = _read_suite(path)
    if not loaded:
        return None
    return f"{loaded.get('slug')}@{loaded.get('version')}" if loaded.get("slug") and loaded.get("version") else None


def _manifest_evidence(config: dict[str, Any], splits: dict[str, str] | None) -> dict[str, str] | None:
    """The digest of every test set's frame list, by test set name: the data of a suite pinned by a manifest."""
    if not splits or not hasattr(visin, "manifest_digest"):
        return None
    try:
        conditions = {name: visin.read_split(path) for name, path in splits.items()}
    except visin.VisinError as exc:
        logger.warning("The test sets' frame lists could not be read, so the data is not reported: %s", exc)
        return None
    return {"kind": "external", "manifestSha256": visin.manifest_digest(conditions)}


def _hub_evidence(config: dict[str, Any], splits: dict[str, str] | None) -> dict[str, str] | None:
    """The Hub repo and full commit the dataset root resolved to: the data of a suite pinned to a Hub dataset."""
    reference = _dataset_reference(config.get("Dataset", {}).get("dataset_root"), "")
    if isinstance(reference, dict) and reference.get("source") == HUB and reference.get("revision"):
        return {"kind": HUB, "repo": reference["name"], "commit": reference["revision"]}
    return None


_DATA_EVIDENCE = {"external": _manifest_evidence, HUB: _hub_evidence}


def _scored_data(
    config: dict[str, Any], suite_file: str | None, splits: dict[str, str] | None
) -> dict[str, str] | None:
    """What was scored, in the shape the suite pins it: read from the suite file, never assumed to be a frame list.

    ``_DATA_EVIDENCE`` says what fusion can report for each kind of data a suite can pin; a new kind adds one entry.
    A kind fusion cannot observe is absent (a Visin dataset's archive digest is not known to the evaluator), so the
    evaluation is ``reported`` rather than refused as a different measurement.
    """
    loaded = _read_suite(suite_file) if suite_file else None
    protocol = (loaded or {}).get("protocol") or loaded or {}
    gather = _DATA_EVIDENCE.get((protocol.get("data") or {}).get("kind"))
    return gather(config, splits) if gather else None


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
